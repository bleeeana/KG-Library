import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import os, datetime

class GraphNN(nn.Module):
    def __init__(self, preprocessor, hidden_dim=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.preprocessor = preprocessor
        self.device = preprocessor.device
        self.num_layers = num_layers

        num_entities = len(preprocessor.entity_id) + 1
        num_relations = len(preprocessor.relation_id) + 1
        input_dim = preprocessor.feature_matrix.size(1)

        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )

        self.skip_weights = nn.Parameter(torch.ones(num_layers))
        self.gamma = nn.Parameter(torch.tensor(10.0))

        self.convs = nn.ModuleList([
            HeteroConv({
                edge_type: GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=4,
                    concat=False,
                    dropout=dropout
                ) for edge_type in preprocessor.hetero_graph.edge_types
            }, aggr='mean') for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.to(self.device)

    def forward(self, graph : HeteroData):
        node_features = self.feature_proj(self.preprocessor.feature_matrix.to(self.device))
        node_embeddings = F.normalize(
            self.entity_embedding(torch.arange(node_features.size(0), device=self.device)),
            p=2, dim=1
        )

        x = {"entity": 0.7 * node_features + 0.5 * node_embeddings}
        x_initial = x.copy()

        for i, conv in enumerate(self.convs):
            x_updated = conv(x, graph.edge_index_dict)
            alpha = torch.sigmoid(self.skip_weights[i])
            x = {
                key: F.leaky_relu(self.dropout(val), 0.2) + alpha * x_initial[key]
                for key, val in x_updated.items()
            }

        return x

    def score_function(self, head, tail, relation):
        return self.gamma - torch.norm(head + relation - tail, p=2, dim=1)

    def get_entity_embedding(self, ids):
        return F.normalize(self.entity_embedding(ids.to(self.device)), p=2, dim=1)

    def get_relation_embedding(self, ids):
        return F.normalize(self.relation_embedding(ids.to(self.device)), p=2, dim=1)


class GraphTrainer:
    def __init__(self, model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10):
        self.model = model
        self.device = model.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.current_epoch = 0
        log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir=log_dir)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=patience, factor=0.5, verbose=True)

        print(f"TensorBoard logs will be saved to: {log_dir}")
        print("To view results, run in terminal:")
        print(f"tensorboard --logdir={os.path.abspath('runs')}")
        print("Then open http://localhost:6006 in your browser")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            all_scores = []
            all_labels = []

            for batch in self.train_loader:
                batch = batch.to(self.device)
                graph = self.model.preprocessor.create_hetero_from_batch(batch)
                output = self.model(graph)

                h = output["entity"][batch.head.squeeze()]
                t = output["entity"][batch.tail.squeeze()]
                r = self.model.get_relation_embedding(batch.edge_attr.squeeze())

                scores = self.model.score_function(h, t, r)
                loss = F.binary_cross_entropy_with_logits(scores, batch.y)

                self.optimizer.zero_grad()
                loss.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"gradients/{name}", param.grad, epoch)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                all_scores.append(scores.detach().cpu())
                all_labels.append(batch.y.cpu())

            train_scores = torch.cat(all_scores)
            train_labels = torch.cat(all_labels)
            train_auc = roc_auc_score(train_labels, train_scores.sigmoid())

            val_auc = self.evaluate(self.val_loader)
            self.scheduler.step(val_auc)

            self.writer.add_scalar("Loss/train", epoch_loss / len(self.train_loader), epoch)
            self.writer.add_scalar("AUC/train", train_auc, epoch)
            self.writer.add_scalar("AUC/val", val_auc, epoch)
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], epoch)


            print(f"Epoch {epoch:03d} | Loss: {epoch_loss / len(self.train_loader):.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

            self.current_epoch += 1

        self.writer.close()

    def evaluate(self, loader):
        self.model.eval()
        scores_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                graph = self.model.preprocessor.create_hetero_from_batch(batch)
                output = self.model(graph)

                h = output["entity"][batch.head.squeeze()]
                t = output["entity"][batch.tail.squeeze()]
                r = self.model.get_relation_embedding(batch.edge_attr.squeeze())

                scores = self.model.score_function(h, t, r)
                scores_list.append(scores.sigmoid().cpu())
                labels_list.append(batch.y.cpu())

        scores = torch.cat(scores_list)
        labels = torch.cat(labels_list)
        return roc_auc_score(labels, scores)
