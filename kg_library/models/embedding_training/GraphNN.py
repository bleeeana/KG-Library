import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv
from kg_library.models import EmbeddingPreprocessor, create_dataloader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score


class GraphNN(nn.Module):
    def __init__(self, preprocessor, dimension=64, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.preprocessor = preprocessor
        self.num_layers = num_layers
        self.embedding_dim = dimension
        self.device = preprocessor.device

        self.entity_embedding = nn.Embedding(len(preprocessor.entity_id) + 1, dimension)
        self.relation_embedding = nn.Embedding(len(preprocessor.relation_id) + 1,
                                               dimension)

        self.gamma = nn.Parameter(torch.Tensor([12.0]))

        self.feature_transform = nn.Sequential(
            nn.Linear(preprocessor.feature_matrix.size(1), dimension),
            nn.LayerNorm(dimension),
            nn.GELU()
        ).to(self.device)

        self.convs = nn.ModuleList([
            HeteroConv({
                edge_type: GATConv(
                    (-1, -1),
                    dimension,
                    heads=4,
                    concat=False,
                    dropout=dropout_rate
                )
                for edge_type in preprocessor.hetero_graph.edge_types
            }, aggr='mean')
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, data: HeteroData):
        x_features = self.feature_transform(self.preprocessor.feature_matrix.to(self.device))
        x_embed = F.normalize(self.entity_embedding(
            torch.arange(x_features.size(0), device=self.device)
        ), p=2, dim=1)

        x = {"entity": x_features + x_embed}

        for conv in self.convs:
            x = conv(x, data.edge_index_dict)
            x = {k: F.gelu(self.dropout(v)) for k, v in x.items()}

        return x

    def score_function(self, h, t, r):

        h = F.normalize(h, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)
        r = F.normalize(r, p=2, dim=1)

        re_h, im_h = torch.chunk(h, 2, dim=1)
        re_r, im_r = torch.chunk(r, 2, dim=1)
        re_t, im_t = torch.chunk(t, 2, dim=1)

        re_score = re_h * re_r - im_h * im_r
        im_score = re_h * im_r + im_h * re_r

        diff = torch.cat([re_score - re_t, im_score - im_t], dim=1)
        return self.gamma - torch.norm(diff, p=2, dim=1)

    def get_entity_embedding(self, ids: torch.Tensor):
        return self.entity_embedding(ids.to(self.device))

    def get_relation_embedding(self, ids):
        emb = self.relation_embedding(ids.to(self.device))
        return F.normalize(emb, p=2, dim=1)

class GraphTrainer:
    def __init__(self, preprocessor : EmbeddingPreprocessor, epochs=100, lr=1e-3, batch_size=64):
        self.preprocessor = preprocessor
        self.device = preprocessor.device
        self.model = GraphNN(preprocessor).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7, patience=5, verbose=True)

        self.epochs = epochs
        self.train_loader, self.val_loader, self.test_loader = create_dataloader(preprocessor, batch_size)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            loss = 0.0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                output_dict = self.model(self.preprocessor.create_hetero_from_batch(batch))

                h_batch = batch.head.squeeze()
                t_batch = batch.tail.squeeze()
                r_batch = batch.edge_attr.squeeze()
                labels_batch = batch.y

                h_emb = output_dict["entity"][h_batch]
                t_emb = output_dict["entity"][t_batch]
                r_emb = self.model.get_relation_embedding(r_batch)

                scores = self.model.score_function(h_emb, t_emb, r_emb)
                loss = F.binary_cross_entropy_with_logits(scores, labels_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.item()
                auc = self.evaluate(self.val_loader)
                self.scheduler.step(auc)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                auc = self.evaluate(self.val_loader)
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Val AUC: {auc:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in dataloader:
                output_dict = self.model(self.preprocessor.create_hetero_from_batch(batch))

                batch = batch.to(self.device)
                h = batch.head.squeeze()
                t = batch.tail.squeeze()
                r = batch.edge_attr.squeeze()
                y = batch.y

                h_emb = output_dict['entity'][h]
                t_emb = output_dict['entity'][t]
                r_emb = self.model.get_relation_embedding(r)
                scores = self.model.score_function(h_emb, t_emb, r_emb)
                probs = torch.sigmoid(scores)
                all_scores.append(probs.cpu())
                all_labels.append(y.cpu())

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        return roc_auc_score(all_labels, all_scores)
