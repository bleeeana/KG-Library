import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.optim import Adam
from sklearn.metrics import roc_auc_score
from kg_library.models import create_dataloader, create_single_batch_dataloader

class GraphNN(nn.Module):
    def __init__(self, preprocessor, dimension = 64, output_dimension = 32, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.preprocessor = preprocessor
        self.num_layers = num_layers
        self.hidden_channels = dimension
        self.embedding_dimension = dimension
        self.device = preprocessor.device

        self.entity_embedding = nn.Embedding(len(self.preprocessor.entity_id) + 1, dimension)
        self.relation_embedding = nn.Embedding(len(self.preprocessor.relation_id) + 1, dimension)
        self.feature_transform = None
        self.relation_transform = nn.Linear(dimension, output_dimension).to(self.device)
        self.convolution_layers = nn.ModuleList([
            HeteroConv({
                edge_type: SAGEConv((-1, -1), dimension if i < num_layers - 1 else output_dimension, aggr='mean')
                for edge_type in self.preprocessor.hetero_graph.edge_types
            }, aggr='mean')
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data: HeteroData):
        x_features = self.preprocessor.feature_matrix.to(self.device)

        if self.feature_transform is None:
            features_dim = x_features.size(1)
            self.feature_transform = nn.Linear(features_dim, self.embedding_dimension).to(self.device)

        x_features = self.feature_transform(x_features)
        entity_indices = torch.arange(x_features.size(0), device=self.device)
        x_embedding = self.entity_embedding(entity_indices)
        x_combined = x_features + x_embedding

        x_dict = {"entity": x_combined}
        for layer in self.convolution_layers:
            x_dict = layer(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        return x_dict

    def score_function(self, h_emb, t_emb, r_emb):
        return torch.sum(h_emb * r_emb * t_emb, dim=1)

    def get_entity_embedding(self, ids: torch.Tensor):
        return self.entity_embedding(ids.to(self.device))

    def get_relation_embedding(self, ids: torch.Tensor):
        return self.relation_transform(self.relation_embedding(ids.to(self.device)))

class GraphTrainer:
    def __init__(self, preprocessor, epochs=100, lr=0.01, weight_decay=1e-5, batch_size=64):
        self.preprocessor = preprocessor
        self.device = preprocessor.device
        self.epochs = epochs

        self.model = GraphNN(preprocessor).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.train_loader, self.val_loader, self.test_loader = create_single_batch_dataloader(preprocessor)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in self.train_loader:
                batch = batch.to(self.device)
                output_dict = self.model(self.preprocessor.hetero_graph)

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

                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                auc = self.evaluate(self.val_loader)
                print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Val AUC: {auc:.4f}")

    def evaluate(self, dataloader):
        self.model.eval()
        all_scores = []
        all_labels = []

        output_dict = self.model(self.preprocessor.hetero_graph)

        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                h_batch = batch.head.squeeze()
                t_batch = batch.tail.squeeze()
                r_batch = batch.edge_attr.squeeze()

                labels_batch = batch.y

                h_emb = output_dict["entity"][h_batch]
                t_emb = output_dict["entity"][t_batch]
                r_emb = self.model.get_relation_embedding(r_batch)

                scores = self.model.score_function(h_emb, t_emb, r_emb)
                probs = torch.sigmoid(scores)
                all_scores.append(probs.detach().cpu())
                all_labels.append(labels_batch.detach().cpu())

        all_scores = torch.cat(all_scores).numpy()
        all_labels = torch.cat(all_labels).numpy()

        return roc_auc_score(all_labels, all_scores)
