import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch.optim import Adam
from sklearn.metrics import roc_auc_score

class GraphNN(nn.Module):
    def __init__(self, preprocessor, hidden_dimensions=64, embedding_dimension=32, num_layers=2, dropout_rate=0.3):
        super().__init__()
        self.preprocessor = preprocessor
        self.num_layers = num_layers
        self.hidden_channels = hidden_dimensions
        self.embedding_dimension = embedding_dimension
        self.device = preprocessor.device

        self.entity_embedding = nn.Embedding(len(self.preprocessor.entity_id) + 1, embedding_dimension)
        self.relation_embedding = nn.Embedding(len(self.preprocessor.relation_id) + 1, embedding_dimension)

        self.convolution_layers = nn.ModuleList([
            HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_dimensions if i < num_layers - 1 else embedding_dimension)
                for edge_type in self.preprocessor.hetero_graph.edge_types
            }, aggr='mean')
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data: HeteroData):
        x_dict = {"entity": self.preprocessor.feature_matrix}
        for layer in self.convolution_layers:
            x_dict = layer(x_dict, data.edge_index_dict)
            x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        return x_dict

    def score_function(self, h_emb, t_emb, r_emb):
        return torch.sum(h_emb * r_emb * t_emb, dim=1)

    def get_entity_embedding(self, ids: torch.Tensor):
        return self.entity_embedding(ids.to(self.device))

    def get_relation_embedding(self, ids: torch.Tensor):
        return self.relation_embedding(ids.to(self.device))


class GraphTrainer:
    def __init__(self, preprocessor, epochs=100, lr=0.01, weight_decay=1e-5):
        self.preprocessor = preprocessor
        self.device = preprocessor.device
        self.epochs = epochs

        self.model = GraphNN(preprocessor).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.best_auc = 0

    def train(self):
        triplets_train, triplets_val, _ = self.preprocessor.split_triplets
        labels_train, labels_val, _ = self.preprocessor.labels

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()

            output_dict = self.model(self.preprocessor.hetero_graph)

            h, r, t = zip(*triplets_train)
            h = torch.tensor(h, dtype=torch.long, device=self.device)
            r = [self.preprocessor.relation_id.get(rel, len(self.preprocessor.relation_id)) for rel in r]
            r = torch.tensor(r, dtype=torch.long, device=self.device)
            t = torch.tensor(t, dtype=torch.long, device=self.device)

            h_emb = output_dict["entity"][h]
            t_emb = output_dict["entity"][t]
            r_emb = self.model.get_relation_embedding(r)

            scores = self.model.score_function(h_emb, t_emb, r_emb)
            loss = F.binary_cross_entropy_with_logits(scores, torch.tensor(labels_train, dtype=torch.float32, device=self.device))

            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                auc = self.evaluate(split="val")
                print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Val AUC: {auc:.4f}")

    def evaluate(self, split="val"):
        self.model.eval()
        if split == "val":
            triplets = self.preprocessor.split_triplets[1]
            labels = self.preprocessor.labels[1]
        else:
            triplets = self.preprocessor.split_triplets[2]
            labels = self.preprocessor.labels[2]
        output_dict = self.model(self.preprocessor.hetero_graph)

        h, r, t = zip(*triplets)
        h = torch.tensor(h, dtype=torch.long, device=self.device)
        r = [self.preprocessor.relation_id.get(rel, len(self.preprocessor.relation_id)) for rel in r]
        r = torch.tensor(r, dtype=torch.long, device=self.device)
        t = torch.tensor(t, dtype=torch.long, device=self.device)

        h_emb = output_dict["entity"][h]
        t_emb = output_dict["entity"][t]
        r_emb = self.model.get_relation_embedding(r)
        with torch.no_grad():
            scores = self.model.score_function(h_emb, t_emb, r_emb)
            probs = torch.sigmoid(scores)
            auc = roc_auc_score(labels, probs.cpu().numpy())
        return auc