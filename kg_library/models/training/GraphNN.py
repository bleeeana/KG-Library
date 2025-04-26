import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv
from kg_library import get_config
from kg_library.models import EmbeddingPreprocessor

class GraphNN(nn.Module):
    def __init__(self, preprocessor : EmbeddingPreprocessor, hidden_dim=get_config()["hidden_dim"], num_layers=3, dropout=0.3):
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

    def get_config(self) -> dict:
        return {
            "hidden_dim": self.entity_embedding.embedding_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout.p
        }

    def save_for_inference(self, path="model.pt"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model.get_config()
        }, path)

    # без дообучения
    @staticmethod
    def load_model(model_path="model.pt", map_location='cuda', preprocessor : EmbeddingPreprocessor = None, ):
        checkpoint = torch.load(model_path, map_location=map_location)
        model = GraphNN(preprocessor, hidden_dim=checkpoint["model_config"]["hidden_dim"], num_layers=checkpoint["model_config"]["num_layers"], dropout=checkpoint["model_config"]["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(preprocessor.device)
        return model


