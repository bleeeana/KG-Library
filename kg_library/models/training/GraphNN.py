import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
# noinspection PyProtectedMember
from torch_geometric.nn import HeteroConv, GATv2Conv
from kg_library import get_config
from kg_library.models import EmbeddingPreprocessor

class GraphNN(nn.Module):
    def __init__(self, preprocessor : EmbeddingPreprocessor, hidden_dim=get_config()["hidden_dim"], num_layers=3, dropout=0.3):
        super().__init__()
        self.preprocessor : EmbeddingPreprocessor = preprocessor
        self.device : torch.device = preprocessor.device
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

        self.layers = nn.ModuleList([
            HeteroConv({
                edge_type: GATv2Conv(
                    (-1, -1),
                    hidden_dim,
                    heads=4,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,

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

        x = {"entity": 0.6 * node_features + 0.4 * node_embeddings}
        x_initial = {k: v.clone() for k, v in x.items()}

        for i, conv in enumerate(self.layers):
            x_updated = conv(x, graph.edge_index_dict)
            alpha = torch.sigmoid(self.skip_weights[i])
            x = {
                key: (1 - alpha) * F.leaky_relu(self.dropout(val), 0.2) + alpha * x_initial[key]
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

    # без дообучения
    @staticmethod
    def load_model(model_path="model.pt", map_location='cuda', preprocessor : EmbeddingPreprocessor = None, ):
        checkpoint = torch.load(model_path, map_location=map_location)
        model = GraphNN(preprocessor, hidden_dim=checkpoint["model_config"]["hidden_dim"], num_layers=checkpoint["model_config"]["num_layers"], dropout=checkpoint["model_config"]["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(preprocessor.device)
        return model

    def transfer_weights(self, source_model, source_preprocessor):
        with torch.no_grad():
            for entity, old_id in source_preprocessor.entity_id.items():
                if entity in self.preprocessor.entity_id:
                    new_id = self.preprocessor.entity_id[entity]
                    self.entity_embedding.weight[new_id] = source_model.entity_embedding.weight[old_id]

            for entity, new_id in self.preprocessor.entity_id.items():
                if entity not in source_preprocessor.entity_id:
                    entity_type = self.preprocessor.graph.nodes[new_id].feature

                    if entity_type in source_preprocessor.entities_by_type and source_preprocessor.entities_by_type[entity_type]:
                        type_ids = source_preprocessor.entities_by_type[entity_type]
                        type_embeddings = source_model.entity_embedding.weight[type_ids]
                        avg_embedding = type_embeddings.mean(dim=0)
                    else:
                        avg_embedding = source_model.entity_embedding.weight.data.mean(dim=0)

                    noise = torch.randn_like(avg_embedding) * 0.01
                    self.entity_embedding.weight[new_id] = F.normalize(avg_embedding + noise, p=2, dim=0)

            for relation, old_id in source_preprocessor.relation_id.items():
                if relation in self.preprocessor.relation_id:
                    new_id = self.preprocessor.relation_id[relation]
                    self.relation_embedding.weight[new_id] = source_model.relation_embedding.weight[old_id]

            for relation, new_id in self.preprocessor.relation_id.items():
                if relation not in source_preprocessor.relation_id:
                    base_relation = relation.split(':')[0] if ':reversed' in relation else relation

                    if base_relation in source_preprocessor.relation_id:
                        base_id = source_preprocessor.relation_id[base_relation]
                        base_embedding = source_model.relation_embedding.weight[base_id]
                        noise = torch.randn_like(base_embedding) * 0.01
                        self.relation_embedding.weight[new_id] = F.normalize(base_embedding + noise, p=2, dim=0)
                    else:
                        avg_embedding = source_model.relation_embedding.weight.data.mean(dim=0)
                        noise = torch.randn_like(avg_embedding) * 0.01
                        self.relation_embedding.weight[new_id] = F.normalize(avg_embedding + noise, p=2, dim=0)