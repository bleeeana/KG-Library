import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from kg_library import get_config
from kg_library.models import EmbeddingPreprocessor
from kg_library.models.training.CompGCNConv import CompGCNConv


class GraphNN(nn.Module):
    def __init__(self, preprocessor: EmbeddingPreprocessor, hidden_dim=get_config()["hidden_dim"],
                 num_layers=3, dropout=0.2):
        super().__init__()
        self.preprocessor = preprocessor
        self.device = preprocessor.device
        self.num_layers = num_layers

        num_entities = len(preprocessor.entity_id)
        num_relations = len(preprocessor.relation_id)

        num_node_types = len(preprocessor.feature_names) if hasattr(preprocessor, 'feature_names') else 1
        self.entity_embedding = nn.Embedding(num_entities, hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)

        self.layers = nn.ModuleList([
            CompGCNConv(hidden_dim, hidden_dim, num_relations, num_node_types=num_node_types)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.skip_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(num_layers)
        ])

        self.gamma = nn.Parameter(torch.tensor(10.0))
        self.to(self.device)

    def forward(self, graph: HeteroData):
        x = {"entity": self.entity_embedding.weight}
        x_initial = {"entity": x["entity"].clone()}

        for i, conv in enumerate(self.layers):
            x_updated = conv(
                x["entity"],
                graph.edge_index_dict,
                graph.edge_type_dict,
                graph.node_type_dict
            )
            x_updated = {
                "entity": self.dropout(F.leaky_relu(x_updated["entity"], 0.2))
            }
            alpha = torch.sigmoid(self.skip_weights[i])
            x = {
                "entity": (1 - alpha) * x_updated["entity"] + alpha * x_initial["entity"]
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

    @staticmethod
    def load_model(model_path="model.pt", map_location='cuda', preprocessor: EmbeddingPreprocessor = None):
        checkpoint = torch.load(model_path, map_location=map_location)
        model = GraphNN(
            preprocessor,
            hidden_dim=checkpoint["model_config"]["hidden_dim"],
            num_layers=checkpoint["model_config"]["num_layers"],
            dropout=checkpoint["model_config"]["dropout"]
        )
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

                    if (entity_type in source_preprocessor.entities_by_type and
                            source_preprocessor.entities_by_type[entity_type]):
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

                        if ':reversed' in relation:
                            base_embedding = -base_embedding

                        noise = torch.randn_like(base_embedding) * 0.01
                        self.relation_embedding.weight[new_id] = F.normalize(base_embedding + noise, p=2, dim=0)
                    else:
                        avg_embedding = source_model.relation_embedding.weight.data.mean(dim=0)
                        noise = torch.randn_like(avg_embedding) * 0.01
                        self.relation_embedding.weight[new_id] = F.normalize(avg_embedding + noise, p=2, dim=0)