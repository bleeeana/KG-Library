from torch_geometric.data import HeteroData
from kg_library.models import GraphNN, EmbeddingPreprocessor
import torch
import torch.nn.functional as F


class TripletEvaluator:
    def __init__(self, model: GraphNN):
        self.model: GraphNN = model
        self.device_string = str(self.model.device)
        self.preprocessor: EmbeddingPreprocessor = model.preprocessor

    @torch.no_grad()
    def get_new_entity_embedding(self, entity_type) -> torch.Tensor:
        same_type_ids = self.preprocessor.entities_by_type.get(entity_type, None)

        if same_type_ids:
            type_embeddings = self.model.get_entity_embedding(
                torch.tensor(same_type_ids).to(self.device_string)
            )
            embedding = type_embeddings.mean(dim=0, keepdim=True)
        else:
            embedding = self.model.entity_embedding.weight.data.mean(dim=0, keepdim=True)
        noise = torch.randn_like(embedding) * 0.01
        embedding = embedding + noise

        return F.normalize(embedding, p=2, dim=1)

    @torch.no_grad()
    def score_new_triplet(self, graph: HeteroData, head_feature: str, tail_feature: str, head_name: str, tail_name: str,
                          relation: str,
                          head_id: int = None, tail_id: int = None, add_to_graph: bool = False,
                          feature_names: list[str] = None):
        if graph is None:
            graph = self.preprocessor.hetero_graph.to(self.device_string)

        embeddings = self.model(graph)["entity"]

        head_embedding = self.get_entity_embedding(embeddings, head_id, head_feature)
        tail_embedding = self.get_entity_embedding(embeddings, tail_id, tail_feature)

        if relation in self.preprocessor.relation_id:
            relation_id = self.preprocessor.relation_id[relation]
            relation_emb = self.model.get_relation_embedding(torch.tensor([relation_id]).to(self.device_string))
        else:
            relation_emb = F.normalize(
                self.model.relation_embedding.weight.data.mean(dim=0, keepdim=True), p=2, dim=1
            )

        score = self.model.score_function(head_embedding, tail_embedding, relation_emb)
        score_value = score.item()

        if add_to_graph and relation is not None:
            self.preprocessor.graph.add_new_triplet(
                head=head_name,
                relation=relation,
                tail=tail_name,
                check_synonyms=True,
                head_feature=head_feature,
                tail_feature=tail_feature
            )

            added_triplet = None
            for triplet in self.preprocessor.graph.triplets:
                if (triplet[0].name == head_name and
                        triplet[1].get_relation() == relation and
                        triplet[2].name == tail_name):
                    added_triplet = triplet
                    break

            self.preprocessor.preprocess(feature_names)
            self.model.preprocessor = self.preprocessor
            return score_value, added_triplet

        return score_value

    def get_entity_embedding(self, embeddings, entity_id, entity_type):
        if entity_id is not None:
            return embeddings[entity_id].unsqueeze(0)
        else:
            return self.get_new_entity_embedding(entity_type=entity_type)

    @torch.no_grad()
    def link_prediction_in_graph(self, threshold=0.65, top_k=1, batch_size=128, target_nodes=None):
        self.model.eval()
        device = self.device_string
        graph = self.preprocessor.hetero_graph.to(device)
        entity_embeddings = self.model(graph)["entity"]
        num_entities = entity_embeddings.size(0)

        if target_nodes is not None:
            valid_node_ids = [
                self.preprocessor.entity_id[name] for name in target_nodes if name in self.preprocessor.entity_id
            ]
        else:
            valid_node_ids = list(range(num_entities))

        valid_relations = [
            (name, idx) for name, idx in self.preprocessor.relation_id.items()
            if name != "loop" and not name.endswith(":reversed")
        ]

        possible_links = []

        for head_idx in valid_node_ids:
            head_embedding = entity_embeddings[head_idx].unsqueeze(0)
            valid_tail_indices = torch.tensor(
                [i for i in valid_node_ids if i != head_idx], device=device
            )

            for rel_name, rel_idx in valid_relations:
                relation_emb = self.model.get_relation_embedding(
                    torch.tensor([rel_idx], device=device)
                )

                all_scores = torch.tensor([], device=device)

                for i in range(0, len(valid_tail_indices), batch_size):
                    batch_indices = valid_tail_indices[i:i + batch_size]
                    batch_tail_embeddings = entity_embeddings[batch_indices]

                    batch_scores = self.model.score_function(
                        head_embedding.expand(len(batch_indices), -1),
                        batch_tail_embeddings,
                        relation_emb.expand(len(batch_indices), -1)
                    )
                    all_scores = torch.cat([all_scores, torch.sigmoid(batch_scores.squeeze())])

                mask = all_scores > threshold
                if mask.any():
                    filtered_scores = all_scores[mask]
                    filtered_indices = valid_tail_indices[mask]

                    top_n = min(top_k, len(filtered_scores))
                    top_scores, top_idxs = torch.topk(filtered_scores, top_n)

                    for score, tail_idx in zip(top_scores.tolist(), filtered_indices[top_idxs].tolist()):
                        possible_links.append({
                            'head': self.preprocessor.graph.nodes[head_idx],
                            'relation': self.preprocessor.graph.edges[rel_idx],
                            'tail': self.preprocessor.graph.nodes[tail_idx],
                            'score': score
                        })
            torch.cuda.empty_cache()

        return sorted(possible_links, key=lambda x: x['score'], reverse=True)