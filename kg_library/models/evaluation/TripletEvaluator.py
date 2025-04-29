from torch_geometric.data import HeteroData

from kg_library.models import GraphNN, EmbeddingPreprocessor
import torch
import torch.nn.functional as F


class TripletEvaluator:
    def __init__(self, model: GraphNN):
        self.model = model
        self.device = model.device
        self.preprocessor: EmbeddingPreprocessor = model.preprocessor

    @torch.no_grad()
    def get_new_entity_embedding(self, entity_type):
        same_type_ids = []

        if entity_type is not None:
            for node_name, node_id in self.preprocessor.entity_id.items():
                node_type = None
                for node in self.preprocessor.graph.nodes:
                    if node.name == node_name:
                        node_type = node.feature
                        break

                if node_type == entity_type:
                    same_type_ids.append(node_id)

        if same_type_ids:
            type_embeddings = self.model.get_entity_embedding(
                torch.tensor(same_type_ids).to(self.device)
            )
            embedding = type_embeddings.mean(dim=0, keepdim=True)
        else:
            embedding = self.model.entity_embedding.weight.data.mean(dim=0, keepdim=True)
        noise = torch.randn_like(embedding) * 0.01
        embedding = embedding + noise

        return F.normalize(embedding, p=2, dim=1)

    @torch.no_grad()
    def score_new_triplet(self, graph : HeteroData, head_feature: str, tail_feature: str, head_name: str, tail_name: str, relation: str,
                          head_id: int = None, tail_id: int = None, add_to_graph: bool = False):
        if graph is None:
            graph = self.preprocessor.hetero_graph.to(self.device)

        with torch.no_grad():
            embeddings = self.model(graph)
            entity_embeddings = embeddings["entity"]

        if head_id is not None:
            head_embedding = entity_embeddings[head_id].unsqueeze(0)
        else:
            projected_head = self.model.feature_proj(self.preprocessor.get_feature_tensor(head_feature).to(self.device))

            head_entity_emb = self.get_new_entity_embedding(entity_type=head_feature)
            head_embedding = 0.6 * projected_head + 0.4 * head_entity_emb

        if tail_id is not None:
            tail_embedding = entity_embeddings[tail_id].unsqueeze(0)
        else:
            projected_tail = self.model.feature_proj(self.preprocessor.get_feature_tensor(tail_feature).to(self.device))
            tail_entity_emb = self.get_new_entity_embedding(entity_type=tail_feature)
            tail_embedding = 0.6 * projected_tail + 0.4 * tail_entity_emb

        if relation in self.preprocessor.relation_id:
            relation_id = self.preprocessor.relation_id[relation]
            relation_emb = self.model.get_relation_embedding(torch.tensor([relation_id]).to(self.device))
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

            self.preprocessor.preprocess()
            self.model.preprocessor = self.preprocessor
            return score_value, added_triplet

        return score_value

    @torch.no_grad()
    def score_triplet_from_graph(self, head_idx, tail_idx, relation_id):
        self.model.eval()
        device = self.device
        graph = self.preprocessor.hetero_graph.to(device)
        embeddings = self.model(graph)
        entity_embeddings = embeddings["entity"]

        head_embedding = entity_embeddings[head_idx].unsqueeze(0)
        tail_embedding = entity_embeddings[tail_idx].unsqueeze(0)
        relation_emb = self.model.get_relation_embedding(torch.tensor([relation_id]).to(device))

        score = self.model.score_function(head_embedding, tail_embedding, relation_emb)
        return torch.sigmoid(score).item()

    @torch.no_grad()
    def link_prediction_in_graph(self, threshold=0.75, top_k=10, batch_size=128):

        id_to_entity = {index: entity for entity, index in self.preprocessor.entity_id.items()}
        device = self.device
        possible_links = []

        graph = self.preprocessor.hetero_graph.to(device)
        embeddings = self.model(graph)
        entity_embeddings = embeddings["entity"]
        num_entities = entity_embeddings.size(0)

        for head_idx in range(num_entities):
            head_embedding = entity_embeddings[head_idx].unsqueeze(0)

            for rel_name, rel_idx in self.preprocessor.relation_id.items():
                if rel_name == "loop" or  rel_name.endswith(":reversed"):
                    continue
                relation_emb = self.model.get_relation_embedding(
                    torch.tensor([rel_idx]).to(self.device))

                valid_indices = [i for i in range(num_entities) if i != head_idx]

                all_scores = []
                for start_idx in range(0, len(valid_indices), batch_size):
                    batch_indices = valid_indices[start_idx:start_idx + batch_size]

                    batch_tail_embeddings = entity_embeddings[batch_indices]

                    batch_heads = head_embedding.repeat(len(batch_indices), 1)
                    batch_relations = relation_emb.repeat(len(batch_indices), 1)

                    batch_scores = self.model.score_function(batch_heads, batch_tail_embeddings, batch_relations)
                    batch_scores = torch.sigmoid(batch_scores).squeeze()

                    all_scores.append((batch_scores, batch_indices))

                all_batch_scores = torch.cat([scores for scores, _ in all_scores])
                all_batch_indices = [idx for _, indices in all_scores for idx in indices]

                above_thresh_mask = all_batch_scores > threshold
                if above_thresh_mask.any():
                    above_thresh_scores = all_batch_scores[above_thresh_mask]
                    above_thresh_indices = [all_batch_indices[i] for i, is_above in enumerate(above_thresh_mask) if
                                            is_above]

                    if len(above_thresh_scores) > top_k:
                        top_scores, top_idx = torch.topk(above_thresh_scores, top_k)
                        top_indices = [above_thresh_indices[i] for i in top_idx.tolist()]
                        top_scores = top_scores.tolist()
                    else:
                        top_scores = above_thresh_scores.tolist()
                        top_indices = above_thresh_indices

                    for score, tail_idx in zip(top_scores, top_indices):
                        possible_links.append({
                            'head': id_to_entity[head_idx],
                            'relation': rel_name,
                            'tail': id_to_entity[tail_idx],
                            'score': score
                        })
                torch.cuda.empty_cache()

        possible_links.sort(key=lambda x: x['score'], reverse=True)
        return possible_links
