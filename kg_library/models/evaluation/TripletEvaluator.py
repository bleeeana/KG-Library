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
    def score_new_triplet(self, head_feature: str, tail_feature: str, head_name, tail_name, relation, head_id=None, tail_id=None,  add_to_graph=False):

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
    def link_prediction_in_graph(self, threshold=0.75, top_k=10):
        id_to_entity = {index: entity for entity, index in self.preprocessor.entity_id.items()}
        possible_links = []
        num_entities = len(self.preprocessor.entity_id)

        for head_idx in range(num_entities):
            for rel_name, rel_idx in self.preprocessor.relation_id.items():
                tail_scores = []
                for tail_idx in range(num_entities):
                    if tail_idx != head_idx:
                        score = self.score_triplet_from_graph(head_idx, tail_idx, rel_idx)
                        if score > threshold:
                            tail_scores.append((tail_idx, score))

                tail_scores.sort(key=lambda x: x[1], reverse=True)
                for tail_idx, score in tail_scores[:top_k]:
                    possible_links.append({
                        'head': id_to_entity[head_idx],
                        'relation': rel_name,
                        'tail': id_to_entity[tail_idx],
                        'score': score
                    })

        possible_links.sort(key=lambda x: x['score'], reverse=True)
        return possible_links