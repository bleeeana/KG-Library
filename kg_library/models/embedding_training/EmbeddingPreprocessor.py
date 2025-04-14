import torch
from kg_library.common import GraphData
from torch_geometric.data import HeteroData
import numpy as np

class EmbeddingPreprocessor:
    def __init__(self, graph : GraphData):
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__verify_device()

    def __verify_device(self):
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    def build_feature_matrix(self) -> tuple[dict[str, int], dict[str, int], torch.Tensor]:
        unique_entities, unique_relations = self.graph.get_unique_sets()
        entity_id = {entity: i for i, entity in enumerate(unique_entities)}
        print(f"unique entities: {unique_entities}, unique relations: {unique_relations}")
        relation_id = {relation: i for i, relation in enumerate(unique_relations)}
        features = np.zeros((len(unique_entities), len(unique_relations)), dtype=np.float32)

        for head, relation, tail in self.graph.triplets:
            print(f"head: {head.name}, relation: {relation.get_relation()}, tail: {tail.name}")
            h_id = entity_id[head.name]
            r_id = relation_id[relation.get_relation()]
            features[h_id][r_id] = 1.0
        print(features)
        feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        return entity_id, relation_id, feature_tensor

    def build_hetero_graph(self, features : torch.Tensor = None) -> tuple[HeteroData, dict, dict, torch.Tensor]:
        if features is None:
            features = self.build_feature_matrix()
        entity_id, relation_id, feature_tensor = features
        hetero_graph = HeteroData()
        hetero_graph["entity"].x = feature_tensor
        edge_index_dict = {}
        for head, relation, tail in self.graph.triplets:
            relation_name = relation.get_relation()
            h_id = entity_id[head.name]
            t_id = entity_id[tail.name]
            if ('entity', relation_name, 'entity') not in edge_index_dict:
                edge_index_dict[('entity', relation_name, 'entity')] = [[], []]
            edge_index_dict[('entity', relation_name, 'entity')][0].append(h_id)
            edge_index_dict[('entity', relation_name, 'entity')][1].append(t_id)
        for relation_tuple, (src, dest) in edge_index_dict.items():
            hetero_graph[relation_tuple].edge_index = torch.tensor([src, dest], dtype=torch.long).to(self.device)
        return hetero_graph, entity_id, relation_id, feature_tensor

    def prepare_training_data(self, features : torch.Tensor = None) -> tuple[list[tuple[dict, str, dict]], list[int], dict[str, int], dict[str, int]]:
        if features is None:
            features = self.build_feature_matrix()
        entity_id, relation_id, feature_tensor = features
        print(entity_id, relation_id, feature_tensor)
        positive_triplets = [(entity_id[head.name], relation.get_relation(), entity_id[tail.name]) for head, relation, tail in self.graph.triplets]
        all_entities = list(entity_id.values())
        negative_triplets = []
        for head, relation, tail in positive_triplets:
            corrupted_heads = [e for e in all_entities if e != head and (e, relation, tail) not in positive_triplets]
            corrupted_tails = [e for e in all_entities if e != tail and (head, relation, e) not in positive_triplets]
            if corrupted_heads:
                negative_triplets.append((np.random.choice(corrupted_heads), relation, tail))
            if corrupted_tails:
                negative_triplets.append((head, relation, np.random.choice(corrupted_tails)))
        labels = [1] * len(positive_triplets) + [0] * len(negative_triplets)
        triplets = positive_triplets + negative_triplets
        return triplets, labels, entity_id, relation_id