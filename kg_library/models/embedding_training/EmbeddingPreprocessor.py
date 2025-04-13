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
        test_tensor = torch.randn(3, 3).to(self.device)
        print(f"Test tensor device: {test_tensor.device}\n")

    def build_feature_matrix(self):
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

        feature_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        return entity_id, relation_id, feature_tensor

    def build_hetero_graph(self):
        entity_id, relation_id, feature_tensor = self.build_feature_matrix()
        hetero_graph = HeteroData()
        hetero_graph["Entity"].x = feature_tensor
        return hetero_graph