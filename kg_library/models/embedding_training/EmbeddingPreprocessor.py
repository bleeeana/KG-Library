import torch
from torch_geometric.loader import DataLoader

from kg_library.common import GraphData
from torch_geometric.data import HeteroData, Batch
from sklearn.model_selection import train_test_split
import numpy as np

class EmbeddingPreprocessor:
    def __init__(self, graph: GraphData):
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.entity_id = None
        self.relation_id = None
        self.feature_matrix = None
        self.hetero_graph = None
        self.split_triplets = None
        self.labels = None
        self.__verify_device()

    def __verify_device(self):
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    def build_feature_matrix(self) -> None:
        unique_entities, unique_relations = self.graph.get_unique_sets()
        self.entity_id = {entity: i for i, entity in enumerate(unique_entities)}
        self.relation_id = {relation: i for i, relation in enumerate(unique_relations)}
        feature_names = sorted({node.feature for node in self.graph.nodes})
        feature_index = {name: i for i, name in enumerate(feature_names)}
        print(f"Detected features: {feature_names}")
        features = np.zeros((len(unique_entities), len(feature_names)), dtype=np.float32)

        for node in self.graph.nodes:
            eid = self.entity_id[node.name]
            if node.feature in feature_index:
                features[eid][feature_index[node.feature]] = 1.0

        self.feature_matrix = torch.tensor(features, dtype=torch.float32).to(self.device)

    def build_hetero_graph(self) -> None:
        self.hetero_graph = HeteroData()
        self.hetero_graph["entity"].x = self.feature_matrix
        edge_index_dict = {}
        for head, relation, tail in self.graph.triplets:
            relation_name = relation.get_relation()
            h_id = self.entity_id[head.name]
            t_id = self.entity_id[tail.name]
            if ('entity', relation_name, 'entity') not in edge_index_dict:
                edge_index_dict[('entity', relation_name, 'entity')] = [[], []]
            edge_index_dict[('entity', relation_name, 'entity')][0].append(h_id)
            edge_index_dict[('entity', relation_name, 'entity')][1].append(t_id)
        for relation_tuple, (src, dest) in edge_index_dict.items():
            self.hetero_graph[relation_tuple].edge_index = torch.tensor([src, dest], dtype=torch.long).to(self.device)

    def prepare_training_data(
            self,
            test_size,
            val_size,
            random_state: int = 1
    ) -> None:
        positive_triplets = [
            (self.entity_id[head.name], relation.get_relation(), self.entity_id[tail.name])
            for head, relation, tail in self.graph.triplets
        ]
        negative_triplets = self.__generate_negative_triplets(positive_triplets)

        triplets = positive_triplets + negative_triplets
        labels = [1] * len(positive_triplets) + [0] * len(negative_triplets)

        self.__split_data(triplets, labels, test_size, val_size, random_state)

    def __generate_negative_triplets(
            self,
            positive_triplets: list
    ) -> list:
        all_entities = list(self.entity_id.values())
        positive_set = set(positive_triplets)
        negative_triplets = []

        for index, (head, relation, tail) in enumerate(positive_triplets):
            possible_heads = [e for e in all_entities if e != head and (e, relation, tail) not in positive_set and self.graph.nodes[e].feature != self.graph.nodes[head].feature]
            possible_tails = [e for e in all_entities if e != tail and (head, relation, e) not in positive_set and self.graph.nodes[e].feature != self.graph.nodes[tail].feature]
            if possible_heads and index % 2 == 0:
                negative_triplets.append((np.random.choice(possible_heads), relation, tail))
            if possible_tails and index % 2 == 1:
                negative_triplets.append((head, relation, np.random.choice(possible_tails)))
        return negative_triplets

    def __split_data(
            self,
            triplets: list,
            labels: list,
            test_size: float,
            val_size: float,
            random_state: int
    ) -> None:
        triplets_array = np.array(triplets, dtype=object)
        labels_array = np.array(labels)
        assert test_size + val_size < 1.0, "Сумма test_size и val_size должна быть меньше 1"

        train_val_idx, test_idx = train_test_split(
            np.arange(len(triplets)),
            test_size=test_size,
            stratify=labels_array,
            random_state=random_state
        )
        val_ratio = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            stratify=labels_array[train_val_idx],
            random_state=random_state
        )
        self.split_triplets = [
            triplets_array[train_idx],
            triplets_array[val_idx],
            triplets_array[test_idx]
        ]
        self.labels = [
            labels_array[train_idx],
            labels_array[val_idx],
            labels_array[test_idx]
        ]
        self.__validate_class_balance()

    def __validate_class_balance(self) -> None:
        for name, labels in zip(['Train', 'Validation', 'Test'], self.labels):
            unique, counts = np.unique(labels, return_counts=True)
            print(f"{name} set class balance: {dict(zip(unique, counts))}")

    def generate_split_hetero_data(self, loader : DataLoader) -> list[HeteroData]:
        print(loader.dataset)
        result_hetero_data = []
        for index, batch in enumerate(loader):
            batch : Batch.to_data_list
            result_hetero_data.append(self.create_hetero_from_batch(batch))
        return result_hetero_data

    @staticmethod
    def get_unique_ids(head_id, tail_id) -> list[int]:
        unique_ids = torch.cat([head_id, tail_id]).unique()
        return unique_ids

    def create_hetero_from_batch(self, batch : Batch.to_data_list) -> HeteroData:
        head = batch.head
        tail = batch.tail
        unique_ids = self.get_unique_ids(head, tail)
        #print(f"Unique ids: {unique_ids}")
        feature_matrix = self.feature_matrix[unique_ids]
        hetero_data = HeteroData()
        hetero_data["entity"].x = feature_matrix
        edge_index_dict = {}
        for i in range(len(head)):
            h_id = head[i]
            t_id = tail[i]
            relation_name = next(key for key, value in self.relation_id.items() if value == batch.edge_attr[i])
            if ('entity', relation_name, 'entity') not in edge_index_dict:
                edge_index_dict[('entity', relation_name, 'entity')] = [[], []]
            edge_index_dict[('entity', relation_name, 'entity')][0].append(h_id)
            edge_index_dict[('entity', relation_name, 'entity')][1].append(t_id)
        for relation_tuple, (src, dest) in edge_index_dict.items():
            hetero_data[relation_tuple].edge_index = torch.tensor([src, dest], dtype=torch.long).to(self.device)
        return hetero_data

    def preprocess(self, test_size: float = 0.001, val_size: float = 0.199, random_state: int = 1) -> None:
        self.build_feature_matrix()
        self.build_hetero_graph()
        self.prepare_training_data(test_size, val_size, random_state)

    def get_config(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "relation_id": self.relation_id,
            "feature_matrix": self.feature_matrix,
            "hetero_graph": self.hetero_graph,
            "split_triplets": self.split_triplets,
            "labels": self.labels
        }

    def load_config(self, config: dict) -> None:
        self.entity_id = config["entity_id"]
        self.relation_id = config["relation_id"]
        self.feature_matrix = config["feature_matrix"]
        self.hetero_graph = config["hetero_graph"]
        self.split_triplets = config["split_triplets"]
        self.labels = config["labels"]