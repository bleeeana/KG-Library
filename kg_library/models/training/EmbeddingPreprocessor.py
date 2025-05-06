from typing import Optional
import torch
from torch_geometric.loader import DataLoader
from kg_library.common import GraphData
from torch_geometric.data import HeteroData, Batch
from sklearn.model_selection import train_test_split
import numpy as np

class EmbeddingPreprocessor:
    def __init__(self, graph: GraphData):
        self.graph = graph
        self.device : torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names : list[str] = []
        self.entity_id : dict[str, int] = {}
        self.relation_id : dict[str, int] = {}
        self.feature_matrix : Optional[torch.Tensor] = None
        self.hetero_graph : Optional[HeteroData] = None
        self.split_triplets: Optional[list[np.ndarray]] = None
        self.labels: Optional[list[np.ndarray]] = None
        self.entities_by_type: dict[str, list[int]] = {}

        self._verify_device()

    def _verify_device(self):
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Selected device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    def get_feature_tensor(self, feature_name: str) -> torch.Tensor:
        if not self.feature_names:
            self.feature_names = sorted({node.feature for node in self.graph.nodes})
        feature = torch.zeros(len(self.feature_names), dtype=torch.float32).to(self.device)
        feature[self.feature_names.index(feature_name)] = 1.0
        return feature

    def expand_feature_names(self, feature_names):
        for feature_name in feature_names:
            if feature_name and feature_name.lower() not in self.feature_names:
                self.feature_names.append(feature_name.lower())
        print(f"Updated features: {self.feature_names}")

    def build_feature_matrix(self, feature_names : list[str]) -> None:
        self.entity_id = {entity.name: i for i, entity in enumerate(self.graph.nodes)}
        self.relation_id = {relation.get_relation(): i for i, relation in enumerate(self.graph.edges)}
        self.feature_names = sorted({node.feature.lower() for node in self.graph.nodes})
        print(f"Detected features: {self.feature_names}")
        self.expand_feature_names(feature_names)
        feature_index = {name: i for i, name in enumerate(self.feature_names)}
        features = np.zeros((len(self.graph.nodes), len(self.feature_names)), dtype=np.float32)

        for node in self.graph.nodes:
            eid = self.entity_id[node.name]
            features[eid][feature_index[node.feature]] = 1.0

        self.feature_matrix = torch.tensor(features, dtype=torch.float32).to(self.device)

    def build_hetero_graph(self) -> None:
        self.hetero_graph = HeteroData()
        self.hetero_graph["entity"].x = self.feature_matrix
        edge_index_dict = {}
        edge_type_dict = {}
        for head, relation, tail in self.graph.triplets:
            relation_name = relation.get_relation()
            h_id = self.entity_id[head.name]
            t_id = self.entity_id[tail.name]
            if ('entity', relation_name, 'entity') not in edge_index_dict:
                edge_index_dict[('entity', relation_name, 'entity')] = [[], []]
            edge_index_dict[('entity', relation_name, 'entity')][0].append(h_id)
            edge_index_dict[('entity', relation_name, 'entity')][1].append(t_id)
            edge_type_dict[relation_name] = self.relation_id[relation_name]
        for relation_tuple, (src, dest) in edge_index_dict.items():
            self.hetero_graph[relation_tuple].edge_index = torch.tensor([src, dest], dtype=torch.long).to(self.device)
        self.hetero_graph.edge_type_dict = edge_type_dict

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

    def __generate_negative_triplets(self, positive_triplets: list, type_variation_prob=0.3) -> list:
        all_entities = list(self.entity_id.values())
        positive_set = set(positive_triplets)
        negative_triplets = []

        for entity in all_entities:
            entity_type = self.graph.nodes[entity].feature
            if entity_type not in self.entities_by_type:
                self.entities_by_type[entity_type] = []
            self.entities_by_type[entity_type].append(entity)

        for index, (head, relation, tail) in enumerate(positive_triplets):
            head_type = self.graph.nodes[head].feature
            tail_type = self.graph.nodes[tail].feature

            change_type = np.random.random() < type_variation_prob

            if index % 2 == 0:
                if change_type:
                    possible_types = [t for t in self.entities_by_type.keys() if t != head_type]
                    if possible_types:
                        chosen_type = np.random.choice(possible_types)
                        possible_heads = [e for e in self.entities_by_type[chosen_type]
                                          if e != head and (e, relation, tail) not in positive_set]
                    else:
                        possible_heads = []
                else:
                    possible_heads = [e for e in all_entities
                                      if e != head and (e, relation, tail) not in positive_set
                                      and self.graph.nodes[e].feature != head_type]
                if possible_heads:
                    negative_triplets.append((np.random.choice(possible_heads), relation, tail))
            else:
                if change_type:
                    possible_types = [t for t in self.entities_by_type.keys() if t != tail_type]
                    if possible_types:
                        chosen_type = np.random.choice(possible_types)
                        possible_tails = [e for e in self.entities_by_type[chosen_type]
                                          if e != tail and (head, relation, e) not in positive_set]
                    else:
                        possible_tails = []
                else:
                    possible_tails = [e for e in all_entities
                                      if e != tail and (head, relation, e) not in positive_set
                                      and self.graph.nodes[e].feature != tail_type]
                if possible_tails:
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
        self.split_triplets = [triplets_array[train_idx], triplets_array[val_idx], triplets_array[test_idx]]
        self.labels = [labels_array[train_idx], labels_array[val_idx], labels_array[test_idx]]
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
        edge_type_dict = {}
        for i in range(len(head)):
            h_id = head[i]
            t_id = tail[i]
            relation_name = next(key for key, value in self.relation_id.items() if value == batch.edge_attr[i])
            if ('entity', relation_name, 'entity') not in edge_index_dict:
                edge_index_dict[('entity', relation_name, 'entity')] = [[], []]
            edge_index_dict[('entity', relation_name, 'entity')][0].append(h_id)
            edge_index_dict[('entity', relation_name, 'entity')][1].append(t_id)
            edge_type_dict[relation_name] = batch.edge_attr[i].item()
        for relation_tuple, (src, dest) in edge_index_dict.items():
            hetero_data[relation_tuple].edge_index = torch.tensor([src, dest], dtype=torch.long).to(self.device)
        hetero_data.edge_type_dict = edge_type_dict
        return hetero_data

    def preprocess(self, feature_names, test_size: float = 0.001, val_size: float = 0.199, random_state: int = 1) -> None:
        self.build_feature_matrix(feature_names)
        self.build_hetero_graph()
        self.prepare_training_data(test_size, val_size, random_state)

    def get_or_create_relation_id(self, relation: str) -> int:
        if relation not in self.relation_id:
            self.relation_id[relation] = len(self.relation_id)
        return self.relation_id[relation]

    def get_or_create_entity_id(self, entity: str) -> int:
        if entity not in self.entity_id:
            self.entity_id[entity] = len(self.entity_id)
        return self.entity_id[entity]

    def get_config(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "relation_id": self.relation_id,
            "hetero_graph": self.hetero_graph,
            "split_triplets": self.split_triplets,
            "labels": self.labels,
            "entities_by_type": self.entities_by_type,
            "feature_names": self.feature_names
        }

    def load_config(self, config: dict) -> None:
        self.entity_id = config["entity_id"]
        self.relation_id = config["relation_id"]
        self.hetero_graph = config["hetero_graph"]
        self.split_triplets = config["split_triplets"]
        self.labels = config["labels"]
        self.entities_by_type = config["entities_by_type"]
        self.feature_matrix = self.hetero_graph["entity"].x
        self.feature_names = config["feature_names"]
