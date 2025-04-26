import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader

from kg_library.models import EmbeddingPreprocessor


class TripletsDataset(Dataset):
    def __init__(self, triplets, labels, relation_id):
        super().__init__()
        self.triplets = triplets
        self.labels = labels
        self.relation_id = relation_id

    def len(self) -> int:
        return len(self.triplets)

    def get(self, idx) -> Data:
        h, r, t = self.triplets[idx]
        return Data(
            head=torch.tensor([h], dtype=torch.long),
            tail=torch.tensor([t], dtype=torch.long),
            edge_attr=torch.tensor([self.relation_id[r]], dtype=torch.long),
            y=torch.tensor([self.labels[idx]], dtype=torch.float)
        )


class InferenceTriplets(Dataset):
    def __init__(self, triplets : list[tuple[int, int, int]], preprocessor : EmbeddingPreprocessor, labels : list[int]):
        super().__init__()
        self.preprocessor = preprocessor
        self.labels = labels
        self.triplets = triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, item) -> Data:
        h, r, t = self.triplets[item]
        head_id = self.preprocessor.get_or_create_entity_id(h)
        tail_id = self.preprocessor.get_or_create_entity_id(t)
        relation_id = self.preprocessor.get_or_create_relation_id(r)
        labels = self.labels[item]
        return Data(
            head=torch.tensor([head_id], dtype=torch.long),
            tail=torch.tensor([tail_id], dtype=torch.long),
            edge_attr=torch.tensor([relation_id], dtype=torch.long),
            y=torch.tensor([labels], dtype=torch.float)
        )


def create_single_batch_dataloader(preprocessor: EmbeddingPreprocessor) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data = TripletsDataset(preprocessor.split_triplets[0], preprocessor.labels[0], preprocessor.relation_id)
    val_data = TripletsDataset(preprocessor.split_triplets[1], preprocessor.labels[1], preprocessor.relation_id)
    test_data = TripletsDataset(preprocessor.split_triplets[2], preprocessor.labels[2], preprocessor.relation_id)

    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    return train_loader, val_loader, test_loader


def create_dataloader(preprocessor : EmbeddingPreprocessor, batch_size=64) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data = TripletsDataset(preprocessor.split_triplets[0], preprocessor.labels[0], preprocessor.relation_id)
    val_data = TripletsDataset(preprocessor.split_triplets[1], preprocessor.labels[1], preprocessor.relation_id)
    test_data = TripletsDataset(preprocessor.split_triplets[2], preprocessor.labels[2], preprocessor.relation_id)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def create_inference_dataloader(preprocessor : EmbeddingPreprocessor, triplets : list[tuple[int, int, int]], labels : list[int], batch_size=64) -> DataLoader:
    dataset = InferenceTriplets(triplets, preprocessor, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)