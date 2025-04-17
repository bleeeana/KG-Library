import torch
from torch.utils.data import DataLoader
class TripletsDataset(torch.utils.data.Dataset):
    def __init__(self, triplets, labels):
        self.triplets = triplets
        self.labels = labels

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx], self.labels[idx]

def create_dataloader(preprocessor, batch_size = 64):
    train_data = TripletsDataset(preprocessor.split_triplets[0], preprocessor.labels[0])
    val_data = TripletsDataset(preprocessor.split_triplets[1], preprocessor.labels[1])
    test_data = TripletsDataset(preprocessor.split_triplets[2], preprocessor.labels[2])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, test_loader, val_loader