import torch
from kg_library.common import GraphData, NodeData, EdgeData
from torch_geometric.data import HeteroData

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
        pass


def main():
    graph = GraphData()
    preprocessor = EmbeddingPreprocessor(graph)
    print("Embedding Preprocessor finished")


if __name__ == "__main__":
    main()