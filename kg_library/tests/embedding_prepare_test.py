import unittest
from kg_library.models import EmbeddingPreprocessor
from kg_library.models import create_dataloader
from kg_library.utils import create_test_graph, create_mini_test_graph

class EmbeddingPrepareTest(unittest.TestCase):

    def test_feature_matrix(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.build_feature_matrix()
        print(preprocessor.feature_matrix)
        self.assertIsNotNone(preprocessor.feature_matrix)

    def test_creating_leaf_tensor(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.build_feature_matrix()
        preprocessor.build_hetero_graph()
        print(preprocessor.hetero_graph.to_dict())
        print(preprocessor.hetero_graph.metadata())
        self.assertIsNotNone(preprocessor.hetero_graph)

    def test_full_preprocessing(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess()
        print(f"feature matrix: {preprocessor.feature_matrix}")
        print(f"hetero data node types: {preprocessor.hetero_graph.node_types}")
        print(f"hetero data edge types: {preprocessor.hetero_graph.edge_types}")
        print(f"hetero data: {preprocessor.hetero_graph.to_dict()}")
        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")
        print(f"triplets: {preprocessor.split_triplets}")
        print(f"labels: {preprocessor.labels}")
        self.assertIsNotNone(preprocessor.split_triplets and preprocessor.labels)

    def test_create_batch_hetero_data(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess()
        print(f"feature matrix: {preprocessor.feature_matrix}")
        #print(f"hetero data: {preprocessor.hetero_graph.to_dict()}")
        train_loader, val_loader, test_loader = create_dataloader(preprocessor)
        train_batches_hetero_graph = preprocessor.generate_split_hetero_data(train_loader)
        val_batches_hetero_graph = preprocessor.generate_split_hetero_data(val_loader)
        test_batches_hetero_graph = preprocessor.generate_split_hetero_data(test_loader)
        self.assertIsNotNone([train_batches_hetero_graph, val_batches_hetero_graph, test_batches_hetero_graph])

if __name__ == '__main__':
    unittest.main()
