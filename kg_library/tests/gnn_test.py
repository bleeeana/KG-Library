import unittest
from kg_library.models import EmbeddingPreprocessor
from kg_library.models import GraphTrainer
from kg_library.utils import create_test_graph

class GNNTest(unittest.TestCase):
    def test_train(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess()
        #print(f"feature matrix: {preprocessor.feature_matrix}")
        #print(f"hetero data node types: {preprocessor.hetero_graph.node_types}")
        #print(f"hetero data: {preprocessor.hetero_graph.to_dict()}")
        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")
        #print(f"triplets: {preprocessor.split_triplets}")
        #print(f"labels: {preprocessor.labels}")
        trainer = GraphTrainer(preprocessor, epochs=200000, batch_size=128, lr=0.0002)

        trainer.train()
        val_auc = trainer.evaluate(trainer.val_loader)

        print(f"Final Val AUC: {val_auc}")
        self.assertGreater(val_auc, 0.7, "Model AUC is too low, might not be learning")


if __name__ == '__main__':
    unittest.main()