import unittest

from kg_library.common import GraphJSON
from kg_library.models import EmbeddingPreprocessor
from kg_library.models import GraphTrainer, GraphNN, create_dataloader
from kg_library.utils import create_test_graph


class GNNTest(unittest.TestCase):
    def test_train(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess()
        model = GraphNN(preprocessor)
        train_loader, test_loader, val_loader = create_dataloader(preprocessor, batch_size=128)

        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")

        trainer = GraphTrainer(model, train_loader, val_loader, epochs=1000, lr=0.0005)

        trainer.train()
        val_auc = trainer.evaluate(trainer.val_loader)

        print(f"Final Val AUC: {val_auc}")
        self.assertGreater(val_auc, 0.7, "Model AUC is too low, might not be learning")


    def test_load_graph_train(self):
        graph = GraphJSON.load("test.json")
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess()
        model = GraphNN(preprocessor)
        train_loader, test_loader, val_loader = create_dataloader(preprocessor, batch_size=64)

        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")

        trainer = GraphTrainer(model, train_loader, val_loader, epochs=15, lr=0.0005)

        trainer.train()
        val_auc = trainer.evaluate(trainer.val_loader)

        print(f"Final Val AUC: {val_auc}")
        self.assertGreater(val_auc, 0.7, "Model AUC is too low, might not be learning")


if __name__ == '__main__':
    unittest.main()