import unittest

from kg_library.common import GraphJSON
from kg_library.models import EmbeddingPreprocessor
from kg_library.models import GraphTrainer, GraphNN, create_dataloader
from kg_library.utils import create_test_graph
from kg_library import AppFacade

class GNNTest(unittest.TestCase):
    def test_train(self):
        graph = create_test_graph()
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess([])
        model = GraphNN(preprocessor, num_layers=2)
        train_loader, test_loader, val_loader = create_dataloader(preprocessor, batch_size=128)

        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")
        print(f"hetero data: {preprocessor.hetero_graph.to_dict()}")

        trainer = GraphTrainer(model, train_loader, val_loader, epochs=1000, lr=0.0005)

        trainer.train(save=False)
        val_auc = trainer.evaluate(trainer.val_loader)

        print(f"Final Val AUC: {val_auc}")
        self.assertGreater(val_auc, 0.7, "Model AUC is too low, might not be learning")

    def test_load_graph_train(self):
        graph = GraphJSON.load("test.json")
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.preprocess([])
        model = GraphNN(preprocessor)
        train_loader, test_loader, val_loader = create_dataloader(preprocessor, batch_size=64)

        print(f"entity id: {preprocessor.entity_id}")
        print(f"relation id: {preprocessor.relation_id}")
        print(f"hetero data: {preprocessor.hetero_graph.to_dict()}")

        trainer = GraphTrainer(model, train_loader, val_loader, epochs=15, lr=0.0005)

        trainer.train()
        val_auc = trainer.evaluate(trainer.val_loader)

        print(f"Final Val AUC: {val_auc}")
        self.assertGreater(val_auc, 0.7, "Model AUC is too low, might not be learning")

    def test_finetune(self):
        app_facade = AppFacade()
        app_facade.generate_graph_for_learning(False, False, True, "base_graph.json")
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()