import unittest
from kg_library.common import GraphData
from kg_library.models import EmbeddingPreprocessor

class EmbeddigTest(unittest.TestCase):
    def test_init(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "Relation1", "Node 2")
        graph.add_new_triplet("Node 2", "second", "new node")
        graph.add_loop_reversed_triplet()
        graph.print()
        preprocessor = EmbeddingPreprocessor(graph)
        _, _, tensor = preprocessor.build_feature_matrix()
        print(tensor)
        assert tensor is not None

if __name__ == '__main__':
    unittest.main()
