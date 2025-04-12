import unittest
from kg_library.common import GraphData
from kg_library.models import EmbeddingPreprocessor

class EmbeddigTest(unittest.TestCase):
    def test_init(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "Relation1", "Node 2")
        graph.add_new_triplet("Node 2", "Relation2", "Node 3")
        graph.add_loop_reversed_triplet()
        preprocessor = EmbeddingPreprocessor(graph)



if __name__ == '__main__':
    unittest.main()
