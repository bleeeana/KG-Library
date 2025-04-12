import unittest
from kg_library.common import GraphData, NodeData

class TestGraphData(unittest.TestCase):
    def test_init(self):
        graph = GraphData()
        print("empty graph")
        self.assertEqual(graph.nodes, [])
        self.assertEqual(graph.edges, [])

    def test_add_node(self):
        graph = GraphData()
        print("Added 1 element")
        node = NodeData("Node 1")
        graph.add_node(node)
        self.assertEqual(graph.nodes, [node])

    def test_add_triplet(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "Relation", "Node 2")
        self.assertEqual(len(graph.edges), 1)
        self.assertEqual(len(graph.nodes), 2)

    def test_find_node(self):
        graph = GraphData()
        node = NodeData("Node 1")
        graph.add_node(node)
        found_node = graph.find_node("Node 1")
        self.assertEqual(found_node, node)

    def test_find_edge(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "Relation", "Node 2")
        found_edge = graph.find_edge("Relation")
        self.assertEqual(found_edge.get_relation(), "Relation")

    def test_get_adjacency_matrix(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "Relation", "Node 2")
        matrix = graph.get_adjacency_matrix()
        self.assertEqual(matrix, [[0, 1], [0, 0]])

    def test_synonymic_model(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "character", "Node 2")
        graph.add_new_triplet("Node 1", "characters", "Node 2")
        self.assertEqual(len(graph.edges), 1)

if __name__ == '__main__':
    unittest.main()