import unittest
from kg_library.common import GraphData, EdgeData, NodeData

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
        node1 = NodeData("Node 1")
        node2 = NodeData("Node 2")
        edge = EdgeData("Relation")
        edge.set_ends(node1, node2)
        graph.add_edge(edge)
        found_edge = graph.find_edge("Relation")
        self.assertEqual(found_edge, edge)

    def test_get_adjacency_matrix(self):
        graph = GraphData()
        node1 = NodeData("Node 1")
        node2 = NodeData("Node 2")
        edge = EdgeData("Relation")
        edge.set_ends(node1, node2)
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_edge(edge)
        matrix = graph.get_adjacency_matrix()
        self.assertEqual(matrix, [[0, 1], [0, 0]])

if __name__ == '__main__':
    unittest.main()