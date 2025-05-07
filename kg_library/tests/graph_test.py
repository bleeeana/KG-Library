import unittest
from kg_library.common import GraphData, NodeData, GraphJSON
from kg_library import AppFacade

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

    def test_synonymic_model(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "character", "Node 2")
        graph.add_new_triplet("Node 1", "characters", "Node 2")
        self.assertEqual(len(graph.edges), 1)

    def test_saving_loading_minimal_graph(self):
        graph = GraphData()
        graph.add_new_triplet("Node 1", "Relation", "Node 2")
        graph.add_new_triplet("Node 2", "Relation2", "Node 3")
        graph.add_new_triplet("Node 3", "Relation3", "Node 1")
        GraphJSON.save(graph, "minitest.json")
        loaded_graph = GraphJSON.load("minitest.json")
        self.assertEqual(len(graph.nodes), len(loaded_graph.nodes))
        self.assertEqual([node.name for node in graph.nodes], [node.name for node in loaded_graph.nodes])
        self.assertEqual([node.feature for node in graph.nodes],
                         [node.feature for node in loaded_graph.nodes])
        self.assertEqual(len(graph.edges), len(loaded_graph.edges))
        self.assertEqual([edge.get_relation() for edge in graph.edges],
                         [edge.get_relation() for edge in loaded_graph.edges])
        self.assertEqual(len(graph.triplets), len(loaded_graph.triplets))

    def test_save_to_json(self):
        app_facade = AppFacade()
        app_facade.generate_graph_for_learning()
        GraphJSON.save(app_facade.graph, "base_graph.json")
        loaded_graph = GraphJSON.load("base_graph.json")
        self.assertEqual(len(app_facade.graph.nodes), len(loaded_graph.nodes))
        self.assertEqual([node.name for node in app_facade.graph.nodes], [node.name for node in loaded_graph.nodes])
        self.assertEqual([node.feature for node in app_facade.graph.nodes], [node.feature for node in loaded_graph.nodes])
        self.assertEqual(len(app_facade.graph.edges), len(loaded_graph.edges))
        self.assertEqual([edge.get_relation() for edge in app_facade.graph.edges], [edge.get_relation() for edge in loaded_graph.edges])
        self.assertEqual(len(app_facade.graph.triplets), len(loaded_graph.triplets))


    def test_balanced_graph(self):
        graph = GraphJSON.load("base_graph.json")
        count = {}
        for node in graph.nodes:
            if node.feature in count.keys():
                count[node.feature] += 1
            else:
                count[node.feature] = 1

        print(count)
        self.assertLessEqual(count["default"] / len(graph.nodes), 0.3)

    def test_cloning(self):
        graph = GraphJSON.load("base_graph.json")
        cloned_graph = graph.clone()
        self.assertEqual(len(graph.nodes), len(cloned_graph.nodes))
        self.assertEqual([node.name for node in graph.nodes], [node.name for node in cloned_graph.nodes])
        self.assertEqual([node.feature for node in graph.nodes], [node.feature for node in cloned_graph.nodes])
        self.assertEqual(len(graph.edges), len(cloned_graph.edges))
        self.assertEqual([edge.get_relation() for edge in graph.edges], [edge.get_relation() for edge in cloned_graph.edges])
        self.assertEqual(len(graph.triplets), len(cloned_graph.triplets))

if __name__ == '__main__':
    unittest.main()