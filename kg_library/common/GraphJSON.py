from kg_library.common import GraphData, EdgeData, NodeData
import json

class NodeJSON:
    @staticmethod
    def to_json(node : NodeData) -> dict:
        return {
            "name" : node.name,
            "feature" : node.feature,
            "input_relations" : [EdgeJSON.to_json(edge) for edge in node.get_inputs()],
            "output_relations" : [EdgeJSON.to_json(edge) for edge in node.get_outputs()]
        }

    @staticmethod
    def from_json(node_dict : dict) -> NodeData:
        node = NodeData(node_dict["name"], feature=node_dict["feature"])
        for edge in node_dict["input_relations"]:
            node.add_input(EdgeJSON.from_json(edge))
        for edge in node_dict["output_relations"]:
            node.add_output(EdgeJSON.from_json(edge))
        return node


class EdgeJSON:
    @staticmethod
    def to_json(edge : EdgeData) -> dict:
        return {
            "relation" : edge.get_relation(),
            "subject" : NodeJSON.to_json(edge.subject),
            "object" : NodeJSON.to_json(edge.object)
        }

    @staticmethod
    def from_json(edge_dict : dict) -> EdgeData:
        return EdgeData(edge_dict["relation"]).set_ends(NodeJSON.from_json(edge_dict["subject"]), NodeJSON.from_json(edge_dict["object"]))

class GraphJSON:
    @staticmethod
    def to_json(graph : GraphData) -> str:
        dict = {
            "nodes" : [NodeJSON.to_json(node) for node in graph.nodes],
            "edges" : [EdgeJSON.to_json(edge) for edge in graph.edges],
            "triplets" : []
        }
        for triplet in graph.triplets:
            dict["triplets"].append({
                "head" : graph.nodes.index(triplet[0]),
                "relation" : graph.edges.index(triplet[1]),
                "tail" : graph.edges.index(triplet[1])
            })
        return json.dumps(dict, indent=2)

    @staticmethod
    def from_json( json_dict : str) -> GraphData:
        graph_dict = json.loads(json_dict)
        graph = GraphData()
        for node_dict in graph_dict["nodes"]:
            graph.add_node(NodeJSON.from_json(node_dict))
        for edge_dict in graph_dict["edges"]:
            graph.add_edge(EdgeJSON.from_json(edge_dict))
        for triplet_dict in graph_dict["triplets"]:
            head = graph.nodes[triplet_dict["head"]]
            relation = graph.edges[triplet_dict["relation"]]
            tail = graph.nodes[triplet_dict["tail"]]
            graph.triplets.append((head, relation, tail))
        return graph

    @staticmethod
    def save(graph : GraphData, filepath : str):
        with open(filepath, "w") as f:
            f.write(GraphJSON.to_json(graph))

    @staticmethod
    def load(filepath : str) -> GraphData:
        with open(filepath, "r") as f:
            return GraphJSON.from_json(f.read())