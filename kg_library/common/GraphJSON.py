from kg_library.common import GraphData, EdgeData, NodeData
import json
import os
from kg_library.utils import PathManager
class NodeJSON:
    @staticmethod
    def to_json(node : NodeData) -> dict:
        return {
            "name" : node.name,
            "feature" : node.feature,
        }

    @staticmethod
    def from_json(node_dict : dict) -> NodeData:
        return NodeData(node_dict["name"], feature=node_dict["feature"])


class EdgeJSON:
    @staticmethod
    def to_json(edge : EdgeData) -> dict:
        return {
            "relation" : edge.get_relation(),
        }

    @staticmethod
    def from_json(edge_dict : dict) -> EdgeData:
        return EdgeData(edge_dict["relation"])

class GraphJSON:
    @staticmethod
    def to_json(graph : GraphData) -> str:
        dict_from_json = {
            "nodes" : [NodeJSON.to_json(node) for node in graph.nodes],
            "edges" : [EdgeJSON.to_json(edge) for edge in graph.edges],
            "triplets" : []
        }
        for triplet in graph.triplets:
            dict_from_json["triplets"].append({
                "head" : graph.nodes.index(triplet[0]),
                "relation" : graph.edges.index(triplet[1]),
                "tail" : graph.nodes.index(triplet[2])
            })
        return json.dumps(dict_from_json, indent=2)

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
            graph.add_new_triplet_direct(head, relation, tail)
        #graph.print()
        return graph

    @staticmethod
    def save(graph: GraphData, filepath: str):

        if os.path.dirname(filepath) == "":
            PathManager.ensure_dirs()
            filepath = PathManager.get_output_path(filepath)

        with open(filepath, "w") as f:
            f.write(GraphJSON.to_json(graph))

    @staticmethod
    def load(filepath: str) -> GraphData:
        

        original_path = filepath

        if os.path.dirname(filepath) == "":
            output_path = PathManager.get_output_path(filepath)
            input_path = PathManager.get_input_path(filepath)

            if os.path.exists(output_path):
                filepath = output_path
            elif os.path.exists(input_path):
                filepath = input_path

        if not os.path.exists(filepath):
            print(f"File {original_path} not found")
            base_path = PathManager.get_input_path("base_graph.json")
            if os.path.exists(base_path):
                filepath = base_path
            else:
                filepath = "base_graph.json"  

        with open(filepath, "r") as f:
            return GraphJSON.from_json(f.read())
