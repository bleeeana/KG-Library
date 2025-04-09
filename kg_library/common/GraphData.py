from typing import Optional, Tuple
from sentence_transformers import SentenceTransformer, util
from kg_library.common import NodeData, EdgeData
from kg_library.db import Neo4jConnection
from typing import List

class GraphData:
    def __init__(self):
        self.nodes: List[NodeData] = []  # Список, содержащий элементы типа NodeData
        self.edges: List[EdgeData] = []

    def add_node(self, node : NodeData):
        self.nodes.append(node)

    def add_edge(self, edge : EdgeData):
        self.edges.append(edge)

    def print(self):
        print("Nodes:")
        for n in self.nodes:
            print(f"  {n}")
        print("Edges:")
        for e in self.edges:
            print(f"  {e}")

    def add_loop_triplet(self, node : str):
        node = self.__find_or_create_node(node)
        loop_edge = EdgeData("loop")
        self.add_edge(loop_edge)
        loop_edge.set_ends(node, node)
        node.add_input(loop_edge)
        node.add_output(loop_edge)

    def __add_loop_triplet(self, node : NodeData):
        if  not node.contains_edge("loop"):
            loop_edge = EdgeData("loop")
            self.add_edge(loop_edge)
            loop_edge.set_ends(node, node)
            node.add_input(loop_edge)
            node.add_output(loop_edge)

    def __check_for_synonyms(self, new_triplet : Tuple[str, str, str], model : SentenceTransformer) -> bool:
        head, relation, tail = new_triplet
        existing_triplets = self.__get_triplets()
        for existing_triplet in existing_triplets:
            existing_head, existing_relation, existing_tail = existing_triplet
            existing_triplet_embedding = model.encode(f"{existing_head} {existing_relation} {existing_tail}")
            similarity = util.cos_sim(existing_triplet_embedding, model.encode(f"{head} {relation} {tail}"))
            if similarity > 0.9:
                return True

        return False

    def add_new_triplet(self, head : str, relation : str, tail : str, model : SentenceTransformer) -> None:
        if self.__check_for_synonyms((head, relation, tail), model):
            return
        head_node = self.__find_or_create_node(head)
        self.__add_loop_triplet(head_node)
        tail_node = self.__find_or_create_node(tail)
        self.__add_loop_triplet(tail_node)
        relation_edge = EdgeData(relation)
        self.add_edge(relation_edge)
        relation_edge.set_ends(head_node, tail_node)
        head_node.add_output(relation_edge)
        tail_node.add_input(relation_edge)

    def __find_or_create_node(self, name : str) -> NodeData:
        if self.find_node(name) is None:
            node = NodeData(name)
            self.add_node(node)
            return node
        else:
            return self.find_node(name)

    def find_node(self, node: str) -> Optional[NodeData]:
        return next((n for n in self.nodes if n.name == node), None)

    def find_edge(self, edge: str) -> Optional[EdgeData]:
        return next((e for e in self.edges if str(e) == edge), None)

    def fill_database(self, neo4j_connection : Neo4jConnection):
        triplets = self.__get_triplets()
        print(triplets)
        for subj, rel, obj in triplets:
            query = (
                "MERGE (a:Entity {name: $subj}) "
                "MERGE (b:Entity {name: $obj}) "
                "MERGE (a)-[r:RELATION {type: $rel}]->(b)"
            )
            neo4j_connection.run_query(query, {"subj": subj, "rel": rel, "obj": obj})

    def get_adjacency_matrix(self) -> List[List[int]]:
        result = []
        for node in self.nodes:
            result.append([0] * len(self.nodes))
            for edge in node.get_outputs():
                target_node = edge.object
                result[-1][self.nodes.index(target_node)] = 1
        return result

    def __get_triplets(self) -> List[Tuple[str, Optional[str], Optional[str]]]:
        triplets_set = set()
        for node in self.nodes:
            for edge in node.get_outputs():
                target_node = edge.object
                triplets_set.add((node.name, edge.get_relation(), target_node.name))
            for edge in node.get_inputs():
                source_node = edge.subject
                triplets_set.add((source_node.name, edge.get_relation(), node.name))
        return list(triplets_set)

