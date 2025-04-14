from sentence_transformers import SentenceTransformer, util
from kg_library.common import NodeData, EdgeData
from kg_library.db import Neo4jConnection
from collections import OrderedDict
from typing import Optional

class GraphData:
    def __init__(self):
        self.nodes: list[NodeData] = []
        self.edges: list[EdgeData] = []
        self.__synonymic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.triplets : list[tuple[NodeData, EdgeData, NodeData]] = []

    def add_node(self, node : NodeData):
        self.nodes.append(node)

    def add_edge(self, edge : EdgeData):
        self.edges.append(edge)

    def print(self):
        for triplet in self.triplets:
            head, relation, tail = triplet
            print(f"{head.name} -> {relation.get_relation()} -> {tail.name}")

    def add_loop_triplet(self, node : str):
        node = self.__find_or_create_node(node)
        loop_edge = EdgeData("loop")
        self.add_edge(loop_edge)
        loop_edge.set_ends(node, node)
        node.add_input(loop_edge)
        node.add_output(loop_edge)
        self.triplets.append((node, loop_edge, node))

    def add_loop_reversed_triplet(self):
        true_triplets = self.triplets.copy()
        for triplet in true_triplets:
            head, relation, tail = triplet
            self.add_new_triplet(tail.name, f"{relation.get_relation()}:reversed", head.name, False)
        for node in self.nodes:
            self.__add_loop_triplet(node)

    def __add_loop_triplet(self, node : NodeData):
        if not node.contains_edge("loop"):
            loop_edge = EdgeData("loop")
            self.add_edge(loop_edge)
            loop_edge.set_ends(node, node)
            node.add_input(loop_edge)
            node.add_output(loop_edge)
            self.triplets.append((node, loop_edge, node))

    def __check_for_synonyms(self, new_triplet : tuple[str, str, str]) -> bool:
        head, relation, tail = new_triplet
        new_embedding = self.__synonymic_model.encode(f"{head} {relation} {tail}")
        for existing_triplet in self.triplets:
            existing_head, existing_relation, existing_tail = existing_triplet
            existing_embedding = self.__synonymic_model.encode(f"{existing_head.name} {existing_relation.get_relation()} {existing_tail.name}")
            embedding_similarity = util.cos_sim(new_embedding, existing_embedding)
            #print(f"Existing triplet: {existing_triplet}", f"New triplet: {new_triplet}")
            #print(f"Embedding similarity: {embedding_similarity}")
            if embedding_similarity > 0.75:
                return True
        return False

    def add_new_triplet(self, head : str, relation : str, tail : str, check_synonyms : bool = True) -> None:
        print(f"Adding new triplet: {head} -> {relation} -> {tail}, check synonyms: {check_synonyms}")
        if check_synonyms:
            if self.__check_for_synonyms((head, relation, tail)):
                return
        head_node = self.__find_or_create_node(head)
        tail_node = self.__find_or_create_node(tail)
        relation_edge = EdgeData(relation)
        self.add_edge(relation_edge)
        relation_edge.set_ends(head_node, tail_node)
        head_node.add_output(relation_edge)
        tail_node.add_input(relation_edge)
        self.triplets.append((head_node, relation_edge, tail_node))

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
        return next((e for e in self.edges if e.get_relation() == edge), None)

    def fill_database(self, neo4j_connection : Neo4jConnection):
        print(self.triplets)
        for subj, rel, obj in self.triplets:
            query = (
                "MERGE (a:Entity {name: $subj}) "
                "MERGE (b:Entity {name: $obj}) "
                "MERGE (a)-[r:RELATION {type: $rel}]->(b)"
            )
            neo4j_connection.run_query(query, {"subj": subj, "rel": rel, "obj": obj})

    def get_adjacency_matrix(self) -> list[list[int]]:
        result = []
        for node in self.nodes:
            result.append([0] * len(self.nodes))
            for edge in node.get_outputs():
                target_node = edge.object
                result[-1][self.nodes.index(target_node)] = 1
        return result

    def get_unique_sets(self) -> tuple[list[str], list[str]]:
        unique_nodes = list(OrderedDict.fromkeys(node.name for node in self.nodes))
        unique_relations = list(OrderedDict.fromkeys(edge.get_relation() for edge in self.edges))
        return unique_nodes, unique_relations