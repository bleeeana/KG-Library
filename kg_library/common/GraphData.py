from kg_library.common import NodeData, EdgeData
from kg_library.db import Neo4jConnection
from typing import Optional

class GraphData:
    def __init__(self):
        self.nodes: list[NodeData] = []
        self.edges: list[EdgeData] = []
        self.triplets : list[tuple[NodeData, EdgeData, NodeData]] = []

    def add_node(self, node : NodeData):
        self.nodes.append(node)

    def add_edge(self, edge : EdgeData):
        self.edges.append(edge)

    def print(self):
        for triplet in self.triplets:
            head, relation, tail = triplet
            print(f"{head.name}({head.feature}) -> {relation.get_relation()} -> {tail.name}({tail.feature})")

    def add_loop_reversed_triplet(self):
        true_triplets = self.triplets.copy()
        for triplet in true_triplets:
            head, relation, tail = triplet
            self.add_new_triplet(tail.name, f"{relation.get_relation()}:reversed", head.name, False)
        for node in self.nodes:
            self._add_loop_triplet(node)

    def _add_loop_triplet(self, node : NodeData):
        if not node.contains_edge("loop"):
            loop_edge = self._find_or_create_edge("loop")
            node.add_input(loop_edge)
            node.add_output(loop_edge)
            self.triplets.append((node, loop_edge, node))

    def _check_for_synonyms(self, new_triplet : tuple[str, str, str]) -> bool:
        # Заглушка
        return False

    def add_new_triplet_direct(self, head : NodeData, relation : EdgeData, tail : NodeData):
        self.triplets.append((head, relation, tail))
        head.add_output(relation)
        tail.add_input(relation)

    def add_new_triplet(self, head : str, relation : str, tail : str, check_synonyms : bool = True, head_feature : str = "default", tail_feature : str = "default"):
        #print(f"Adding new triplet: {head}({head_feature}) -> {relation} -> {tail}({tail_feature}), check synonyms: {check_synonyms}")
        if check_synonyms:
            if self._check_for_synonyms((head, relation, tail)):
                return
        head_node = self._find_or_create_node(head, head_feature)
        tail_node = self._find_or_create_node(tail, tail_feature)
        relation_edge = self._find_or_create_edge(relation)
        head_node.add_output(relation_edge)
        tail_node.add_input(relation_edge)
        self.triplets.append((head_node, relation_edge, tail_node))

    def _find_or_create_node(self, name : str, feature : str = "default") -> NodeData:
        exist_node = self.find_node(name)
        if exist_node is None:
            node = NodeData(name, feature=feature)
            self.add_node(node)
            return node
        else:
            exist_node.set_new_feature(feature)
            return exist_node

    def _find_or_create_edge(self, relation : str) -> EdgeData:
        exist_edge = self.find_edge(relation)
        if exist_edge is None:
            edge = EdgeData(relation)
            self.add_edge(edge)
            return edge
        else:
            return exist_edge

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

    def has_triplet_direct(self, head : NodeData, relation : EdgeData, tail : NodeData) -> bool:
        return (head, relation, tail) in self.triplets

    def has_triplet(self, head : str, relation : str, tail : str) -> bool:
        return any((triplet[0].name, triplet[1].get_relation(), triplet[2].name) == (head, relation, tail) for triplet in self.triplets)

    def clone(self):
        new_graph = GraphData()
        new_graph.entities = self.nodes.copy()
        new_graph.relations = self.edges.copy()
        new_graph.triplets = self.triplets.copy()
        return new_graph