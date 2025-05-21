from kg_library.common import NodeData, EdgeData
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

    def add_new_triplet_without_check(self, head : NodeData, relation : EdgeData, tail : NodeData):
        self.triplets.append((head, relation, tail))
        head.add_output(relation)
        tail.add_input(relation)

    def add_new_triplet_direct(self, head : NodeData, relation : EdgeData, tail : NodeData):
        if not self.has_triplet(head.name, relation.get_relation(), tail.name):
            self.triplets.append((head, relation, tail))
        head.add_output(relation)
        tail.add_input(relation)
        if head.name not in [node.name for node in self.nodes]:
            self.nodes.append(head)
        if tail.name not in [node.name for node in self.nodes]:
            self.nodes.append(tail)
        if relation.get_relation() not in [edge.get_relation() for edge in self.edges]:
            self.edges.append(relation)

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

    def has_triplet_direct(self, head : NodeData, relation : EdgeData, tail : NodeData) -> bool:
        return (head, relation, tail) in self.triplets

    def has_triplet(self, head : str, relation : str, tail : str) -> bool:
        return any((triplet[0].name, triplet[1].get_relation(), triplet[2].name) == (head, relation, tail) for triplet in self.triplets)

    def merge_with_another_graph(self, other_graph):
        merged = self.clone()
        for triplet in other_graph.triplets:
            merged.add_new_triplet_direct(triplet[0], triplet[1], triplet[2])
        return merged

    def get_node_names(self):
        return [node.name for node in self.nodes]

    def clone(self):
        new_graph = GraphData()
        nodes_map = {}
        edges_map = {}
        for node in self.nodes:
            new_node = NodeData(node.name, feature=node.feature)
            nodes_map[node] = new_node
            new_graph.nodes.append(new_node)

        for edge in self.edges:
            new_edge = EdgeData(edge.get_relation())
            edges_map[edge] = new_edge
            new_graph.edges.append(new_edge)

        for node, new_node in nodes_map.items():
            for input_rel in node.get_inputs():
                new_node.add_input(edges_map[input_rel])
            for output_rel in node.get_outputs():
                new_node.add_output(edges_map[output_rel])

        for head, relation, tail in self.triplets:
            new_graph.triplets.append((nodes_map[head], edges_map[relation], nodes_map[tail]))

        return new_graph