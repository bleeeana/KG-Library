from kg_library.common import EdgeData
from typing import List
class NodeData:
    def __init__(self, name, input_relation : EdgeData = None, output_relation : EdgeData = None):
        self.__input_relations = []
        self.__output_relations = []
        self.name = name
        if input_relation is not None:
            self.add_input(input_relation)
        if output_relation is not None:
            self.add_output(output_relation)
        print(self.__input_relations)
        self.__embedding = None

    def add_input(self, edge : EdgeData):
        self.__input_relations.append(edge)

    def add_output(self, edge : EdgeData):
        self.__output_relations.append(edge)

    def get_inputs(self) -> List[EdgeData]:
        return self.__input_relations

    def get_outputs(self) -> List[EdgeData]:
        return self.__output_relations

    def contains_edge(self, edge : str) -> bool:
        return any(str(e) == edge for e in self.__input_relations) or any(str(e) == edge for e in self.__output_relations)

    def __str__(self) -> str:
        return f"node {self.name}"