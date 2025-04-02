from kg_library.common import EdgeData
from typing import List
class NodeData:
    def __init__(self, name, input : EdgeData = None, output : EdgeData = None):
        self.__inputs = []
        self.__outputs = []
        self.name = name
        if input is not None:
            self.add_input(input)
        if output is not None:
            self.add_output(output)
        print(self.__inputs)
        self.__embedding = None

    def add_input(self, edge : EdgeData):
        self.__inputs.append(edge)

    def add_output(self, edge : EdgeData):
        self.__outputs.append(edge)

    def get_inputs(self) -> List[EdgeData]:
        return self.__inputs

    def get_outputs(self) -> List[EdgeData]:
        return self.__outputs

    def contains_edge(self, edge : str) -> bool:
        return any(str(e) == edge for e in self.__inputs) or any(str(e) == edge for e in self.__outputs)

    def __str__(self) -> str:
        return f"node {self.name}"