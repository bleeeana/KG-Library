from kg_library.common import EdgeData
from typing import List
class NodeData:
    def __init__(self, name, input : EdgeData = None, output : EdgeData = None):
        self.__inputs = []
        self.__outputs = []
        self.name = name
        self.__inputs.append(input)
        self.__outputs.append(output)
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

    def __str__(self) -> str:
        return f"node {self.name}"