from kg_library.common import EdgeData
class NodeData:
    def __init__(self, name, input_relation : EdgeData = None, output_relation : EdgeData = None, feature : str = "default"):
        self.__input_relations = []
        self.__output_relations = []
        self.name = name
        self.feature = feature
        if input_relation is not None:
            self.add_input(input_relation)
        if output_relation is not None:
            self.add_output(output_relation)

    def set_new_feature(self, feature : str):
        if feature != "default":
            self.feature = feature

    def add_input(self, edge : EdgeData):
        self.__input_relations.append(edge)

    def add_output(self, edge : EdgeData):
        self.__output_relations.append(edge)

    def get_inputs(self) -> list[EdgeData]:
        return self.__input_relations

    def get_outputs(self) -> list[EdgeData]:
        return self.__output_relations

    def contains_edge(self, edge : str) -> bool:
        return any(str(e) == edge for e in self.__input_relations) or any(str(e) == edge for e in self.__output_relations)

    def __str__(self) -> str:
        return f"node {self.name}"