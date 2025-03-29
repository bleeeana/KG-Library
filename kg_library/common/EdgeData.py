from kg_library.common.NodeData import NodeData

class EdgeData:
    def __init__(self, relation : str):
        self.__relation = relation
        self.__embedding = None
        self.weight = 0
        self.subject = None
        self.object = None

    def get_relation(self) -> str:
        return self.__relation

    def __str__(self):
        return f"edge {self.__relation}"

    def set_ends(self, head_node : NodeData, tail_node : NodeData):
        self.subject = head_node
        self.object = tail_node