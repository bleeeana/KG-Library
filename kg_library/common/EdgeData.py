class EdgeData:
    def __init__(self, relation : str):
        self._relation = relation
        self._subject = None
        self._object = None

    def get_relation(self) -> str:
        return self._relation

    def __str__(self) -> str:
        return f"edge {self._relation}"
