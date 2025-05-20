from neo4j import GraphDatabase
from kg_library import get_config
from kg_library.common import GraphData

class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or get_config()["neo4j_uri"]
        self.user = user or get_config()["neo4j_auth"][0]
        self.password = password or get_config()["neo4j_auth"][1]
        print(f"Connecting to Neo4j at {self.uri} as {self.user} with password {self.password}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def fill_database(self, graph: GraphData):
        for subj, rel, obj in graph.triplets:
            query = (
                "MERGE (a:Entity {name: $subj}) "
                "SET a.type = $subj_type "
                "MERGE (b:Entity {name: $obj}) "
                "SET b.type = $obj_type " 
                "MERGE (a)-[r:RELATION {type: $rel}]->(b)"
            )
            self.run_query(
                query,
                {
                    "subj": subj.name,
                    "subj_type": subj.feature,
                    "rel": rel.get_relation(),
                    "obj": obj.name,
                    "obj_type": obj.feature,
                }
            )

    def load_graph(self) -> GraphData:
        query = """
            MATCH (s)-[r]->(o)
            RETURN 
                s.name as subject, 
                s.type as subject_type,
                type(r) as relation,
                o.name as object, 
                o.type as object_type
            """
        results = self.run_query(query)
        graph = GraphData()
        for result in results:
            subj = result["subject"]
            sub_type = result["subject_type"]
            rel = result["relation"]
            obj = result["object"]
            obj_type = result["object_type"]
            graph.add_new_triplet(head=subj, relation=rel, tail=obj, head_feature=sub_type, tail_feature=obj_type)

        return graph

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters)