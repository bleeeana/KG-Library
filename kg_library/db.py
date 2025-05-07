from neo4j import GraphDatabase
from kg_library import get_config


class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or get_config()["neo4j_uri"]
        self.user = user or get_config()["neo4j_auth"][0]
        self.password = password or get_config()["neo4j_auth"][1]
        print(f"Connecting to Neo4j at {self.uri} as {self.user} with password {self.password}")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters)