from neo4j import GraphDatabase
from __init__ import NEO4J_URI, NEO4J_AUTH


class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_AUTH[0]
        self.password = password or NEO4J_AUTH[1]
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            return session.run(query, parameters)