import os

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/testpassword")
EMBEDDING_DIM = 300
HIDDEN_DIM = 128
__version__ = "0.1.0"
def get_config():
    return {
        "neo4j_uri": NEO4J_URI,
        "neo4j_auth": NEO4J_AUTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
    }
