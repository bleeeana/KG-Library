import os
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = os.getenv("NEO4J_AUTH").split('/')
EMAIL = os.getenv("EMAIL")
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
BATCHSIZE = 128
__version__ = "0.1.0"
def get_config():
    return {
        "neo4j_uri": NEO4J_URI,
        "neo4j_auth": NEO4J_AUTH,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "email": EMAIL,
        "batch_size": BATCHSIZE
    }