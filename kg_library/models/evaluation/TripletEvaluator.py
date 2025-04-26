from kg_library.models import GraphNN, EmbeddingPreprocessor
import torch
from sklearn.metrics import roc_auc_score

class TripletEvaluator:
    def __init__(self, model : GraphNN):
        self.model = model
        self.device = model.device
        self.preprocessor = model.preprocessor

    def evaluate(self, new_triplets : list[tuple[str,str,str]], threshold : float = 0.75):
        self.model.eval()
        results = []
