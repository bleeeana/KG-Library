class EarlyStoppingController:
    def __init__(self, patience=8, min_delta=0.001, mode='max', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif self.has_improved(score):
            self.best_state = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Validation score improved from {self.best_score} to {score}")
            self.best_score = score
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f"Early stopping triggered after {self.counter} epochs")
                self.early_stop = True

    def has_improved(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta

    def restore_best_state(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)