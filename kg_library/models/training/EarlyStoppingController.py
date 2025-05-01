class EarlyStoppingController:
    def __init__(self, monitor_scores, patience=5, verbose=True):
        self.monitor_scores = monitor_scores
        self.patience = patience
        self.verbose = verbose
        self.best_scores = {metric: None for metric in monitor_scores}
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, current_scores: dict, model):
        improved = False

        for metric, config in self.monitor_scores.items():
            current = current_scores.get(metric)
            if current is None:
                raise ValueError(f"Metric '{metric}' not found in current_scores")

            best = self.best_scores[metric]
            mode = config.get("mode", "max")
            min_delta = config.get("min_delta", 0.0)

            if best is None:
                self.best_scores[metric] = current
                improved = True
            elif self._has_improved(current, best, mode, min_delta):
                if self.verbose:
                    print(f"{metric} improved from {best:.5f} to {current:.5f}")
                self.best_scores[metric] = current
                improved = True

        if improved:
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Early stop counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")

    def _has_improved(self, current, best, mode, min_delta):
        if mode == "max":
            return current > best + min_delta
        elif mode == "min":
            return current < best - min_delta
        else:
            raise ValueError(f"Invalid mode '{mode}'. Use 'max' or 'min'.")

    def restore_best_state(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
