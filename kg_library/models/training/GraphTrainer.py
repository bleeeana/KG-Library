from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import os, datetime
from kg_library.models import EarlyStoppingController
from kg_library.common import GraphJSON
import torch.nn.functional as F
import torch
from kg_library.models import EmbeddingPreprocessor, GraphNN

class GraphTrainer:
    def __init__(self, model, train_loader, val_loader, epochs=100, lr=1e-3, patience=10):
        self.model = model
        self.device = model.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.current_epoch = 0
        log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir=log_dir)
        self.early_stopper = EarlyStoppingController()
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=patience, factor=0.5)

        print(f"TensorBoard logs will be saved to: {log_dir}")
        print("To view results, run in terminal:")
        print(f"tensorboard --logdir={os.path.abspath('runs')}")
        print("Then open http://localhost:6006 in your browser")

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            all_scores = []
            all_labels = []

            for batch in self.train_loader:
                torch.cuda.empty_cache()
                batch = batch.to(self.device)
                graph = self.model.preprocessor.create_hetero_from_batch(batch)
                output = self.model(graph)

                h = output["entity"][batch.head.squeeze()]
                t = output["entity"][batch.tail.squeeze()]
                r = self.model.get_relation_embedding(batch.edge_attr.squeeze())

                scores = self.model.score_function(h, t, r)
                loss = F.binary_cross_entropy_with_logits(scores, batch.y)

                self.optimizer.zero_grad()
                loss.backward()

                total_grad_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                self.writer.add_scalar("grad_norms/total", total_grad_norm, epoch)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                all_scores.append(scores.detach().cpu())
                all_labels.append(batch.y.cpu())

            train_scores = torch.cat(all_scores)
            train_labels = torch.cat(all_labels)
            train_auc = roc_auc_score(train_labels, train_scores.sigmoid())

            val_auc = self.evaluate(self.val_loader)
            self.scheduler.step(val_auc)

            self.writer.add_scalar("Loss/train", epoch_loss / len(self.train_loader), epoch)
            self.writer.add_scalar("AUC/train", train_auc, epoch)
            self.writer.add_scalar("AUC/val", val_auc, epoch)
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch:03d} | Loss: {epoch_loss / len(self.train_loader):.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

            self.early_stopper(val_auc, self.model)

            if self.early_stopper.early_stop:
                print("Early stopping, saving the best configuration")
                self.early_stopper.restore_best_state(self.model)
                self.save_with_config()
                break

            self.current_epoch += 1
            self.save_with_config()
            print("Model saved")

        self.writer.close()

    def evaluate(self, loader):
        self.model.eval()
        scores_list, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                graph = self.model.preprocessor.create_hetero_from_batch(batch)
                output = self.model(graph)

                h = output["entity"][batch.head.squeeze()]
                t = output["entity"][batch.tail.squeeze()]
                r = self.model.get_relation_embedding(batch.edge_attr.squeeze())

                scores = self.model.score_function(h, t, r)
                scores_list.append(scores.sigmoid().cpu())
                labels_list.append(batch.y.cpu())

        scores = torch.cat(scores_list)
        labels = torch.cat(labels_list)
        return roc_auc_score(labels, scores)

    def save_with_config(self, model_path="model_with_config.pt", graph_path="test.json"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model.get_config(),
            'preprocessor_config': self.model.preprocessor.get_config(),
            'graph' : graph_path
        }, model_path)


    # для дообучения
    @staticmethod
    def load_model_for_training(model_path="model_with_config.pt", map_location='cuda', train_loader = None, val_loader = None):
        checkpoint = torch.load(model_path, map_location=map_location)
        graph = GraphJSON.load(checkpoint["graph"])
        preprocessor = EmbeddingPreprocessor(graph)
        preprocessor.load_config(checkpoint["preprocessor_config"])
        model = GraphNN(preprocessor, hidden_dim=checkpoint["model_config"]["hidden_dim"], num_layers=checkpoint["model_config"]["num_layers"], dropout=checkpoint["model_config"]["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(preprocessor.device)
        return GraphTrainer(model, train_loader, val_loader)
