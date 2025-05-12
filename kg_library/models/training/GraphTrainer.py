from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
import os, datetime
from kg_library.models import EarlyStoppingController
import torch.nn.functional as F
import torch
from kg_library.models import EmbeddingPreprocessor, GraphNN
from kg_library.utils import PathManager
from torch.serialization import add_safe_globals
import torch_geometric.data.storage
class GraphTrainer:
    def __init__(self, model, train_loader, val_loader, epochs=100, lr=1e-3, patience=4):
        self.model = model
        self.device = model.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.current_epoch = 0
        self.global_step = 0
        log_dir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = SummaryWriter(log_dir=log_dir)
        self.early_stopper = EarlyStoppingController(
            monitor_scores={
                "auc": {"mode": "max", "min_delta": 0.001},
                "loss": {"mode": "min", "min_delta": 0.001}
            },
            patience=4
        )
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=patience, factor=0.5)

        print(f"TensorBoard logs will be saved to: {log_dir}")
        print("To view results, run in terminal:")
        print(f"tensorboard --logdir={os.path.abspath('runs')}")
        print("Then open http://localhost:6006 in your browser")

    def train(self, save: bool = True):
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            all_scores = []
            all_labels = []

            for batch_idx, batch in enumerate(self.train_loader):
                batch_loss, batch_scores, batch_labels = self._process_batch(batch, is_training=True)
                
                epoch_loss += batch_loss
                all_scores.append(batch_scores)
                all_labels.append(batch_labels)

                if batch_idx % 5 == 0:
                    self.writer.add_scalar("Loss/train_batch", batch_loss, self.global_step)
                    if len(batch_scores) > 0 and len(batch_labels) > 0:
                        batch_auc = self._calculate_auc(batch_scores, batch_labels)
                        self.writer.add_scalar("AUC/train_batch", batch_auc, self.global_step)
                
                self.global_step += 1

            train_scores = torch.cat(all_scores)
            train_labels = torch.cat(all_labels)
            train_auc = self._calculate_auc(train_scores, train_labels)

            val_auc, val_loss = self.evaluate(self.val_loader)
            self.scheduler.step(val_auc)

            self.writer.add_scalar("Loss/train", epoch_loss / len(self.train_loader), epoch)
            self.writer.add_scalar("AUC/train", train_auc, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("AUC/val", val_auc, epoch)
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], epoch)

            print(f"Epoch {epoch:03d} | Loss: {epoch_loss / len(self.train_loader):.4f} | "
                  f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")

            self.early_stopper(
                {"auc": val_auc, "loss": val_loss},
                self.model
            )

            if self.early_stopper.early_stop:
                print("Early stopping, saving the best configuration")
                self.early_stopper.restore_best_state(self.model)
                if save:
                    self.save_with_config()
                break

            self.current_epoch += 1
            if save:
                self.save_with_config()
                print("Model saved")

        self.writer.close()

    def _process_batch(self, batch, is_training=True):
        torch.cuda.empty_cache()
        batch = batch.to(self.device)
        graph = self.model.preprocessor.create_hetero_from_batch(batch)
        output = self.model(graph)

        h = output["entity"][batch.head.squeeze()]
        t = output["entity"][batch.tail.squeeze()]
        r = self.model.get_relation_embedding(batch.edge_attr.squeeze())
        scores = self.model.score_function(h, t, r)
        loss = F.binary_cross_entropy_with_logits(scores, batch.y)

        if is_training and self.global_step % 10 == 0:
            self.writer.add_scalar("Model/gamma", self.model.gamma.item(), self.global_step)

        if is_training:
            gamma_reg = 0.01 * (self.model.gamma - torch.tensor(10.0, device=self.device)) ** 2
            loss += gamma_reg

        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            total_grad_norm = self._calculate_gradient_norm()
            self.writer.add_scalar("grad_norms/total", total_grad_norm, self.global_step)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        return loss.item(), scores.detach().cpu(), batch.y.cpu()
    
    def _calculate_gradient_norm(self):
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        return total_grad_norm ** 0.5
    
    def _calculate_auc(self, scores, labels):
        if len(scores) == 0 or len(labels) == 0:
            return 0.0
        try:
            return roc_auc_score(labels, torch.sigmoid(scores))
        except ValueError:
            return 0.5

    def evaluate(self, loader):
        self.model.eval()
        scores_list, labels_list = [], []
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch_loss, batch_scores, batch_labels = self._process_batch(batch, is_training=False)
                total_loss += batch_loss
                total_batches += 1
                scores_list.append(torch.sigmoid(batch_scores))
                labels_list.append(batch_labels)

        scores = torch.cat(scores_list)
        labels = torch.cat(labels_list)
        val_auc = roc_auc_score(labels, scores)
        val_loss = total_loss / total_batches

        return val_auc, val_loss

    def save_with_config(self, model_path="model_with_config.pt", graph_path="base_graph.json"):
        model_path = PathManager.get_model_path(model_path)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': self.model.get_config(),
            'preprocessor_config': self.model.preprocessor.get_config(),
            'graph' : graph_path
        }, model_path)

    @staticmethod
    def load_model_for_training(preprocessor : EmbeddingPreprocessor, model_path="model_with_config.pt", map_location='cuda', train_loader = None, val_loader = None):
        add_safe_globals([torch_geometric.data.storage.BaseStorage])
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
        model = GraphNN(preprocessor, hidden_dim=checkpoint["model_config"]["hidden_dim"], num_layers=checkpoint["model_config"]["num_layers"], dropout=checkpoint["model_config"]["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(preprocessor.device)
        return GraphTrainer(model, train_loader, val_loader)