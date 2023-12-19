

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from Model import mse, mae, saveModel
from Dataloader import DataLoaderQM9



class Trainer:

    
    def __init__(self, Model: torch.nn.Module, loss: any, target: int, optimizer: torch.optim, Dataloader, scheduler: torch.optim, device: torch.device = "cpu"):

        self.Model = Model
        self.target = target
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_set = Dataloader
        self.valid_set = Dataloader.get_val()
        self.test_set = Dataloader.get_test()


        self.learning_curve = []
        self.valid_perf= []
        self.learning_rates = []
        self.summaries, self.summaries_axes = plt.subplots(1,3, figsize=(10,5))


    def _train_epoch(self) -> None:

        for batch_num, batch in enumerate(self.train_set):
            # Using our chosen device
            batch = {key: value.to(self.device) for key, value in batch.items()}
            targets = batch["targets"][:, self.target].to(self.device).unsqueeze(dim=-1)

            # Backpropagate using the selected loss
            outputs = self.Model(batch)
            loss = self.loss(outputs, targets)

            if batch_num % 100 == 0:
                print(f"Current loss {loss} Current batch {batch_num}/{len(self.train_set)} "
                    f"({100 * batch_num / len(self.train_set):.2f}%)")

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_num == len(self.train_set):
                self.learning_curve.append(loss.item())
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)

        # Cleanup at the end of the epoch
        torch.cuda.empty_cache()


    def _eval_model(self) -> torch.Tensor:
        val_loss = torch.zeros(1, device=self.device)

        with torch.no_grad():
            if not self.valid_set:
                return torch.zeros_like(val_loss)

            for batch_num, batch in enumerate(self.valid_set):
                if not batch:
                    continue

                batch = {key: value.to(self.device) for key, value in batch.items()}
                targets = batch["targets"][:, self.target].unsqueeze(dim=-1)
                pred_val = self.Model(batch)

                current_batch_loss = self.loss(pred_val, targets).item()
                val_loss += current_batch_loss

            return val_loss / (batch_num + 1)  # Add 1 to avoid division by zero if the loop is not executed


    def _train(self, num_epoch: int = 100, early_stopping: int = 30, alpha: float = 0.9) -> str:

        patience = 0
        min_loss = float('inf')  # Initialize min_loss with a large value
        best_model_path = None  # where the best model is

        for epoch in range(1, num_epoch + 1):
            self._train_epoch()
            # Validate at the end of an epoch
            val_loss = self._eval_model()
            print(f"### End of epoch: Validation loss for {epoch} is {val_loss.item()}")
            self.scheduler.step(val_loss)
            val_loss_s = val_loss.item()

            # Exponential smoothing for validation
            self.valid_perf.append(val_loss_s if epoch == 1 else alpha * val_loss_s + (1 - alpha) * self.valid_perf[-1])

            if torch.isfinite(val_loss) and val_loss < min_loss:
                patience = 0
                min_loss = val_loss_s
                saveModel(self.Model, path="./best_PaiNNModel.pth")
                best_model_path = "./best_PaiNNModel.pth"
            else:
                patience += 1
                if patience >= early_stopping:
                    break

        return best_model_path




            
            