import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

def mse(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean((preds - targets).square())

def mae(preds: torch.Tensor, targets: torch.Tensor):
    return torch.mean(torch.abs(preds - targets))

class Trainer:
    """ Responsible for training loop and validation """
    
    def __init__(self, model: torch.nn.Module, loss: any, target: int, optimizer: torch.optim, data_loader, scheduler: torch.optim, device: torch.device = "cpu"):
        """ Constructor
        Args:   
            model: Model to use (usually PaiNN)
            loss: loss function to use during traning
            target: the index of the target we want to predict 
            optimizer: optimizer to use during training
            data_loader: DataLoader object containing train/val/test sets
            device: device on which to execute the training
        """
        self.model = model
        self.target = target
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_set = data_loader
        self.valid_set = data_loader.get_val()
        self.test_set = data_loader.get_test()
        self.learning_curve = []
        self.valid_perf= []
        self.learning_rates = []
        self.summaries, self.summaries_axes = plt.subplots(1,3, figsize=(10,5))


    def _train_epoch(self) -> dict:
        """ Training logic for an epoch
        """
        for i, batch in enumerate(self.train_set):
            # Using our chosen device
            targets = batch["targets"][:, self.target].to(self.device).unsqueeze(dim=-1)

            # Backpropagate using the selected loss
            outputs = self.model(batch)
            loss = self.loss(outputs, targets)

            if i%100 == 0:
                print(f"Current loss {loss} Current batch {i}/{len(self.train_set)} ({100*i/len(self.train_set):.2f}%)")


            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if i == len(self.train_set) - 1:
                self.learning_curve.append(loss.item())
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)

            # Cleanup at the end of the batch
            del batch
            del targets
            del loss
            del outputs
            torch.cuda.empty_cache()

    def _eval_model(self):
        val_loss = torch.zeros(1).to(self.device)

        with torch.no_grad():
            for i, batch in enumerate(self.valid_set):
                pred_val = self.model(batch)
                targets = batch["targets"][:, self.target].to(self.device).unsqueeze(dim=-1)
                
                val_loss = val_loss + self.loss(pred_val, targets)
                
                del targets
                del pred_val

        return val_loss/(i+1)

    def _train(self, num_epoch: int = 3, early_stopping: int = 2, alpha: float = 0.9):
        """ Method to train the model
        Args:
            num_epoch: number of epochs you want to train for
            alpha: exponential smoothing factor
        """
        patience = 0
        for epoch in range(num_epoch):
            self._train_epoch()
            # Validate at the end of an epoch
            val_loss = self._eval_model()
            print(f"### End of the epoch : Validation loss for {epoch} is {val_loss.item()}")
            self.scheduler.step(val_loss)
            val_loss_s = val_loss.item()
            # Exponential smoothing for validation
            self.valid_perf.append(val_loss_s if epoch == 0 else alpha*val_loss_s + (1-alpha)*self.valid_perf[-1])
            
            if epoch != 0 and min(min_loss, val_loss_s) == min_loss:
                patience +=1
                if patience >= early_stopping:
                    break
            else:
                patience = 0

            min_loss = val_loss_s if epoch == 0 else min(min_loss, val_loss_s)

            del val_loss        

    def plot_data(self):
        p_data = (self.learning_curve, self.valid_perf, self.learning_rates)
        plot_names = ['Learning curve','Validation loss for every 400 batches', 'Learning rates']

        for i in range(3):
            self.summaries_axes[i].plot(range(len(p_data[i])), p_data[i])
            self.summaries_axes[i].set_ylabel('Loss')
            self.summaries_axes[i].set_xlabel('Epochs')
            self.summaries_axes[i].set_xlim((0, len(p_data[i])))
            self.summaries_axes[i].set_title(plot_names[i])

        plt.savefig('Loss_plot.png', dpi=800)
        plt.show()


        