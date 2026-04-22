from copy import deepcopy

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
import pickle
from sklearn.metrics import r2_score

import sys
sys.path.append('../model')
import builder

import os
from pathlib import Path

class trainer:
    def __init__(self, configs, model=None, device=None, criterion=None, optimizer=None, scheduler=None, loaders=None, results_path=None):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loaders = loaders
        self.best = 1e10
        self.mse = 0
        self.nmse = 0
        self.best_params = None
        self.training_data = {}
        self.parameter_name = configs.parameter_name
        self.parameter_path = Path(os.getcwd()) / f"model/parameter/{self.parameter_name}"
        self.results_path = Path(results_path) / "full_result" if results_path else self.parameter_path
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.num_epochs = configs.num_epochs
        self.batch_size = configs.batch_size

    def train(self):
        self.model.train()
        self.training_data["train_loss"] = []
        self.training_data["val_error"] = []

        TARGET_R2 = 0.95
        for epoch in range(self.num_epochs):
            loss_sum = 0
            for batch in self.loaders["train"]:
                self.optimizer.zero_grad()
                out = self.model(batch)
                out = out.to(torch.float32)
                loss = self.criterion(out, batch.y.to(self.device).to(torch.float32))
                loss = loss.to(torch.float32)
                loss.backward()
                self.optimizer.step()
                loss_sum += (
                    loss.item() * len(batch.y) / len(self.loaders["train"].dataset)
                )
            self.scheduler.step()
            print(
                f"[{epoch + 1} / {self.num_epochs}],sqrtloss={loss_sum**0.5}",
                flush=True,
            )
            self.training_data["train_loss"].append(loss_sum**0.5)
            mse, nmse, r_squared = self.valid()
            self.save_best(mse, nmse, r_squared)
            self.training_data["val_error"].append(mse)
            if (epoch + 1) % 10 == 0:
                torch.save(self.best_params, str(self.results_path / "model.pth"))
                print(f"[Checkpoint saved at epoch {epoch + 1}, best R²={self.r_squared:.4f}]", flush=True)
            if r_squared >= TARGET_R2:
                print(f"[R²={r_squared:.4f} >= {TARGET_R2} at epoch {epoch + 1} — saving checkpoint, continuing...]", flush=True)
                torch.save(self.best_params, str(self.results_path / "model.pth"))
        torch.save(self.best_params, str(self.results_path / "model.pth"))

    def save_best(self, mse, nmse=0, r_squared=0):
        if mse < self.best:
            self.best = mse
            self.best_params = deepcopy(self.model.state_dict())
            self.nmse = nmse
            self.r_squared = r_squared

    def valid(self):
        return self.calculate_metrics(self.loaders["valid"])
    

    def calculate_metrics(self, data):
        self.model.eval()
        y = []
        pred = []
        for batch in data:
            out = self.model(batch)
            pred.extend(out.detach().cpu().numpy())
            y.extend(batch.y.to(self.device).detach().cpu().numpy())
            
        mse = np.mean((np.array(y) - np.array(pred)) ** 2)
        variance = np.mean((np.array(y) - np.mean(y)) ** 2)
        nmse = mse / variance
        r_squared = r2_score(y, pred)
        print(f"\t mse:{mse} nmse:{nmse} r_squared:{r_squared}", flush=True)

        return mse, nmse, r_squared
    

    def test(self):
        self.training_data["test_pred"] = np.array([])
        self.training_data["test_y"] = np.array([])
        self.test_error = 0
        # print(len(self.loaders["test"].dataset))
        if len(self.loaders["test"].dataset) > 1:
            self.model.load_state_dict(torch.load(str(self.results_path / "model.pth"), weights_only=True))
            
            self.model.eval()
            test_error = 0
            for batch in self.loaders["test"]:
                out = self.model(batch)
                self.training_data["test_pred"] = np.concatenate(
                    (self.training_data["test_pred"], np.atleast_1d(out.cpu().detach().numpy()))
                )
                self.training_data["test_y"] = np.concatenate(
                    (self.training_data["test_y"], np.atleast_1d(batch.y.cpu().detach().numpy()))
                )
                test_error += ((out - batch.y.to(self.device)) ** 2).sum().item()
            test_error = test_error / len(self.loaders["test"].dataset)
            self.test_error = test_error
    

    def saveall(self):
        mydict = {}
        mydict["train_loss"] = self.training_data["train_loss"]
        mydict["val_error"] = self.training_data["val_error"]
        mydict["test_pred"] = self.training_data["test_pred"]
        mydict["test_y"] = self.training_data["test_y"]
        mydict["test_error"] = self.test_error
        mydict["best"] = self.best

        torch.save(mydict, str(self.results_path / "all.pth"))

