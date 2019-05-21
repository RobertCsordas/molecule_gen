#!/usr/bin/env python3

import torch
from chembl import Chembl
import argparse
from graphgen import GraphGen
from tqdm import tqdm
from saver import Saver
from visdom_helper import Plot2D
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-wd", type=float, default=0)
parser.add_argument("-save_interval", type=int, default=5000)
parser.add_argument("-optimizer", default="adam")
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-save_dir", type=str)
parser.add_argument("-gpu", type=str, default="")
opt = parser.parse_args()


class Experiment:
    def __init__(self, opt):
        assert opt.save_dir is not None, "No directory given for saving the model."

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

        self.opt = opt
        self.device = torch.device("cpu" if opt.gpu=="" or not torch.cuda.is_available() else "cuda")

        self.train_set = Chembl("train")
        self.valid_set = Chembl("valid")
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=opt.batch_size, shuffle=True,
                                                        collate_fn=Chembl.collate, num_workers=1)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=256, shuffle=False,
                                                        collate_fn=Chembl.collate, num_workers=1)

        self.model = GraphGen(self.train_set.n_node_types(), self.train_set.n_edge_types(), 128)
        self.model = self.model.to(self.device)

        if opt.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        elif opt.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True)
        else:
            assert False, "Invalid optimizer: %s" % opt.optimizer

        self.loss_plot = Plot2D("loss", 10, xlabel="iter", ylabel="loss")
        self.valid_loss_plot = Plot2D("Validation loss", 1, xlabel="iter", ylabel="loss")

        self.iteration = 0
        self.epoch = 0
        self.saver = Saver(self, opt.save_dir)
        self.saver.load()

    def _move_to_device(self, d):
        if isinstance(d, tuple):
            return tuple(self._move_to_device(list(d)))
        elif isinstance(d, list):
            return [self._move_to_device(v) for v in d]
        elif isinstance(d, dict):
            return {k: self._move_to_device(v) for k, v in d}
        elif torch.is_tensor(d):
            return d.to(self.device)
        else:
            return d

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "loss_plot": self.loss_plot.state_dict(),
            "iteration": self.iteration,
            "epoch": self.epoch
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.loss_plot.load_state_dict(state_dict["loss_plot"])
        self.iteration = state_dict["iteration"]
        self.epuch = state_dict["epoch"]

    def test(self):
        self.model.eval()

        loss_sum = 0
        cnt = 0
        with torch.no_grad():
            for d in tqdm(self.valid_loader):
                d = self._move_to_device(d)
                _, loss = self.model(d[0])

                cnt += d[0][0].shape[0]
                loss_sum += loss.item() * d[0][0].shape[0]

        loss = loss_sum / cnt
        self.valid_loss_plot.add_point(self.iteration, loss)

    def train(self):
        self.model.train()

        while True:
            print("Epoch %d" % self.epoch)

            for d in tqdm(self.train_loader):
                d = self._move_to_device(d)
                # print(d)
                g, loss = self.model(d[0])
                assert torch.isfinite(loss), "Loss is %s" % loss.item()

                self.loss_plot.add_point(self.iteration, loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()

                self.iteration += 1
                # if self.iteration % self.opt.save_interval==0:
                #     self.saver.save(self.iteration)

            self.test()
            self.saver.save(self.iteration)
            self.epoch += 1

e = Experiment(opt)
e.train()
# e.test()