#!/usr/bin/env python3

import torch
from chembl import Chembl
import argparse
from graphgen import GraphGen
from tqdm import tqdm
from saver import Saver

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-wd", type=float, default=0)
parser.add_argument("-save_interval", type=int, default=5000)
parser.add_argument("-optimizer", default="adam")
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-save_dir", type=str)
opt = parser.parse_args()


class Experiment:
    def __init__(self, opt):
        assert opt.save_dir is not None, "No directory given for saving the model."

        self.opt = opt

        self.train_set = Chembl("train")
        # self.valid_set = Chembl("valid")
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=opt.batch_size, shuffle=False,
                                                        collate_fn=Chembl.collate, num_workers=1)
        # self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=256, shuffle=False,
        #                                                 collate_fn=Chembl.collate, num_workers=1)

        self.model = GraphGen(self.train_set.n_node_types(), self.train_set.n_edge_types(), 128)

        self.model = self.model.cuda()

        if opt.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        elif opt.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True)
        else:
            assert False, "Invalid optimizer: %s" % opt.optimizer

        self.iteration = 0
        self.saver = Saver(self, opt.save_dir)
        self.saver.load()

    @classmethod
    def _all_to_cuda(cls, d):
        if isinstance(d, tuple):
            return tuple(cls._all_to_cuda(list(d)))
        elif isinstance(d, list):
            return [cls._all_to_cuda(v) for v in d]
        elif isinstance(d, dict):
            return {k: cls._all_to_cuda(v) for k, v in d}
        elif torch.is_tensor(d):
            return d.cuda()
        else:
            return d

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "iteration": self.iteration
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.iteration = state_dict["iteration"]

    def train(self):
        while True:
            for d in tqdm(self.train_loader):
                d = self._all_to_cuda(d)
                # print(d)
                g, loss = self.model(d[0])
                print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()

                self.iteration += 1
                if self.iteration % self.opt.save_interval==0:
                    self.saver.save(self.iteration)

e = Experiment(opt)
e.train()