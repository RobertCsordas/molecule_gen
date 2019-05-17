#!/usr/bin/env python3

import torch
from chembl import Chembl
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-batch_size", type=int, default=8)
opt = parser.parse_args()


class Experiment:
    def __init__(self):
        self.train_set = Chembl("train")
        self.valid_set = Chembl("valid")
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=opt.batch_size, shuffle=True,
                                                        collate_fn=Chembl.collate, num_workers=1)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=256, shuffle=False,
                                                        collate_fn=Chembl.collate, num_workers=1)

    def train(self):
        for d in self.train_loader:
            print(d)

e = Experiment()
e.train()