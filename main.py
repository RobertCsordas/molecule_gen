#!/usr/bin/env python3

import torch
import gc
from chembl import Chembl
from graphgen import GraphGen
from tqdm import tqdm
from saver import Saver
from visdom_helper import Plot2D, Image
from argument_parser import ArgumentParser
import os

parser = ArgumentParser(description='Process some integers.')
parser.add_argument("-lr", type=float, default=3e-4)
parser.add_argument("-wd", type=float, default=0)
parser.add_argument("-optimizer", default="adam")
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-save_dir", type=str)
parser.add_argument("-force", type=bool, default=0, save=False)
parser.add_argument("-gpu", type=str, default="")
parser.add_argument("-lr_milestones", default="none", parser=ArgumentParser.int_list_parser)
parser.add_argument("-lr_gamma", default=0.3)
opt = parser.parse_and_sync()


class Experiment:
    def __init__(self, opt):
        assert opt.save_dir is not None, "No directory given for saving the model."

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

        self.opt = opt
        self.device = torch.device("cpu" if opt.gpu=="" or not torch.cuda.is_available() else "cuda")

        self.train_set = Chembl("train")
        self.valid_set = Chembl("valid")
        self.test_set = Chembl("test")
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=opt.batch_size, shuffle=True,
                                                        collate_fn=Chembl.collate, num_workers=1)
        self.valid_loader = torch.utils.data.DataLoader(self.valid_set, batch_size=256, shuffle=False,
                                                        collate_fn=Chembl.collate, num_workers=1)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=256, shuffle=False,
                                                        collate_fn=Chembl.collate, num_workers=1)

        self.model = GraphGen(self.train_set.n_node_types(), self.train_set.n_edge_types(), 128,
                              n_max_nodes=30, n_max_edges=2*self.train_set.get_max_bonds())
        self.model = self.model.to(self.device)

        if opt.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        elif opt.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt.lr, momentum=0.99, nesterov=True)
        else:
            assert False, "Invalid optimizer: %s" % opt.optimizer

        self.loss_plot = Plot2D("loss", 10, xlabel="iter", ylabel="loss")
        self.valid_loss_plot = Plot2D("Validation loss", 1, xlabel="iter", ylabel="loss")
        self.percent_valid = Plot2D("Valid molecules", 1, xlabel="iter", ylabel="%")
        self.mol_images = Image("Molecules")

        self.iteration = 0
        self.epoch = 0

        self.best_loss = float("inf")
        self.best_loss_iteration = 0
        self.best_loss_epoch = 0
        self.test_loss = None
        self.patience = 2


        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, opt.lr_milestones, opt.lr_gamma) \
                            if opt.lr_milestones else None
        self.saver = Saver(self, os.path.join(opt.save_dir, "save"))
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
            "valid_loss_plot": self.valid_loss_plot.state_dict(),
            "percent_valid": self.percent_valid.state_dict(),
            "iteration": self.iteration,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "best_loss_iteration": self.best_loss_iteration,
            "best_loss_epoch": self.best_loss_epoch,
            "test_loss": self.test_loss,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.valid_loss_plot.load_state_dict(state_dict["valid_loss_plot"])
        self.loss_plot.load_state_dict(state_dict["loss_plot"])
        self.percent_valid.load_state_dict(state_dict["percent_valid"])
        self.iteration = state_dict["iteration"]
        self.epoch = state_dict["epoch"]
        self.best_loss = state_dict["best_loss"]
        self.best_loss_iteration = state_dict["best_loss_iteration"]
        self.best_loss_epoch = state_dict["best_loss_epoch"]
        self.test_loss = state_dict["test_loss"]
        if self.lr_scheduler is not None:
            s = state_dict.get("lr_scheduler")
            if s is not None:
                self.lr_scheduler.load_state_dict(s)

    def test(self, loader=None):
        self.model.eval()

        loss_sum = 0
        cnt = 0
        with torch.no_grad():
            for d in tqdm(loader):
                d = self._move_to_device(d)
                _, loss = self.model(d[0])

                cnt += d[0][0].shape[0]
                loss_sum += loss.item() * d[0][0].shape[0]

        loss = loss_sum / cnt
        return loss

    def load_the_best(self):
        self.saver.load(self.best_loss_iteration)

    def do_final_test(self):
        self.load_the_best()
        self.test_loss = self.test(self.test_loader)
        print("----------------------------------------------")
        print("Training done.")
        print("    Validation loss:", self.best_loss)
        print("    Test loss:", self.test_loss)
        print("    Epoch:", self.best_loss_epoch)
        print("    Iteration:", self.best_loss_iteration)
        self.saver.save("best")

    def display_generated(self):
        self.model.eval()
        graphs = []
        with torch.no_grad():
            for i in range(1):
                g = self.model.generate(32, self.device)
                g = g.get_final_graph()
                graphs.append(g)

        self.model.train()
        img = self.train_set.draw_molecules(graphs)
        self.mol_images.draw(img)

    def train(self):
        running = True
        while running:
            print("Epoch %d" % self.epoch)

            self.model.train()
            for d in tqdm(self.train_loader):
                d = self._move_to_device(d)
                # print(d)
                g, loss = self.model(d[0])
                assert torch.isfinite(loss), "Loss is %s" % loss.item()

                # self.train_set.graph_to_molecules(g)

                self.loss_plot.add_point(self.iteration, loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                self.optimizer.step()

                self.iteration += 1
                if self.iteration % 30 == 0:
                    self.display_generated()

            # Do a validation step
            validation_loss = self.test(self.valid_loader)
            self.valid_loss_plot.add_point(self.iteration, validation_loss)

            g = self.generate(10000)
            self.percent_valid.add_point(self.iteration, g["ratio_ok"]*100)

            # Early stopping
            if validation_loss <= self.best_loss:
                self.best_loss = validation_loss
                self.best_loss_iteration = self.iteration
                self.best_loss_epoch = self.epoch
            elif (self.epoch - self.best_loss_epoch) > self.patience:
                running = False

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # Save the model
            self.saver.save(self.iteration)
            self.epoch += 1

    def train_done(self):
        return self.test_loss is not None

    def generate(self, n_test=100000):
        v = self.train_set.start_verification()

        bsize = 200
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(n_test//bsize)):
                res = self.model.generate(bsize, self.device)
                self.train_set.verify(v, res.get_final_graph())
                gc.collect()

        return self.train_set.get_verification_results(v)

e = Experiment(opt)
if (not e.train_done()) or opt.force:
    e.train()
    e.do_final_test()
# e.train()
print(e.generate())