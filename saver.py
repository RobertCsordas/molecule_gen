import os
import torch


class Saver:
    def __init__(self, module, save_dir):
        self.module = module
        self.save_dir = save_dir

    def save(self, iteration):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(self.save_dir, "model-%d.pth" % iteration))

    @staticmethod
    def get_checkpoint_index_list(dir):
         if not os.path.isdir(dir):
             return []

         return list(reversed(sorted(
            [int(fn.split(".")[0].split("-")[-1]) for fn in os.listdir(dir) if fn.split(".")[-1] == "pth"])))

    def newest_checkpoint(self):
        f = self.get_checkpoint_index_list(self.save_dir)
        return os.path.join(self.save_dir, "model-%d.pth" % f[0]) if f else None

    def load(self):
        fname = self.newest_checkpoint()
        if not fname:
            return

        print("Loading %s" % fname)
        self.module.load_state_dict(torch.load(fname))
