# Copyright 2019 Robert Csordas. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import os
import torch


class Saver:
    def __init__(self, module, save_dir):
        self.module = module
        self.save_dir = save_dir

    def save(self, suffix):
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(self.module.state_dict(), os.path.join(self.save_dir, "model-%s.pth" % suffix))

    @staticmethod
    def get_checkpoint_index_list(dir):
        if not os.path.isdir(dir):
            return []

        if os.path.isfile(os.path.join(dir, "model-best.pth")):
            return ["best"]

        return list(reversed(sorted(
            [int(fn.split(".")[0].split("-")[-1]) for fn in os.listdir(dir) if fn.split(".")[-1] == "pth"])))

    def _name_from_iter(self, iter):
        return os.path.join(self.save_dir, "model-%s.pth" % iter)

    def newest_checkpoint(self):
        f = self.get_checkpoint_index_list(self.save_dir)
        return self._name_from_iter(f[0]) if f else None

    def load(self, fname=None):
        if fname is None:
            fname = self.newest_checkpoint()
            if not fname:
                return
        elif isinstance(fname, int):
            fname = self._name_from_iter(fname)
        elif not isinstance(fname, str):
            assert False, "Invalid fname"

        print("Loading %s" % fname)
        self.module.load_state_dict(torch.load(fname))
