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

import urllib.request
import os
import gzip
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
import torch
from tqdm import tqdm
import numpy as np
import math
import torch.utils.data
import torch.multiprocessing as mp

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


class Chembl(torch.utils.data.Dataset):
    URL = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_25_chemreps.txt.gz"
    CACHE_DIR = "./cache"
    DOWNLOAD_TO = "./cache/chembl/chembl_25_chemreps.txt.gz"
    PROCESSED_FILE = "./cache/chembl/chembl_25.pth"
    # SPLITS = [0.000005*2, 0.01, 0.004]
    SPLITS = [0.5, 0.1, 0.4]
    SPLIT_NAMES = ["train", "valid", "test"]
    PAD_CHAR = 255

    dataset = None

    def _get_split(self, array, split):
        l = len(array)
        set_i = self.SPLIT_NAMES.index(split)
        start = 0 if set_i==0 else int(math.ceil(l*self.SPLITS[set_i-1]))
        end = start + int(l*self.SPLITS[set_i])
        return array[start:end]

    @staticmethod
    def _atom_type_to_str(atom):
        nh = atom.GetNumExplicitHs()
        if nh!=0:
            return atom.GetSymbol()+":H"+str(nh)
        else:
            return atom.GetSymbol()

    @staticmethod
    def _str_to_atom(str):
        d = str.split(":H")
        a = Chem.Atom(d[0])
        if len(d)>1:
            a.SetNumExplicitHs(int(d[1]))
        return a
    
    def __init__(self, split="train", max_atoms=20, random_order=False, verify_in_process=False, kekulize=False):
        super().__init__()
        self.kekulize = kekulize
        self.verify_in_process = verify_in_process
        assert split in self.SPLIT_NAMES, "Invalid set: %s" % split

        if Chembl.dataset is None:
            if not os.path.isfile(self.PROCESSED_FILE):
                if not os.path.isfile(self.DOWNLOAD_TO):
                    os.makedirs(os.path.dirname(self.DOWNLOAD_TO), exist_ok=True)
                    print("Downloading...")
                    urllib.request.urlretrieve(self.URL, self.DOWNLOAD_TO)

                print("Reading dataset...")
                with gzip.open(self.DOWNLOAD_TO, mode="r") as f:
                    ftext = f.read().decode()

                print("Read. Processing SMILES strings")
                lines = ftext.split("\n")[1:]

                atom_types = set()
                bond_types = set()

                Chembl.dataset = {
                    "smiles": [],
                    "heavy_atom_count": [],
                    "bond_count": []
                }

                for line in tqdm(lines):
                    line = line.split("\t")
                    if len(line)>=2:
                        smiles = line[1]

                        m = Chem.MolFromSmiles(smiles)
                        if m is None:
                            print("WARNING: Invalid SMILES string:", smiles)
                            continue

                        Chem.SanitizeMol(m)

                        failed = False
                        for a in m.GetAtoms():
                            if a.GetNumExplicitHs()!=0 or a.GetFormalCharge()!=0:
                                failed = True
                            # atom_types.add(self._atom_type_to_str(a))
                            atom_types.add(a.GetSymbol())
                            for b in a.GetBonds():
                                bond_types.add(b.GetBondType())

                        if failed:
                            continue
                        canonical_smiles = Chem.MolToSmiles(m)
                        self.dataset["smiles"].append(canonical_smiles)
                        self.dataset["heavy_atom_count"].append(m.GetNumHeavyAtoms())
                        self.dataset["bond_count"].append(m.GetNumBonds())

                # Do a random permutation that is constant amoung runs
                indices = np.random.RandomState(0xB0C1FA52).choice(len(self.dataset["smiles"]),
                                                                   len(self.dataset["smiles"]), replace=False)
                Chembl.dataset = {
                    "smiles": [self.dataset["smiles"][i] for i in indices],
                    "heavy_atom_count": [self.dataset["heavy_atom_count"][i] for i in indices],
                    "bond_count": [self.dataset["bond_count"][i] for i in indices],
                    "bond_types": list(sorted(bond_types)),
                    "atom_types": list(sorted(atom_types)),
                }

                print("Done. Read %d" % len(self.dataset["smiles"]))
                print("Saving")
                torch.save(self.dataset, self.PROCESSED_FILE)
                print("Done.")
            else:
                Chembl.dataset = torch.load(self.PROCESSED_FILE)

        self.dataset["bond_type_to_id"] = {v: k for k, v in enumerate(self.dataset["bond_types"])}
        self.dataset["id_to_bond_type"] = {k: v for k, v in enumerate(self.dataset["bond_types"])}
        self.dataset["atom_type_to_id"] = {v: k for k, v in enumerate(self.dataset["atom_types"])}
        self.dataset["id_to_atom_type"] = {k: v for k, v in enumerate(self.dataset["atom_types"])}

        # Limit the number of atoms.
        used_smiles = [s for i, s in enumerate(self.dataset["smiles"]) if self.dataset["heavy_atom_count"][i] <= max_atoms]
        self.used_set = used_smiles

        self.max_bonds = max(b for i, b in enumerate(self.dataset["bond_count"])
                             if self.dataset["heavy_atom_count"][i] <= max_atoms)

        print("%d atoms match the count limit" % len(self.used_set))
        self.used_set = self._get_split(self.used_set, split)
        print("%d atoms used for %s set." % (len(self.used_set), split))
        print("Atom types:", self.dataset["atom_types"])
        print("Bond types:", self.dataset["bond_types"])

        self.random_order = random_order
        self.seed = None

    def __len__(self):
        return len(self.used_set)

    def __getitem__(self, item):
        # Load a molecule from SMILES and generate a sequence of graph constructing operations from it. The sequence
        # must end with a padding character. That ensures that the generation process will terminate.
        if self.seed is None:
            self.seed = np.random.RandomState()

        schema = []

        m = Chem.MolFromSmiles(self.used_set[item])
        Chem.SanitizeMol(m)
        if self.kekulize:
            Chem.Kekulize(m)

        atoms = m.GetAtoms()
        iterator = self.seed.choice(len(atoms), len(atoms), replace=False).tolist() if self.random_order else\
                   range(len(atoms))

        atom_to_new_id = {}
        next_id = 0
        for ai in iterator:
            atom = atoms[ai]

            # symbol = self._atom_type_to_str(atom)
            symbol = atom.GetSymbol()
            type = self.dataset["atom_type_to_id"][symbol]
            bonds = atom.GetBonds()
            edges = []

            atom_to_new_id[ai] = next_id

            iter2 = self.seed.choice(len(bonds), len(bonds), replace=False).tolist() if self.random_order else\
                   range(len(bonds))

            # Add all the bonds to all the atoms that already exists
            for bi in iter2:
                bond = bonds[bi]
                other_atom = bond.GetBeginAtomIdx()
                if other_atom == atom.GetIdx():
                    other_atom = bond.GetEndAtomIdx()

                other_atom_id = atom_to_new_id.get(other_atom)
                if other_atom_id is not None:
                    edges.append((other_atom_id, self.dataset["bond_type_to_id"][bond.GetBondType()]))

            # Termination node for the edges
            edges.append((0, self.PAD_CHAR))
            schema.append(type)
            schema.append(edges)

            next_id += 1

        # Node termination
        schema.append(self.PAD_CHAR)
        return schema

    @classmethod
    def batchify(cls, seq_list):
        # Output is a list of operations that should be done. Even (0,2,..) elements of the list are the
        # nodes that should be added, the odd ones (1,3...) are the lists of edges to be added. Each element of
        # this list is a tuple of (parent node ID, node type). The elements of this tuple are no. of batches long
        # lists.

        res = []

        longest = max(len(s) for s in seq_list)

        new_node_count = 0
        node_id_to_new = {}

        for i in range(longest):
            if i % 2 == 0:
                # It's a node.
                all_nodes = []
                for si, s in enumerate(seq_list):
                    if i >= len(s) or s[i] == cls.PAD_CHAR:
                        all_nodes.append(cls.PAD_CHAR)
                    else:
                        node_id_to_new[(si, i // 2)] = new_node_count
                        new_node_count += 1
                        all_nodes.append(s[i])

                res.append(all_nodes)
            else:
                # It's an edge
                this_edge_set = []
                max_edges = max(len(s[i]) for s in seq_list if i<len(s))

                for ei in range(max_edges):
                    all_edges = []
                    all_edge_types = []

                    for si, s in enumerate(seq_list):
                        if i >= len(s) or ei >= len(s[i]):
                            all_edge_types.append(cls.PAD_CHAR)
                            all_edges.append(0)
                        else:
                            all_edges.append(node_id_to_new[(si, s[i][ei][0])])
                            all_edge_types.append(s[i][ei][1])

                    this_edge_set.append((all_edges, all_edge_types))

                res.append(this_edge_set)

        return res

    @staticmethod
    def to_tensor(batched_seq):
        res = []
        for si, s in enumerate(batched_seq):
            if si % 2 == 0:
                res.append(torch.tensor(s, dtype=torch.uint8))
            else:
                res.append([(torch.tensor(a[0], dtype=torch.int16), torch.tensor(a[1], dtype=torch.uint8)) for a in s])

        return res

    @classmethod
    def collate(cls, seq):
        return cls.to_tensor(cls.batchify(seq))

    def n_node_types(self):
        return len(self.dataset["atom_types"])

    def n_edge_types(self):
        return len(self.dataset["bond_types"])

    @staticmethod
    def verify_links(batched):
        # Slow, for debug purposes only
        atom_owner = {}
        curr_id = 0

        for i, c in enumerate(batched):
            if i % 2 == 0:
                # node
                for bi, a in enumerate(c):
                    if a == 255:
                        continue
                    atom_owner[curr_id] = bi
                    curr_id += 1
            else:
                # edge
                for e in c:
                    types = e[1]
                    links = e[0]

                    for bi, l in enumerate(links):
                        if types[bi] != 255:
                            assert l in atom_owner, "Link to invalid atom: %l" % l
                            assert atom_owner[l] == bi, "Atom %d owner: %d != link %d" % (l, atom_owner[l], bi)

    def graph_to_molecules(self, graph):
        # Converts a graph output from the network to an RDKit molecule
        molecules = [Chem.RWMol() for i in range(graph.batch_size)]
        atom_counts = [0 for i in range(graph.batch_size)]

        atoms = graph.node_types.cpu().numpy().tolist()
        owners_g = graph.owner_masks.max(0)[1]
        owners = owners_g.cpu().numpy().tolist()

        atom_maps = {}
        for i, a in enumerate(atoms):
            molecules[owners[i]].AddAtom(Chem.Atom(self.dataset["id_to_atom_type"][a]))
            # molecules[owners[i]].AddAtom(self._str_to_atom(self.dataset["id_to_atom_type"][a]))
            atom_maps[i] = (owners[i], atom_counts[owners[i]])
            atom_counts[owners[i]] += 1

        edge_src = graph.edge_source.cpu().numpy().tolist()
        edge_dest = graph.edge_dest.cpu().numpy().tolist()
        edge_type = graph.edge_types.cpu().numpy().tolist()

        # Edges are directed. So each edge is present 2 times in edge_*. Filter them out
        added_edges = [set() for i in range(graph.batch_size)]
        for i, etype in enumerate(edge_type):
            batch, si = atom_maps[edge_src[i]]
            batch2, di = atom_maps[edge_dest[i]]

            assert batch==batch2, "Cross batch edge?! %d, %d" % (batch, batch2)

            if molecules[batch] is None:
                continue

            if (si, di) in added_edges[batch] or (di, si) in added_edges[batch]:
                molecules[batch] = None
                continue

            if si==di:
                molecules[batch] = None
                continue

            added_edges[batch].add((si,di))
            molecules[batch].AddBond(si, di, self.dataset["id_to_bond_type"][etype])

        # Sanitize the molecule
        final = []
        for m in molecules:
            try:
                Chem.SanitizeMol(m)
                final.append(m)
            except:
                final.append(None)

        return final

    def all_graphs_ok(self, graph):
        molecules = self.graph_to_molecules(graph)
        for m in molecules:
            if m is None:
                print([m is not None for m in molecules])
                return False
        return True

    def _verify(self, v, graph):
        # Count valid, new and unique molecules.
        molecules = self.graph_to_molecules(graph)

        for m in molecules:
            v["n_total"] += 1

            if m is None:
                continue

            canonical_smiles = Chem.MolToSmiles(m)

            v["n_ok"] += 1
            v["n_new"] += int(canonical_smiles not in self.used_set)
            if canonical_smiles not in v["known"]:
                v["unique"] += 1
                v["known"].add(canonical_smiles)

    def _get_verification_results(self, v):
        # Compute the result of the verification process
        return {
           "ratio_ok": v["n_ok"] / v["n_total"],
           "ratio_not_in_training": (v["n_new"] / v["n_ok"] if v["n_ok"] > 0 else 0),
           "ratio_unique": (v["unique"] / v["n_ok"] if v["n_ok"] > 0 else 0)
        }

    def _start_verification(self):
        return dict(n_ok=0, n_total=0, n_new=0, unique=0, known=set())

    def start_verification(self):
        if not self.verify_in_process:
            return self._start_verification()

        # Verification is a slow process. So start background process to do it.
        def worker(queue):
            data = self._start_verification()
            while True:
                graph = queue.get()
                if graph is None:
                    queue.put(self._get_verification_results(data))
                    break

                self._verify(data, graph)

        queue = mp.Queue(1)
        p = mp.Process(target=worker, args=(queue,))
        p.start()

        return dict(process=p, queue=queue)

    def verify(self, v, graph):
        if not self.verify_in_process:
            return self._verify(v, graph)

        v["queue"].put(graph)

    def get_verification_results(self, v):
        if not self.verify_in_process:
            return self._get_verification_results(v)

        v["queue"].put(None)
        v["process"].join()

        res = v["queue"].get()

        v["process"] = None
        v["queue"] = None
        return res

    def get_max_bonds(self):
        return self.max_bonds

    def draw_molecules(self, graphs):
        mols = []

        if not isinstance(graphs, list):
            graphs = [graphs]

        for graph in graphs:
            for m in self.graph_to_molecules(graph):
                if m is None:
                    continue

                try:
                    Chem.SanitizeMol(m)
                except:
                    continue

                mols.append(m)

        if not mols:
            return None

        return Draw.MolsToGridImage(mols, molsPerRow=int(math.ceil(math.sqrt(len(mols)))))


if __name__=="__main__":
    dataset = Chembl()

    Chembl.verify_links(dataset.batchify([dataset[i] for i in range(64)])[0])

    print(dataset[1])
    print("-------------------------------------------------------------------------")
    print(dataset[2])
    print("-------------------------------------------------------------------------")
    print(dataset.batchify([dataset[1], dataset[2]]))
    Chembl.verify_links(dataset.batchify([dataset[1], dataset[2]])[0])
    print("-------------------------------------------------------------------------")
    print(dataset.to_tensor(*dataset.batchify([dataset[1], dataset[2]])))




