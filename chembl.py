import urllib.request
import os
import gzip
from rdkit import Chem
import torch
from tqdm import tqdm
import numpy as np
import math
import torch.utils.data

class Chembl(torch.utils.data.Dataset):
    URL = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_25_chemreps.txt.gz"
    CACHE_DIR = "./cache"
    DOWNLOAD_TO = "./cache/chembl/chembl_25_chemreps.txt.gz"
    PROCESSED_FILE = "./cache/chembl/chembl_25.pth"
    SPLITS = [0.5, 0.1, 0.4]
    SPLIT_NAMES = ["train", "valid", "test"]
    PAD_CHAR = 255

    dataset = None

    def _get_split(self, array, split):
        l = len(array)
        set_i = self.SPLIT_NAMES.index(split)
        start = 0 if set_i==0 else int(math.ceil(l*self.SPLITS[set_i-1]))
        end = int(l*self.SPLITS[set_i])
        return array[start:end]

    def __init__(self, split="train", max_atoms=20, random_order=False):
        super().__init__()
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
                    "heavy_atom_count": []
                }

                for line in tqdm(lines):
                    line = line.split("\t")
                    if len(line)>=2:
                        smiles = line[1]

                        m = Chem.MolFromSmiles(smiles)
                        if m is None:
                            print("WARNING: Invalid SMILES string:", smiles)
                            continue

                        for a in m.GetAtoms():
                            atom_types.add(a.GetSymbol())
                            for b in a.GetBonds():
                                bond_types.add(b.GetBondType())

                        canonical_smiles = Chem.MolToSmiles(m)
                        self.dataset["smiles"].append(canonical_smiles)
                        self.dataset["heavy_atom_count"].append(m.GetNumHeavyAtoms())

                # Do a random permutation that is constant amoung runs
                indices = np.random.RandomState(0xB0C1FA52).choice(len(self.dataset["smiles"]),
                                                                   len(self.dataset["smiles"]), replace=False)
                Chembl.dataset = {
                    "smiles": [self.dataset["smiles"][i] for i in indices],
                    "heavy_atom_count": [self.dataset["heavy_atom_count"][i] for i in indices],
                    "bond_types": list(sorted(bond_types)),
                    "atom_types": list(sorted(atom_types))
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

        used_smiles = [s for i, s in enumerate(self.dataset["smiles"]) if self.dataset["heavy_atom_count"][i] <= max_atoms]
        self.used_set = used_smiles

        print("%d atoms match the count limit" % len(self.used_set))
        self.used_set = self._get_split(self.used_set, split)
        print("%d atoms used for %s set." % (len(self.used_set), split))

        self.random_order = random_order
        self.seed = None

    def __len__(self):
        return len(self.used_set)

    def __getitem__(self, item):
        if self.seed is None:
            self.seed = np.random.RandomState()

        schema = []

        m = Chem.MolFromSmiles(self.used_set[item])

        atom_to_index = {}

        atoms = m.GetAtoms()
        iterator = self.seed.choice(len(atoms), len(atoms), replace=False).tolist() if self.random_order else\
                   range(len(atoms))


        next_atom_index = 0
        for ai in iterator:
            atom = atoms[ai]
            atom_to_index[atom.GetIdx()] = next_atom_index
            type = self.dataset["atom_type_to_id"][atom.GetSymbol()]
            bonds = atom.GetBonds()
            edges = []

            iter2 = self.seed.choice(len(bonds), len(bonds), replace=False).tolist() if self.random_order else\
                   range(len(bonds))

            for bi in iter2:
                bond = bonds[bi]
                other_atom = bond.GetBeginAtomIdx()
                if other_atom == atom.GetIdx():
                    other_atom = bond.GetEndAtomIdx()

                other_atom_local_index = atom_to_index.get(other_atom)
                if other_atom_local_index is not None:
                    edges.append((other_atom_local_index, self.dataset["bond_type_to_id"][bond.GetBondType()]))

            schema.append(type)
            schema.append(edges)

            next_atom_index += 1

        return schema


    @staticmethod
    def onehot(len, index):
        a = [0]*len
        a[index] = 1
        return a

    @classmethod
    def batchify(cls, seq_list):
        # Output data[0] is a list of operations that should be done. Even (0,2,..) elements of the list are the
        # nodes that should be added, the odd ones (1,3...) are the lists of nodes to be added. Each element of
        # this list is a touple of (parent node ID, node type). The elements of this tuple are no. of batches long
        # lists.
        #
        # Output data[1] is a tensor with one-hot columns indicating to which batch a the given node belongs to.

        res = []
        node_owner_mask = []

        n_seq = len(seq_list)
        longest = max(len(s) for s in seq_list)

        new_node_count = 0
        node_id_to_new = {}

        for i in range(longest):
            if i % 2 == 0:
                # It's a node.

                all_nodes = []
                for si, s in enumerate(seq_list):
                    if i >= len(s):
                        all_nodes.append(cls.PAD_CHAR)
                    else:
                        node_id_to_new[(si, i//2)] = new_node_count
                        new_node_count += 1
                        all_nodes.append(s[i])
                        node_owner_mask.append(cls.onehot(n_seq, si))

                res.append(all_nodes)
            else:
                # It's an edge

                this_edge_set = []

                all_edges = []
                all_edge_types = []

                max_edges = max(len(s[i]) for s in seq_list if i<len(s))

                for ei in range(max_edges):
                    for si, s in enumerate(seq_list):
                        if i >= len(s) or ei >= len(s[i]):
                            all_edge_types.append(cls.PAD_CHAR)
                            all_edges.append(0)
                        else:
                            all_edges.append(node_id_to_new[(si, s[i][ei][0])])
                            all_edge_types.append(s[i][ei][1])

                    this_edge_set.append((all_edges, all_edge_types))

                res.append(this_edge_set)

        return res, node_owner_mask

    @staticmethod
    def to_tensor(batched_seq, owner_mask):
        res = []
        for si, s in enumerate(batched_seq):
            if si % 2 == 0:
                res.append(torch.tensor(s, dtype=torch.uint8))
            else:
                res.append([(torch.tensor(a[0], dtype=torch.uint8), torch.tensor(a[1], dtype=torch.uint8)) for a in s])

        return res, torch.tensor(owner_mask, dtype=torch.uint8)

    @classmethod
    def collate(cls, seq):
        return cls.to_tensor(*cls.batchify(seq))

if __name__=="__main__":
    dataset = Chembl()

    print(dataset[1])
    print("-------------------------------------------------------------------------")
    print(dataset[2])
    print("-------------------------------------------------------------------------")
    print(dataset.batchify([dataset[1], dataset[2]]))
    print("-------------------------------------------------------------------------")
    print(dataset.to_tensor(*dataset.batchify([dataset[1], dataset[2]])))



