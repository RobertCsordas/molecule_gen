import urllib.request
import os
import gzip
from rdkit import Chem
import torch
from tqdm import tqdm
import numpy as np
import math

class Chembl:
    URL = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_25_chemreps.txt.gz"
    CACHE_DIR = "./cache"
    DOWNLOAD_TO = "./cache/chembl/chembl_25_chemreps.txt.gz"
    PROCESSED_FILE = "./cache/chembl/chembl_25.pth"
    SPLITS = [0.5, 0.1, 0.4]
    SPLIT_NAMES = ["train", "valid", "test"]
    PAD_CHAR = -1

    dataset = None

    def _get_split(self, array, split):
        l = len(array)
        set_i = self.SPLIT_NAMES.index(split)
        start = 0 if set_i==0 else int(math.ceil(l*self.SPLITS[set_i-1]))
        end = int(l*self.SPLITS[set_i])
        return array[start:end]

    def __init__(self, max_atoms=20, split="train", random_order=False):
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

    @classmethod
    def batchify(cls, seq_list):
        res = []
        node_owner_mask = []

        longest = max(len(s) for s in seq_list)

        for i in range(longest):
            if i % 2 == 0:
                # It's a node.

                all_nodes = []
                for si, s in enumerate(seq_list):
                    if i >= len(s):
                        all_nodes.append(cls.PAD_CHAR)
                    else:
                        all_nodes.append(s[i])



if __name__=="__main__":
    dataset = Chembl()

    print(dataset[1])
    print(dataset[2])

