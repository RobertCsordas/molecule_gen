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

    def __init__(self, split="train", max_atoms=20, random_order=False, verify_in_process=False):
        super().__init__()
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

                        for a in m.GetAtoms():
                            atom_types.add(a.GetSymbol())
                            for b in a.GetBonds():
                                bond_types.add(b.GetBondType())

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

        used_smiles = [s for i, s in enumerate(self.dataset["smiles"]) if self.dataset["heavy_atom_count"][i] <= max_atoms]
        self.used_set = used_smiles

        self.max_bonds = max(b for i, b in enumerate(self.dataset["bond_count"])
                             if self.dataset["heavy_atom_count"][i] <= max_atoms)

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

            # Termination node for the edges
            edges.append((0, self.PAD_CHAR))
            schema.append(type)
            schema.append(edges)

            next_atom_index += 1

        # Node termination
        schema.append(self.PAD_CHAR)
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
                    if i >= len(s) or s[i]==cls.PAD_CHAR:
                        all_nodes.append(cls.PAD_CHAR)
                    else:
                        node_id_to_new[(si, i // 2)] = new_node_count
                        new_node_count += 1
                        all_nodes.append(s[i])
                        node_owner_mask.append(cls.onehot(n_seq, si))

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

        return res, node_owner_mask

    @staticmethod
    def to_tensor(batched_seq, owner_mask):
        res = []
        for si, s in enumerate(batched_seq):
            if si % 2 == 0:
                res.append(torch.tensor(s, dtype=torch.uint8))
            else:
                res.append([(torch.tensor(a[0], dtype=torch.int16), torch.tensor(a[1], dtype=torch.uint8)) for a in s])

        return res, torch.tensor(owner_mask, dtype=torch.uint8)

    @classmethod
    def collate(cls, seq):
        return cls.to_tensor(*cls.batchify(seq))

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
        molecules = [Chem.RWMol() for i in range(graph.batch_size)]
        atom_counts = [0 for i in range(graph.batch_size)]

        atoms = graph.node_types.cpu().numpy().tolist()
        owners = graph.owner_masks.max(0)[1].cpu().numpy().tolist()

        atom_maps = {}

        for i, a in enumerate(atoms):
            molecules[owners[i]].AddAtom(Chem.Atom(self.dataset["id_to_atom_type"][a]))
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
                continue

            if si==di:
                molecules[batch] = None
                continue

            added_edges[batch].add((si,di))
            molecules[batch].AddBond(si, di, self.dataset["id_to_bond_type"][etype])

        return molecules

    def _verify(self, v, graph):
        molecules = self.graph_to_molecules(graph)

        for m in molecules:
            v["n_total"] += 1

            if m is None:
                continue

            try:
                s = Chem.SanitizeMol(m)
                canonical_smiles = Chem.MolToSmiles(m)
            except:
                continue

            v["n_ok"] += 1
            v["n_new"] += int(canonical_smiles not in self.used_set)
            if canonical_smiles not in v["known"]:
                v["unique"] += 1
                v["known"].add(canonical_smiles)

    def _get_verification_results(self, v):
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




