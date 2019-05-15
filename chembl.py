import urllib.request
import os
import gzip
from rdkit import Chem
import torch
from tqdm import tqdm
import numpy as np

class Chembl:
    URL = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_25_chemreps.txt.gz"
    CACHE_DIR = "./cache"
    DOWNLOAD_TO = "./cache/chembl/chembl_25_chemreps.txt.gz"
    PROCESSED_FILE = "./cache/chembl/chembl_25.pth"

    def __init__(self, max_atoms=20):
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

            self.dataset = {
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
                    canonical_smiles = Chem.MolToSmiles(m)
                    self.dataset["smiles"].append(canonical_smiles)
                    self.dataset["heavy_atom_count"].append(m.GetNumHeavyAtoms())

            # Do a random permutation that is constant amoung runs
            indices = np.random.RandomState(0xB0C1FA52).choice(len(self.dataset["smiles"]),
                                                               len(self.dataset["smiles"]), replace=False)
            self.dataset = {
                "smiles": [self.dataset["smiles"][i] for i in indices],
                "heavy_atom_count": [self.dataset["heavy_atom_count"][i] for i in indices],
            }

            print("Done. Read %d" % len(self.dataset["smiles"]))
            print("Saving")
            torch.save(self.dataset, self.PROCESSED_FILE)
            print("Done.")
        else:
            self.dataset = torch.load(self.PROCESSED_FILE)

        used_smiles = [s for i, s in enumerate(self.dataset["smiles"]) if self.dataset["heavy_atom_count"][i] <= max_atoms]
        self.dataset = used_smiles

        print("Using %d atoms" % len(self.dataset))



if __name__=="__main__":
    dataset = Chembl()