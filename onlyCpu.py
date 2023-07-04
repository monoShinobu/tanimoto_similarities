import time
import random
import sys
from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt 
from multiprocessing import Pool, Manager


def tanimoto_similarity(args):
    pos, tanimotoArray, combinationsArray1, combinationsArray2, tanimotoResultArray = args
    mol1 = combinationsArray1[pos]
    mol2 = combinationsArray2[pos]
    Na = sum(tanimotoArray[mol1])
    Nb = sum(tanimotoArray[mol2])
    Nc = sum(tanimotoArray[mol1] & tanimotoArray[mol2])
    tanimotoResultArray[pos] = Nc / (Na + Nb - Nc)

if __name__ == "__main__":
    # Reading the input CSV file.
    ligands_df = pd.read_csv("smiles2.0.csv", index_col=0)

    # Creating molecules and storing in an array
    molecules = [Chem.MolFromSmiles(smiles) for smiles in ligands_df["SMILES"].tolist()]

    # Creating fingerprints for all molecules
    fgrps = [Chem.RDKFingerprint(mol) for mol in molecules]
    fgrps = fgrps[:1000]

    # Calculating number of fingerprints
    nfgrps = len(fgrps)
    print("Number of fingerprints:", nfgrps)

    inicioGPU = time.time()

    # Combinations
    combinations = np.transpose(np.triu_indices(nfgrps, k=1))
    combinationsArray1 = combinations[:, 0].astype("uint16")
    combinationsArray2 = combinations[:, 1].astype("uint16")

    tanimotoArray = np.zeros((len(fgrps), fgrps[0].GetNumBits()), dtype=np.uint8)
    for l, fp in enumerate(fgrps):
        DataStructs.ConvertToNumpyArray(fp, tanimotoArray[l])

    cant = int(((nfgrps - 1) * nfgrps) / 2)
    tanimotoResultArray = Manager().list([0] * cant)

    with Pool() as pool:
        pool.map(tanimoto_similarity, [(pos, tanimotoArray, combinationsArray1, combinationsArray2, tanimotoResultArray) for pos in range(len(tanimotoResultArray))])

    tanimotoResultArray = list(tanimotoResultArray)
    
    finGPU = time.time()

    print("El tiempo con paralelizacion fue de:", finGPU - inicioGPU)