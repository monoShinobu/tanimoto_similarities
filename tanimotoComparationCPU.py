#!/usr/bin/env python3

import time
import random
import sys
from pathlib import Path
import math

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from multiprocessing import Pool, Manager

    

def pairwise_similarityCPU(a, fgrpsAux):
    similarity = DataStructs.BulkTanimotoSimilarity(fgrpsAux[a], fgrpsAux[:a])
    return similarity

if __name__ == "__main__":
    # show full results
    np.set_printoptions(threshold=sys.maxsize)
    
    
    # Reading the input CSV file.
    
    ligands_df = pd.read_csv("smiles2.0.csv" , index_col=0 )
    #print(ligands_df.head())
    
    
    
    # Creating molecules and storing in an array
    molecules = []
    
    """Let's fetch the smiles from the input file and store in molecules array
            We have used '_' because we don't want any other column.
            If you want to fetch index and any other column, then replace '_' with 
                index and write column names after a ','.
    """
    
    for _, smiles in ligands_df[[ "SMILES"]].itertuples():
        molecules.append((Chem.MolFromSmiles(smiles)))
    molecules[:10000]
    
    
    # Creating fingerprints for all molecules
    
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)
    fgrps = [rdkit_gen.GetFingerprint(mol) for mol in molecules]
    
    
    # Calculating number of fingerprints
    nfgrps = len(fgrps)
    print("Number of fingerprints:", nfgrps)
    
    fgrpsAux = fgrps[0:10000]
    i = 0

    inicioCPU = time.time()
    
    with Pool(processes=10) as pool:
        similarities_list = pool.starmap(pairwise_similarityCPU, [(a, fgrpsAux) for a in range(1, len(fgrpsAux))])
        
    finCPU = time.time()
    tiempoCPU = finCPU - inicioCPU
    
    print("Tiempo CPU:", tiempoCPU)
    #print("Similitudes calculadas:", similarities_list)

    
