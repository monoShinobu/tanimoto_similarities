#!/usr/bin/env python3

import time
import random
import sys
from pathlib import Path
import seaborn as sns
import math

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from multiprocessing import Pool, Manager

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
molecules[:15]


# Creating fingerprints for all molecules

rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)
fgrps = [rdkit_gen.GetFingerprint(mol) for mol in molecules]


# Calculating number of fingerprints
nfgrps = len(fgrps)
print("Number of fingerprints:", nfgrps)

#fp_arr = np.zeros((1,))

cambiar = 100

tiempoRDKIT = np.zeros(cambiar)
tiempoGPU = np.zeros(cambiar)
tiempoCPU = np.zeros(cambiar)
cantFgrps = 10000

fgrpsAux = fgrps[0:100]

manager = Manager()


# Defining a function to calculate similarities among the molecules
def pairwise_similarity(fingerprints_list):
    
    global similarities

    similarities = np.zeros((len(fgrpsAux), len(fgrpsAux)))

    for a in range(1, len(fgrpsAux)):
            similarity = DataStructs.BulkTanimotoSimilarity(fgrpsAux[a], fgrpsAux[:a])
            similarities[a, :a] = similarity
            similarities[:a, a] = similarity
            
    return similarities


def pairwise_similarityCPU(a, similaritiesCPU):
    similarity = DataStructs.BulkTanimotoSimilarity(fgrpsAux[a], fgrpsAux[:a])
    similaritiesCPU[a, :a] = similarity
    similaritiesCPU[:a, a] = similarity

for i in range(cambiar):
    #print(i)
    
    
    
    # Calculating similarities of molecules
    inicio = time.time()
    
    pairwise_similarity(fgrpsAux)
    tri_lower_diag = np.tril(similarities, k=0)
    
    fin = time.time()
    
    tiempoRDKIT[i] = fin-inicio
    
    
    print("hola1")
    
    
    
    
    '''
    ################################################################################
    ####################       Paralelizacion en CPU            ####################
    ################################################################################
    '''

    inicioCPU = time.time()
    
    

    similaritiesCPU = manager.Array('d', np.zeros((len(fgrpsAux), len(fgrpsAux))))
    

    with Pool() as pool:
        pool.starmap(pairwise_similarityCPU, [(a, similaritiesCPU) for a in range(1, len(fgrpsAux))])
        
    finCPU = time.time()
    tiempoCPU[i] = finCPU-inicioCPU
    
    print("hola2")
    

    
    #configuracion para cad iteracion
    
    cantFgrps = cantFgrps + 100
    
    fgrpsAux = fgrps[0:cantFgrps]
    
    print(i)
    
