# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:07:02 2023

@author: Jaime Velásquez
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import sys
import time
from multiprocessing import Pool, cpu_count
import seaborn as sns
import matplotlib.pyplot as plt 



# Defining a function to calculate similarities among the molecules
def pairwise_similarityCPU(args):
    i, fgrp_i, fgrps = args
    similarity = DataStructs.BulkTanimotoSimilarity(fgrp_i, fgrps[:i])
    return (i, similarity)

if __name__ == '__main__':
    
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
    
    inicio = time.time()
    
    nfgrps = len(fgrps)
    print("Number of fingerprints:", nfgrps)

    similarities = np.zeros((nfgrps, nfgrps))

    args_list = [(i, fgrps[i], fgrps) for i in range(1, nfgrps)]

    with Pool(processes=4) as pool:
        results = pool.map(pairwise_similarityCPU, args_list)

    for i, similarity in results:
        similarities[i, :i] = similarity
        similarities[:i, i] = similarity
    
    fin = time.time()
    print("Tiempo con el uso de paralelizacion",fin-inicio)
    tri_lower_diag = np.tril(similarities, k=0)
    
    '''
    # definging labels to show on heatmap
    labels = ['lig1','lig2','lig3','lig4','lig5','lig6','lig7', 'lig8', 'lig9', 'lig10', 'lig11', 'lig12', 'lig13', 'lig14', 'lig15']
        
    def lower_tri_heatmap (sim):
        f = open("similarities_lower_tri.txt", "w")

        print (tri_lower_diag, file=f)

        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        lower_tri_plot = sns.heatmap(tri_lower_diag[:15,:15], annot = False, cmap=cmap,center=0,
                square=True, xticklabels=labels, yticklabels=labels, linewidths=.7, cbar_kws={"shrink": .5})

        plt.title('Heatmap of Tanimoto Similarities', fontsize = 20)

        plt.show()

        fig = lower_tri_plot.get_figure()
        fig.savefig("tanimoto_heatmap_lw_tri.png") 

    
    
    lower_tri_heatmap(similarities)
    '''
    
    
    