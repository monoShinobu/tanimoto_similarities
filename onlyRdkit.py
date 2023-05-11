import time
import random
import sys
from pathlib import Path
import seaborn as sns

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

#fp_arr = np.zeros((1,))

inicio = time.time()

# Defining a function to calculate similarities among the molecules
def pairwise_similarity(fingerprints_list):
    
    global similarities

    similarities = np.zeros((nfgrps, nfgrps))

    for i in range(1, nfgrps):
            similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
            similarities[i, :i] = similarity
            similarities[:i, i] = similarity
            
    return similarities


# Calculating similarities of molecules

pairwise_similarity(fgrps)
#tri_lower_diag = np.tril(similarities, k=0)


fin = time.time()

print("El tiempo sin paralelizacion fue de:", fin-inicio)
# Visulaizing the similarities

'''
################################################################################
####################       Paralelizacion en CPU            ####################
################################################################################
'''


from multiprocessing import Pool
import multiprocessing as mp

inicio = time.time()


import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import sys
import time

'''
def pairwise_similarity(args):
    i, fgrps = args

    similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
    #print(similarity)
    #similarities[i, :i] = similarity
    #similarities[:i, i] = similarity

if __name__ == '__main__':
    # show full results
    np.set_printoptions(threshold=sys.maxsize)


    # Reading the input CSV file.

    ligands_df = pd.read_csv("smiles.csv" , index_col=0 )
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

    similarities = np.zeros((nfgrps, nfgrps))

    # Calculating similarities of molecules

    arr = np.arange(1, nfgrps+1)
    
    with Pool() as pool:
        pool.map(pairwise_similarity, [(i, fgrps) for i in range(nfgrps)])

    #tri_lower_diag = np.tril(similarities, k=0)

    fin = time.time()
    print("Tiempo con el uso de paralelizacion",fin-inicio)


    # definging labels to show on heatmap
    labels = ['lig1','lig2','lig3','lig4','lig5','lig6','lig7', 'lig8', 'lig9', 'lig10', 'lig11', 'lig12', 'lig13', 'lig14', 'lig15']
        
        
    def normal_heatmap (sim):
    
        # writing similalrities to a file
        f = open("similarities.txt", "w")
        print (similarities, file=f)
    
        sns.set(font_scale=0.8)
    
        # generating the plot
        
        plot = sns.heatmap(sim[:15,:15], annot = True, annot_kws={"fontsize":5}, center=0,
                square=True, xticklabels=labels, yticklabels=labels, linewidths=.7, cbar_kws={"shrink": .5})
    
        plt.title('Heatmap of Tanimoto Similarities', fontsize = 20) # title with fontsize 20
    
        plt.show()
    
        # saving the plot
    
        fig = plot.get_figure()
        fig.savefig("tanimoto_heatmap.png") 
    
    
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
    
    normal_heatmap(similarities)

    lower_tri_heatmap(similarities)
    '''
        




