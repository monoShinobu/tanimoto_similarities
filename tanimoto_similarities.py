#!/usr/bin/env python3

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

# show full results
np.set_printoptions(threshold=sys.maxsize)


# Reading the input CSV file.

ligands_df = pd.read_csv("smiles1.0.csv" , index_col=0 )
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
inicio = time.time()

pairwise_similarity(fgrps)
tri_lower_diag = np.tril(similarities, k=0)


fin = time.time()

print("El tiempo sin paralelizacion fue de:", fin-inicio)
# Visulaizing the similarities

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

#GPU high computing
import pyopencl as cl
#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#combinaciones posibles

cant = int(((0+nfgrps-1)*nfgrps)/2)

combinations = np.zeros(cant*2,dtype=np.int32)

juan = nfgrps-1
pedro = 0
cont = 0
start = 0
start2 = 1

for i in range(cant):
    if cont/juan >= 1:
        pedro += 1
        juan -= 1
        cont = 0
        start = start2
        start2 += 1
        if juan == 0:
            juan = 1
    combinations[i*2] = int(pedro)
    combinations[i*2+1] = int(start + 1)
    cont += 1
    start += 1
    
#transformacion de los fingerprints de RDKIT a data que se puede mandar a la gpu
fp_arr = np.zeros(2048, dtype=np.int32)
DataStructs.ConvertToNumpyArray(fgrps[0], fp_arr)
fgrpsGpu=np.array(fp_arr)

for i in range(1, nfgrps):
    fp_arr = np.zeros(2048)
    DataStructs.ConvertToNumpyArray(fgrps[i], fp_arr)
    
    fgrpsGpu = np.append(fgrpsGpu, fp_arr)

fgrpsGpu = fgrpsGpu.astype("int32")



'''
fp_arr5 = np.zeros(2048,dtype=np.int32)
fp_arr4 = np.zeros(2048,dtype=np.int32)
DataStructs.ConvertToNumpyArray(fgrps[0], fp_arr5)
DataStructs.ConvertToNumpyArray(fgrps[1], fp_arr4)

count = 0

for i in range(2048):
    if (fp_arr5[i] == 1 and fp_arr4[i] == 1):
        count += 1

print("nc = ", count)
print("na = ", nm[0])
print("nb = ", nm[1])
print("tanimoto = ", count/(nm[0]+nm[1]-count))
'''

inicioGPU = time.time()

#iteracion para cada particula
tanimotoResult = np.array(np.zeros(cant+1, dtype=np.float32))
 
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

tanimotoArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fgrpsGpu)
combinationsArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combinations)
tanimotoResultArray = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=tanimotoResult)

f = open('tanimotoSimilarity.cl', 'r', encoding='utf-8')
kernels = ''.join(f.readlines())
f.close()

prg = cl.Program(ctx, kernels).build()


knl = prg.tanimoto_similarity(queue, (cant+1,), None, tanimotoArray, combinationsArray, tanimotoResultArray)
knl.wait()

finGPU = time.time()

cl.enqueue_copy(queue, tanimotoResult, tanimotoResultArray).wait()

#print(tanimotoResult)
print("El tiempo con paralelizacion fue de:", finGPU-inicioGPU)

