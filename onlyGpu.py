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

inicioGPU = time.time()

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

#GPU high computing
import pyopencl as cl
#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#combinaciones posibles

#progresion aritmetica
cant = int(((nfgrps-1)*nfgrps)/2)


'''
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
'''

# Crear los arreglos para guardar las combinaciones
combinations = np.transpose(np.triu_indices(nfgrps, k=1))
arreglo1 = combinations[:, 0].astype("int32")
arreglo2 = combinations[:, 1].astype("int32")


'''
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




fgrpsGpu = np.zeros((len(fgrps), fgrps[0].GetNumBits()), dtype=np.int64)
for i, fp in enumerate(fgrps):
    fgrpsGpu[i,:] = np.fromstring(fp.ToBitString(), dtype=np.uint8) - ord('0')



#print(fgrpsGpu)



#iteracion para cada particula
tanimotoResult = np.array(np.zeros(cant+1, dtype=np.float32))
 
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

tanimotoArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fgrpsGpu)
#combinationsArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combinations)
combinationsArray1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo1)
combinationsArray2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo2)
tanimotoResultArray = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=tanimotoResult)

f = open('tanimotoSimilarity2.cl', 'r', encoding='utf-8')
kernels = ''.join(f.readlines())
f.close()

prg = cl.Program(ctx, kernels).build()


knl = prg.tanimoto_similarity(queue, (cant+1,), None, tanimotoArray, combinationsArray1, combinationsArray2, tanimotoResultArray)

cl.enqueue_copy(queue, tanimotoResult, tanimotoResultArray).wait()

finGPU = time.time()

#print(tanimotoResult)
print("El tiempo con paralelizacion fue de:", finGPU-inicioGPU)

