import random
import sys
from pathlib import Path
import seaborn as sns
import time
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
from multiprocessing import Pool
import math
import pyopencl as cl


# show full results
np.set_printoptions(threshold=sys.maxsize)


# Reading the input CSV file.

ligands_df = pd.read_csv("smiles1.1.csv" , index_col=0 )
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

fgrps = fgrps[0:4]

# Calculating number of fingerprints
nfgrps = len(fgrps)
print("Number of fingerprints:", nfgrps)

#GPU high computing
import pyopencl as cl
#import os
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

#combinaciones posibles

inicioGPU = time.time()
#progresion aritmetica
cant = int(((nfgrps-1)*nfgrps)/2)


"""
# Crear los arreglos para guardar las combinaciones
combinations = np.transpose(np.triu_indices(nfgrps, k=1))
arreglo1 = combinations[:, 0].astype("int32")
arreglo2 = combinations[:, 1].astype("int32")
"""
"""
# Combinaciones
combinations = np.transpose(np.triu_indices(nfgrps, k=1))
arreglo1 = combinations[:, 0].astype("uint16")
arreglo2 = combinations[:, 1].astype("uint16")


"""
start = 1
salto = 2
largo = nfgrps - 1
arreglo1 = []
for i in range(nfgrps):
  for j in range(largo):
    arreglo1.append(start)
    start += 1
  start += salto
  salto += 1
  largo -= 1

arreglo1 = np.array(arreglo1).astype("int32")



fgrpsGpu = np.zeros((len(fgrps), fgrps[0].GetNumBits()), dtype=np.uint8)
for l, fp in enumerate(fgrps):
    fgrpsGpu[l,:] = np.fromstring(fp.ToBitString(), dtype=np.uint8) - ord('0')


#iteracion para cada particula
tanimotoResult = np.array(np.zeros(cant+1, dtype=np.float32))
 
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

tanimotoArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fgrpsGpu)
#combinationsArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combinations)
combinationsArray1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo1)
#combinationsArray2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo2)
tanimotoResultArray = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=tanimotoResult)
nfgrpsGPU = np.int32(nfgrps)

f = open('tanimotoSimilarity3.cl', 'r', encoding='utf-8')
kernels = ''.join(f.readlines())
f.close()

prg = cl.Program(ctx, kernels).build()


#knl = prg.tanimoto_similarity(queue, (cant,), None, tanimotoArray, combinationsArray1, combinationsArray2, tanimotoResultArray)
knl = prg.tanimoto_similarity(queue, (cant,), (256, ), tanimotoArray, combinationsArray1,nfgrpsGPU, tanimotoResultArray)

cl.enqueue_copy(queue, tanimotoResult, tanimotoResultArray).wait()

finGPU = time.time()

queue.finish()
tanimotoArray.release()
combinationsArray1.release()

tanimotoResultArray.release()

#print(tanimotoResult)
print("El tiempo con paralelizacion fue de:", finGPU-inicioGPU)

