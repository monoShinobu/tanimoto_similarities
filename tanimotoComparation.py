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
cantFgrps = 100

fgrpsAux = fgrps[0:100]


# Defining a function to calculate similarities among the molecules
def pairwise_similarity(fingerprints_list):
    
    global similarities

    similarities = np.zeros((len(fgrpsAux), len(fgrpsAux)))

    for a in range(1, len(fgrpsAux)):
            similarity = DataStructs.BulkTanimotoSimilarity(fgrpsAux[a], fgrpsAux[:a])
            similarities[a, :a] = similarity
            similarities[:a, a] = similarity
            
    return similarities


for i in range(cambiar):
    print(i)
    # Calculating similarities of molecules
    inicio = time.time()
    
    pairwise_similarity(fgrpsAux)
    tri_lower_diag = np.tril(similarities, k=0)
    
    fin = time.time()
    
    tiempoRDKIT[i] = fin-inicio
    
    
    
    #GPU high computing
    import pyopencl as cl
    #import os
    #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    
    #combinaciones posibles
    
    cant = int(((0+len(fgrpsAux)-1)*len(fgrpsAux))/2)
    
    combinations = np.zeros(cant*2,dtype=np.int32)
    
    juan = len(fgrpsAux)-1
    pedro = 0
    cont = 0
    start = 0
    start2 = 1
    
    for h in range(cant):
        if cont/juan >= 1:
            pedro += 1
            juan -= 1
            cont = 0
            start = start2
            start2 += 1
            if juan == 0:
                juan = 1
        combinations[h*2] = int(pedro)
        combinations[h*2+1] = int(start + 1)
        cont += 1
        start += 1
        
    #transformacion de los fingerprints de RDKIT a data que se puede mandar a la gpu
    fp_arr = np.zeros(2048, dtype=np.int32)
    DataStructs.ConvertToNumpyArray(fgrpsAux[0], fp_arr)
    fgrpsGpu=np.array(fp_arr)
    
    for j in range(1, len(fgrpsAux)):
        fp_arr = np.zeros(2048)
        DataStructs.ConvertToNumpyArray(fgrpsAux[j], fp_arr)
        
        fgrpsGpu = np.append(fgrpsGpu, fp_arr)
    
    fgrpsGpu = fgrpsGpu.astype("int32")
    





    inicioGPU = time.time()
    
    #iteracion para cada particula
    tanimotoResult = np.array(np.zeros(cant+1, dtype=np.float32))
     
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    mf = cl.mem_flags
    
    tanimotoArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fgrpsGpu)
    combinationsArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=combinations)
    #NM = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=nm)
    tanimotoResultArray = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=tanimotoResult)
    
    f = open('tanimotoSimilarity.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()
    
    prg = cl.Program(ctx, kernels).build()
    
    
    knl = prg.tanimoto_similarity(queue, (cant+1,), None, tanimotoArray, combinationsArray, tanimotoResultArray)
    knl.wait()
    
    finGPU = time.time()
    
    tiempoGPU[i] = finGPU-inicioGPU
    
    cl.enqueue_copy(queue, tanimotoResult, tanimotoResultArray).wait()
    
    cantFgrps = cantFgrps + 100
    
    fgrpsAux = fgrps[0:cantFgrps]
    
# Graficar ambas líneas
plt.plot(tiempoRDKIT, label="Tiempos Rdkit")
plt.plot(tiempoGPU, label="Tiempos GPU")

# Agregar título y etiquetas de los ejes
plt.title("GPU Vs Rdkit")
plt.xlabel("ejecucion")
plt.ylabel("Tiempo")

# Agregar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()
