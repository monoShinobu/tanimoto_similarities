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
from multiprocessing import Pool
import math

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

cambiar = 11

tiempoRDKIT = np.zeros(cambiar)
tiempoGPU = np.zeros(cambiar)
cantidad = np.zeros(cambiar)
cantFgrps = 100

fgrpsAux = fgrps[0:cantFgrps]

canto = 100

promGPU = 0
promRDKIT = 0

# Defining a function to calculate similarities among the molecules
def pairwise_similarity(fingerprints_list):
    
    global similarities

    similarities = np.zeros((len(fgrpsAux), len(fgrpsAux)))

    for a in range(1, len(fgrpsAux)):
            similarity = DataStructs.BulkTanimotoSimilarity(fgrpsAux[a], fgrpsAux[:a])
            similarities[a, :a] = similarity
            similarities[:a, a] = similarity
            
    return similarities


def pairwise_similarityCPU(args):
    i, fgrp_i, fgrps = args
    similarity = DataStructs.BulkTanimotoSimilarity(fgrp_i, fgrps[:i])
    return (i, similarity)

for i in range(cambiar):
    #print(i)
    
    # Calculating similarities of molecules
    inicio = time.time()
    
    pairwise_similarity(fgrpsAux)
    tri_lower_diag = np.tril(similarities, k=0)
    
    fin = time.time()
    
    tiempoRDKIT[i] = fin-inicio
    
    promRDKIT += fin-inicio
    
    
    
    
    
    
    
    '''
    ################################################################################
    ####################       Paralelizacion en CPU            ####################
    ################################################################################
    '''
    '''
    inicio = time.time()
    if __name__ == '__main__':
        
        similarities = np.zeros((nfgrps, nfgrps))

        args_list = [(i, fgrps[i], fgrps) for i in range(1, nfgrps)]

        with Pool(processes=4) as pool:
            results = pool.map(pairwise_similarityCPU, args_list)

        for i, similarity in results:
            similarities[i, :i] = similarity
            similarities[:i, i] = similarity
    fin = time.time()
    tiempoRDKIT[i] = fin-inicio
    '''
    




    '''
    ################################################################################
    ####################       Paralelizacion en GPU            ####################
    ################################################################################
    '''
    inicioGPU = time.time()
    
    #GPU high computing
    import pyopencl as cl
    #import os
    #os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

    #combinaciones posibles

    #progresion aritmetica
    cant = int(((cantFgrps-1)*cantFgrps)/2)



    # Combinaciones

    combinations = np.transpose(np.triu_indices(cantFgrps, k=1))
    arreglo1 = (combinations[:, 0]*2048).astype("uint16")
    arreglo2 = (combinations[:, 1]*2048).astype("uint16")


    '''
    fgrpsGpu = np.zeros((len(fgrps), fgrps[0].GetNumBits()), dtype=np.uint8)
    for l, fp in enumerate(fgrps):
        fgrpsGpu[l,:] = np.fromstring(fp.ToBitString(), dtype=np.uint8) - ord('0')
    '''

    fgrpsGpu = np.zeros((len(fgrpsAux), fgrpsAux[0].GetNumBits()), dtype=np.uint8)
    for l, fp in enumerate(fgrpsAux):
        fgrpsGpu[l,:] = np.frombuffer(memoryview(fp.ToBitString().encode()), dtype=np.uint8) - ord('0')


    #iteracion para cada particula
    tanimotoResult = np.zeros(cant+1, dtype=np.float32)

     
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    mf = cl.mem_flags

    tanimotoArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fgrpsGpu)
    combinationsArray1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo1)
    combinationsArray2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo2)
    tanimotoResultArray = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=tanimotoResult)

    f = open('tanimotoSimilarity2.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    prg = cl.Program(ctx, kernels).build()

    block_size = 256
    global_size = math.ceil(cant / block_size) * block_size

    knl = prg.tanimoto_similarity(queue, (cant,), None, tanimotoArray, combinationsArray1, combinationsArray2, tanimotoResultArray)

    cl.enqueue_copy(queue, tanimotoResult, tanimotoResultArray).wait()

    finGPU = time.time()

    queue.finish()
    tanimotoArray.release()
    combinationsArray1.release()
    combinationsArray2.release()
    tanimotoResultArray.release()
    
    finGPU = time.time()
    
    tiempoGPU[i] = finGPU-inicioGPU
    promGPU += finGPU-inicioGPU
    
    
    cantFgrps = cantFgrps + 1000
    
    cantidad[i] = canto
    
    canto += 1000
    
    fgrpsAux = fgrps[0:cantFgrps]
    
    print(i)
    
print(tiempoRDKIT)
print(tiempoGPU)

tiempoCPUMio=[0.6675608158111572,
           1.3760614395141602,
           2.330132007598877,
           3.1294760704040527,
           4.131476640701294,
           5.138068437576294,
           6.006777763366699,
           7.387864112854004,
           8.660151243209839,
           10.269251823425293,
           11.903439283370972
           ]

# Graficar ambas líneas
plt.plot(cantidad, tiempoRDKIT, label="Tiempos RDKit")
plt.plot(cantidad, tiempoCPUMio, label="Tiempos CPU")
plt.plot(cantidad, tiempoGPU, label="Tiempos GPU")

# Agregar título y etiquetas de los ejes
plt.title("Similaridad de tanimoto")
plt.xlabel("Cantidad de particulas")
plt.ylabel("Tiempo (seg)")

# Agregar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()
