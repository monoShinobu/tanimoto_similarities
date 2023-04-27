'''
import pyopencl as cl
import numpy as np

# Crea un contexto y una cola de comandos para la GPU
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Define el kernel que se utilizará en los programas
kernel = """
__kernel void multiply(__global const float *input, __global float *output) {
    int gid = get_global_id(0);
    output[gid] = input[gid] * gid;
}
"""

# Genera datos aleatorios para enviar a la GPU
data = np.random.rand(100).astype(np.float32)

# Divide los datos en secciones y crea un buffer para cada sección
num_sections = 4
section_size = len(data) // num_sections

buffers = []
for i in range(num_sections):
    section_start = i * section_size
    section_end = (i+1) * section_size
    section_data = data[section_start:section_end]
    buffer = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=section_data)
    buffers.append(buffer)

# Crea los programas con el kernel definido y compílalos
programs = []
for i in range(num_sections):
    program = cl.Program(ctx, kernel).build()
    programs.append(program)

# Asigna cada buffer a un programa y ejecuta cada programa en paralelo
output_buffers = []
for i in range(num_sections):
    program = programs[i]
    section_buffer = buffers[i]
    output_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=section_size*4)
    output_buffers.append(output_buffer)
    program.multiply(queue, (section_size,), None, section_buffer, output_buffer)

# Copia los resultados de cada buffer a la CPU y procesa los resultados
for i in range(num_sections):
    output_data = np.empty(section_size, dtype=np.float32)
    cl.enqueue_copy(queue, output_data, output_buffers[i])
    print(f"Section {i} output: {output_data}")
'''
    

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

inicioGPU = time.time()
# show full results
np.set_printoptions(threshold=sys.maxsize)


# Reading the input CSV file.

ligands_df = pd.read_csv("smiles2.0.csv" , index_col=0 )
#print(ligands_df.head())



# Creating molecules and storing in an array
molecules = []


for _, smiles in ligands_df[[ "SMILES"]].itertuples():
    molecules.append((Chem.MolFromSmiles(smiles)))
molecules[:15]


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

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

mf = cl.mem_flags

f = open('tanimotoSimilarity2.cl', 'r', encoding='utf-8')
kernels = ''.join(f.readlines())
f.close()

prg = cl.Program(ctx, kernels).build()

#combinaciones posibles
#progresion aritmetica
cant = int(((nfgrps-1)*nfgrps)/2)


juan = nfgrps-1
pedro = 0
cont = 0
start = 0
start2 = 1

fingerPrintGPU1 = np.zeros(2048, dtype=np.int32)
fingerPrintGPU2 = np.zeros(2048, dtype=np.int32)

output_buffers = []

for i in range(cant):
    if cont/juan >= 1:
        pedro += 1
        juan -= 1
        cont = 0
        start = start2
        start2 += 1
        if juan == 0:
            juan = 1
    
    DataStructs.ConvertToNumpyArray(fgrps[pedro], fingerPrintGPU1)
    DataStructs.ConvertToNumpyArray(fgrps[start + 1], fingerPrintGPU2)
    
    buffer1 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fingerPrintGPU1)
    buffer2 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=fingerPrintGPU2)
    
    output_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=1)
    output_buffers.append(output_buffer)
    
    prg.tanimoto_similarity(queue, (cant+1,), None, buffer1, buffer2, output_buffer)
    
    cont += 1
    start += 1

# Copia los resultados de cada buffer a la CPU y procesa los resultados
for i in range(cant):
    output_data = np.empty(1, dtype=np.float32)
    cl.enqueue_copy(queue, output_data, output_buffers[i])
    print(f"Section {i} output: {output_data}")

finGPU = time.time()

#print(tanimotoResult)
print("El tiempo con paralelizacion fue de:", finGPU-inicioGPU)
