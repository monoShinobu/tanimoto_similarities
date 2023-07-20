import os
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdMolDescriptors
import numpy as np
import copy

def duplicityProcess(dictLigand, pathHeaderSMI):
	fragListName = []

	#array with all the fingerprints of the molecules
	fgrpsGpu = np.array([])
	molSizes = []
	nfgrps = 0
	for mol,fragList in dictLigand.items():
		#print(mol,fragList)
		for idFrag in fragList:
			fragName = mol + "-f" + str(idFrag)
			fileSMI = pathHeaderSMI + fragName + ".smi"
			fragFile = open(fileSMI)
			line = fragFile.readline()
			fragFile.close()
			fragSMI = line.strip()
			print(fragSMI)
			fragMOL = Chem.MolFromSmiles(fragSMI)
			fragFP = FingerprintMols.FingerprintMol(fragMOL)

			#fragFP2 = Chem.RDKFingerprint(fragMOL)
			#V2
			
			fp_vec = rdMolDescriptors.GetMorganFingerprintAsBitVect(
				copy.deepcopy(fragMOL),
				radius=2,
				nBits=2048,
			)
			#fp_array = np.frombuffer(fp_vec.ToBitString().encode(), dtype=np.uint8) - ord('0')
			#fp_array1 = np.frombuffer(fragFP.ToBitString().encode(), dtype=np.uint8) - ord('0')
			molSizes.append(len(fp_vec))
			

			#put fingerprints on the array
			#fp_string = fragFP.ToBitString()
			#fp_bytes = fp_string.encode()
			#fp_memoryview = memoryview(fp_bytes)
			
			#V1
			'''
			fp_array = np.frombuffer(memoryview(fragFP.ToBitString().encode()), dtype=np.uint8) - ord('0')
			print("+++++++++++++++++++++++++++++++++++",len(fp_array))
			molSizes.append(len(fp_array))
			dif = 2048 - len(fp_array)
			fp_array = np.append(fp_array, np.zeros(dif, dtype=np.int32))
			'''
			#fp_array = np.array(list(fp_string), dtype=np.int32)
			#print("+++++++++++++++++++++++++++++++++++",len(fp_array))
			fgrpsGpu = np.append(fgrpsGpu,copy.deepcopy(fp_vec))
			nfgrps += 1

			fragListName.append((fragName,mol,idFrag, fragFP))
	fgrpsGpu = np.array(fgrpsGpu, dtype=np.uint8)
	molSizes = np.array(molSizes, dtype=np.int32)

	return fgrpsGpu, molSizes, nfgrps, fragListName