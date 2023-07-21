import os
import subprocess
from rdkit import Chem, DataStructs
from rdkit.Chem import BRICS
from rdkit.Chem.Fingerprints import FingerprintMols
from openbabel import pybel
import numpy as np
import copy
from multiprocessing import Pool, Manager


TOL_SIM = 1e-9
#folder KL contains known ligands in PDB format.
folderKL = "KL" + os.sep
#checking if FAIL folder exists, otherwise it creates
folderFAIL = "FAIL" + os.sep
if not os.path.isdir(folderFAIL):
	os.mkdir(folderFAIL)
#checking if folder for storing fragments exists, otherwise it creates
#folders for Headers and Bodies.
folderFL = "FL" + os.sep
folderSL_H, folderSL_B = "SL-Header" + os.sep, "SL-Body" + os.sep
folderSMI, folderPDB = "SMI" + os.sep, "PDB" + os.sep
pathHeaderSMI = folderFL + folderSL_H + folderSMI
pathBodySMI = folderFL + folderSL_B + folderSMI
pathHeaderPDB = folderFL + folderSL_H + folderPDB
pathBodyPDB = folderFL + folderSL_B + folderPDB
if not os.path.isdir(folderFL):
	subprocess.check_call(["mkdir",folderFL])
	subprocess.check_call(["mkdir",folderFL + folderSL_H])
	subprocess.check_call(["mkdir",pathHeaderSMI])
	subprocess.check_call(["mkdir",pathHeaderPDB])
	subprocess.check_call(["mkdir",folderFL + folderSL_B])
	subprocess.check_call(["mkdir",pathBodySMI])
	subprocess.check_call(["mkdir",pathBodyPDB])

'''
def deconstruction(folderKL):
	N_FAIL,N_FRAG,N_FRAG_NOT = 0,0,0
	dictLigand = {}
	ligandKL = os.listdir(folderKL)
	N_KL = len(ligandKL)
	for ligand in ligandKL:
		#convert PDB to SMI format.	
		nameSplit = ligand.split(".")
		ligandName = nameSplit[0]
		print("Fragment ... ",ligandName)
		inLigand = folderKL + ligand
		outLigand = folderKL + ligandName + ".smi"
		subprocess.check_call(["obabel", inLigand, "-O", outLigand])
		#fragment the molecule in SMILE format.
		# if it fails, .pdb and .smi are moved to folder FAIL.
		try:
			fragSet = fragmentLigand(outLigand)
			N_FRAG += len(fragSet)
			#print("Fragments: ", fragSet)
			n = 0
			listId = []
			for frag in fragSet:
				#create fragment as SMI file.
				fragNameSMI = pathHeaderSMI + ligandName + "-f" + str(n) + ".smi"
				fOut = open(fragNameSMI, "w")
				#print("%40s %5s %10s"%(frag, "--> ", ligandName + "-f" + str(n) + ".smi"))
				fOut.write(frag + "\n")
				fOut.close()
				fragNameSDF = pathHeaderSMI + ligandName + "-f" + str(n) + ".sdf"
				#convert SMI to SDF and validate fragment.
				subprocess.check_call(["obabel", fragNameSMI, "-O", fragNameSDF, "--gen2d"], stderr=subprocess.STDOUT)
				check = validFragment(fragNameSDF)
				if check:
					#convert SMI to PDB and store in PDB folder.
					fragNamePDB = pathHeaderPDB + ligandName + "-f" + str(n) + ".pdb"
					subprocess.check_call(["obabel", fragNameSMI, "-O", fragNamePDB, "--gen3d"], stderr=subprocess.STDOUT)
					listId.append(n)
				else:
					N_FRAG_NOT += 1
					#remove fragment in SMI and SDF format.
					os.remove(fragNameSMI)
					os.remove(fragNameSDF)
				n +=1
			dictLigand[ligandName] = listId

		except TypeError:
			N_FAIL += 1
			print("It's not possible to fragment",ligandName)
			subprocess.check_call(["mv", inLigand, folderFAIL])
			subprocess.check_call(["mv", outLigand, folderFAIL])
	return dictLigand,N_KL,N_FAIL,N_FRAG,N_FRAG_NOT	
'''

def process_ligand(ligand, dictLigand, N_FAIL, N_FRAG, N_FRAG_NOT):
    # convert PDB to SMI format.
    nameSplit = ligand.split(".")
    ligandName = nameSplit[0]
    print("Fragment ... ", ligandName)
    inLigand = folderKL + ligand
    outLigand = folderKL + ligandName + ".smi"
    subprocess.check_call(["obabel", inLigand, "-O", outLigand])

    try:
        fragSet = fragmentLigand(outLigand)
        N_FRAG.value += len(fragSet)
        
        n = 0
        listId = []
        for frag in fragSet:
            # create fragment as SMI file.
            fragNameSMI = pathHeaderSMI + ligandName + "-f" + str(n) + ".smi"
            fOut = open(fragNameSMI, "w")
            fOut.write(frag + "\n")
            fOut.close()
            fragNameSDF = pathHeaderSMI + ligandName + "-f" + str(n) + ".sdf"
            subprocess.check_call(["obabel", fragNameSMI, "-O", fragNameSDF, "--gen2d"], stderr=subprocess.STDOUT)
            check = validFragment(fragNameSDF)
            if check:
                fragNamePDB = pathHeaderPDB + ligandName + "-f" + str(n) + ".pdb"
                subprocess.check_call(["obabel", fragNameSMI, "-O", fragNamePDB, "--gen3d"], stderr=subprocess.STDOUT)
                listId.append(n)
            else:
                N_FRAG_NOT.value += 1
                os.remove(fragNameSMI)
                os.remove(fragNameSDF)
            n += 1
        dictLigand[ligandName] = listId

    except TypeError:
        N_FAIL.value += 1
        print("It's not possible to fragment", ligandName)
        subprocess.check_call(["mv", inLigand, folderFAIL])
        subprocess.check_call(["mv", outLigand, folderFAIL])

def deconstruction(folderKL):
    manager = Manager()
    dictLigand = manager.dict()
    N_FAIL = manager.Value('i', 0)
    N_FRAG = manager.Value('i', 0)
    N_FRAG_NOT = manager.Value('i', 0)
    ligandKL = os.listdir(folderKL)
    N_KL = len(ligandKL)

    with Pool(processes=3) as pool:
        pool.starmap(process_ligand, [(ligand, dictLigand, N_FAIL, N_FRAG, N_FRAG_NOT) for ligand in ligandKL])

    return dict(dictLigand), N_KL, N_FAIL.value, N_FRAG.value, N_FRAG_NOT.value

def fragmentLigand(ligandSMI):
	f = open(ligandSMI)
	line = f.readline()
	f.close()
	line = line.strip().split()
	#print("SMI:",line[0])
	ligandMOL = Chem.MolFromSmiles(line[0])
	#print("MOL:",ligandMOL)
	frags = BRICS.BRICSDecompose(ligandMOL)
	#return a set of fragments
	return frags

def validFragment(fileSDF):
	logP, MW, HBD, HBA = 0.,0.,0.,0.
	print(fileSDF)
	for mol in pybel.readfile("sdf", fileSDF):
		values = mol.calcdesc()
		logP, MW = float(values["logP"]), float(values['MW']) 
		HBD, HBA = int(values['HBD']), int(values['HBA1'])
	#print(logP, MW, HBD, HBA)

	validStr = "Valid! MW: %.2f, logP: %.2f, HBD: %.2f, HBA: %.2f"
	valid = True
	if ((MW > 0) and (MW < 300)):
		if (logP <= 3):
			if ((HBD <= 3) and (HBD >= 0)):
				if ((HBA <= 3) and (HBA >= 0)):
					print(validStr%(MW,logP,HBD,HBA))        
				else:
					print("Invalid,HBA: %.2f ... removing fragment"%HBA)
					valid = False
			else:
				print("Invalid,HBD: %.2f ... removing fragment"%HBD)
				valid = False
		else:
			print("Invalid,logP: %.2f ... removing fragment"%logP)
			valid = False
	else:
		print("Invalid,MW: %.2f ... removing fragment"%MW)
		valid = False
	return valid

def process_ligandRead(mol, fragList, fragListName, lock):
	for idFrag in fragList:
		fragName = mol + "-f" + str(idFrag)
		fileSMI = pathHeaderSMI + fragName + ".smi"
		fragFile = open(fileSMI)
		line = fragFile.readline()
		fragFile.close()
		fragSMI = line.strip()
		fragMOL = Chem.MolFromSmiles(fragSMI)
		fragFP = FingerprintMols.FingerprintMol(fragMOL)
		with lock:
			fragListName.append([fragName, mol, idFrag, fragFP])

def process_similarityCalculation(i, n, fragListName, dupliSet, lock):
    frag_i = fragListName[i][-1]
    for j in range(i + 1, n):
        frag_j = fragListName[j][-1]
        sim = DataStructs.cDataStructs.TanimotoSimilarity(frag_i, frag_j)
        print("%s %s %.3f" % (fragListName[i][0], fragListName[j][0], sim))
        if sim > (1 - TOL_SIM):
            with lock:
            	dupliSet.append(fragListName[j])
	    
def process_ligandDelete(fragName, mol, idFrag, dictLigandShared, lock):
	fileSMI = pathHeaderSMI + fragName + ".smi"
	fileSDF = pathHeaderSMI + fragName + ".sdf"
	filePDB = pathHeaderPDB + fragName + ".pdb"
	os.remove(fileSMI)
	os.remove(fileSDF)
	os.remove(filePDB)
	with lock:
		dictLigandShared[mol] = [x for x in dictLigandShared[mol] if x != idFrag]

def key_function(element):
    fragName, mol, idFrag, _ = element
    return (fragName, mol, idFrag)

def duplicity(dictLigand):
	manage = Manager()
	lock = manage.Lock()

	fragListName = manage.list()
	with Pool(processes=2) as pool:
		pool.starmap(process_ligandRead, [(mol, fragList, fragListName, lock) for mol, fragList in dictLigand.items()])


	dupliSet = manage.list()
	n = len(fragListName)

	with Pool(processes=2) as pool:
		pool.starmap(process_similarityCalculation, [(i, n, fragListName, dupliSet, lock) for i in range(n)])

	dupliSet = set(key_function(element) for element in dupliSet)
	
	dictLigandShared = manage.dict(copy.deepcopy(dictLigand))
	with Pool(processes=2) as pool:
		pool.starmap(process_ligandDelete, [(fragName, mol, idFrag, dictLigandShared, lock) for (fragName,mol,idFrag) in dupliSet])
	
	return dictLigandShared, len(dupliSet)

def process_cleaner(mol,fragList):
	for idFrag in fragList:
		fragName = mol + "-f" + str(idFrag)
		filePDB = pathHeaderPDB + fragName + ".pdb" 
		textH, textC, textM, textCA, textEND = [], [], [], [], []
		c,z,s,d,d2 = 0,0,set(),{},{}
		f = open(filePDB)
		lines = f.readlines()
		f.close()
		for line in lines:
			if "HETATM" in line:
				if "*" in line:
					c += 1
					line = line.split()
					s.add(int(line[1]))
				else:
					line = line.split()
					idLine = int(line[1])
					d[idLine] = c
					textH.append(line)
			elif "CONECT" in line:
				line = line.split()
				listNum = line[1:]
				listNum = list(map(int,listNum))
				setNum = set(listNum)

				if len(s & setNum) == 0:
					listAux = []
					for num in listNum:
						newNum = num - d[num]
						listAux.append(newNum)
					listAux = ["CONECT"] + list(map(str, listAux))
					z += 1
					d2[z] = len(listAux)
					textC.append(listAux)

			elif "MASTER" in line:
				line = line.split()       
				textM.append(line)
			elif "COMPND" in line or "AUTHOR" in line:
				textCA.append(line)
			elif "END":
				textEND.append(line)
	
		text = ""
		for line in textCA:
			text += line

		for line in textH:
			idLine = int(line[1]) - d[int(line[1])]
			line[1] = str(idLine)
		
		for line in textH:
			text += "%-7s"%line[0] + "%4s"%line[1] + "%3s"%line[2] + "%6s"%line[3] + "%6s"%line[4] + "%12s"%line[5] + "%8s"%line[6] + "%8s"%line[7] + "%6s"%line[8] + "%6s"%line[9] + "%12s"%line[10] + "\n"

		for line in textC:
			text += '%-6s'%line[0]
			for i in range(1,len(line)):
				text += "%5s"%line[i]
			text += "\n"
					
		for line in textM:
			line[9] = str(idLine)
			line[11] = str(idLine)    
			
		for line in textM:
			text += "%-6s"%line[0] + "%9s"%line[1] + "%5s"%line[2] + "%5s"%line[3] + "%5s"%line[4] + "%5s"%line[5] + "%5s"%line[6] + "%5s"%line[7] + "%5s"%line[8] + "%5s"%line[9] + "%5s"%line[10] + "%5s"%line[11] + "%5s"%line[12] + "\n"  
			
		for line in textEND:
			text += line
			
		fileBodyPDB = pathBodyPDB + fragName + ".pdb"
		f = open(fileBodyPDB,"w")
		f.write(text)
		f.close()
		#convert PDB to SMI and store in SMI folder.
		fileBodySMI = pathBodySMI + fragName + ".smi"
		subprocess.check_call(["obabel", fileBodyPDB, "-O", fileBodySMI], stderr=subprocess.STDOUT)
	
def cleaner(dictLigand):
	with Pool(processes=3) as pool:
		pool.starmap(process_cleaner, [(mol,fragList) for mol,fragList in dictLigand.items()])
		
