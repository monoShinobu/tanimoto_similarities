from src.deconstruction import *

if __name__ == "__main__":
    start = time.time()

    deconTimeStart = time.time()
    dictLigand,N_KL,N_FAIL,N_FRAG,N_FRAG_NOT = deconstruction(folderKL)
    deconTimeEnd = time.time()

    

    N_FRAG_DUP = 0

    dupliTimeStart = time.time()
    dictLigand,N_FRAG_DUP = duplicity(dictLigand)
    dupliTimeEnd = time.time()
    

    cleanTimeStart = time.time()
    cleaner(dictLigand)
    cleanTimeEnd = time.time()

    print("Tiempo1", deconTimeEnd - deconTimeStart)
    print("Tiempo2", dupliTimeEnd - dupliTimeStart)
    print("Tiempo3", cleanTimeEnd - cleanTimeStart)

    end = time.time()

print("%-40s: %-5d"%("Number of known ligands",N_KL))
print("%-40s: %-5d"%("Number of fail ligands",N_FAIL))
print("%-40s: %-5d"%("Number of fragments",N_FRAG))
print("%-40s: %-5d"%("Number of fragments not valid",N_FRAG_NOT))
print("%-40s: %-5d"%("Number of similar fragments",N_FRAG_DUP))
print("%-40s: %-5d"%("Number of total fragments",N_FRAG - N_FRAG_NOT - N_FRAG_DUP))
print("Total time: %s sec." %round(end - start,2))



