
__kernel void tanimoto_similarity(__global int *tanimotoArray,
                                  __global int *combinationsArray1, int nfgrps,
                                  __global float *tanimotoResultArray) {

  int global_id = get_global_id(0);
  int i;
  int mol1 = combinationsArray1[global_id] / nfgrps;
  int mol2 = combinationsArray1[global_id] - mol1 * nfgrps;
  int Na = 0;
  int Nb = 0;
  int Nc = 0;

  mol1 = mol1*2048;
  mol2 = mol2*2048;

  for (i = 0; i < 2048; i++) {
    Na = Na + tanimotoArray[mol1 + i];
    Nb = Nb + tanimotoArray[mol2 + i];
    if (tanimotoArray[mol1 + i] == 1 && tanimotoArray[mol2 + i] == 1) {
      Nc = Nc + 1;
    }
  }
  tanimotoResultArray[global_id] = (float)Nc / (Na + Nb - Nc);
  //printf("%d,%d    - %f       - %d %d %d", mol1, mol2,(float)Nc / (Na + Nb - Nc), Na, Nb, Nc);
}