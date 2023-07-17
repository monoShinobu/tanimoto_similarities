
__kernel void tanimoto_similarity(__global int *tanimotoArray,
                                  __global ushort *combinationsArray1,
                                  __global ushort *combinationsArray2,
                                  __global float *tanimotoResultArray) {
	//uchar 
  int global_id = get_global_id(0);
  int i;
  int mol1 = combinationsArray1[global_id];
  int mol2 = combinationsArray2[global_id];
  int Na = 0;
  int Nb = 0;
  int Nc = 0;


  for (i = 0; i < 2048; i++) {
    Na = Na + tanimotoArray[mol1 + i];
    Nb = Nb + tanimotoArray[mol2 + i];
    
    if (tanimotoArray[mol1 + i] == 1 && tanimotoArray[mol2 + i] == 1) {
      Nc = Nc + 1;
    }
  }
  tanimotoResultArray[global_id] = (float)Nc / (Na + Nb - Nc);
}