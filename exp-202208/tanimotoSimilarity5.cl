
__kernel void tanimoto_similarity(__global uchar *tanimotoArray,
                                  __global ushort *combinationsArray1,
                                  __global ushort *combinationsArray2,
                                  __global float *tanimotoResultArray,
                                  __global int *molSizesGPU ) {
	//uchar 
  int global_id = get_global_id(0);
  int i;
  int mol1 = combinationsArray1[global_id];
  int mol2 = combinationsArray2[global_id];
  int Na = 0;
  int Nb = 0;
  int Nc = 0;
  int fin = 0;
  int molSize1 = molSizesGPU[mol1 / 2048];
  int molSize2 = molSizesGPU[mol2 / 2048];

  if ( molSize1 < molSize2 ){
    fin = molSize1;
  } else {
    fin = molSize2;
  }

  for (i = 0; i < fin; i++) {
    Na = Na + tanimotoArray[mol1 + i];
    Nb = Nb + tanimotoArray[mol2 + i];
    
    if (tanimotoArray[mol1 + i] == 1 && tanimotoArray[mol2 + i] == 1) {
      Nc = Nc + 1;
    }
  }
  tanimotoResultArray[global_id] = (float)Nc / (Na + Nb - Nc);
}