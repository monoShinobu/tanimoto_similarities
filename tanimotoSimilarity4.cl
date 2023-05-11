
__kernel void tanimoto_similarity(__global uchar *tanimotoArray,
                                   __global ushort *combinationsArray1,
                                   __global ushort *combinationsArray2,
                                   __global float *tanimotoResultArray) {

  int global_id = get_global_id(0);
  int i;
  int mol1 = combinationsArray1[global_id];
  int mol2 = combinationsArray2[global_id];
  ushort Na = 0;
  ushort Nb = 0;
  ushort Nc = 0;
  __local uchar local_tanimotoArray[2048][16];

  mol1 = mol1*2048;
  mol2 = mol2*2048;

  for (i = 0; i < 2048; i++) {
    local_tanimotoArray[i][0] = tanimotoArray[mol1 + i];
    local_tanimotoArray[i][1] = tanimotoArray[mol2 + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    Na = Na + local_tanimotoArray[i][0];
    Nb = Nb + local_tanimotoArray[i][1];
    Nc = Nc + ((Na + Nb) >> 1);
  }
  tanimotoResultArray[global_id] = (float)Nc / (Na + Nb - Nc);
}
