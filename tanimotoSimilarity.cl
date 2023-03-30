
__kernel void tanimoto_similarity(__global int *tanimotoArray,
                                  __global int *combinationsArray,
                                  __global float *tanimotoResultArray) {

  int global_id = get_global_id(0);
  int i;
  int mol1 = combinationsArray[global_id * 2]*2048;
  int mol2 = combinationsArray[global_id * 2 + 1]*2048;
  int Na = 0;
  int Nb = 0;
  int Nc = 0;

  for (i = 0; i < 2048; i++) {
    Na = Na + tanimotoArray[mol1+i];
    Nb = Nb + tanimotoArray[mol2+i];
    if (tanimotoArray[mol1+i] == 1 && tanimotoArray[mol2+i] == 1) {
      Nc = Nc + 1;
    }
  }
  tanimotoResultArray[global_id] = (float)Nc / (Na + Nb - Nc);

  /*
    for (i = 0; i < 2048; i++) {
      if (tanimotoArray[i] == 1 && tanimotoArray[i] == 1) {

        Nc[0] = Nc[0] + 1;
        // Na[mol1] = Na[mol1] + 1;
        // Nb[mol2] = Nb[mol2] + 1;
      }else if (tanimotoArray[actMol1] == 1){

      Na[mol1] = Na[mol1] + 1;

    } else {

      Nb[mol2] = Nb[mol2] + 1;

    }
    }
    */
  /*
    if (tanimotoArray[actMol1] == 1) {
      Na[mol1] = Na[mol1] + 1;
    }

    if (tanimotoArray[actMol2] == 1) {
      Nb[mol2] = Nb[mol2] + 1;
    }
  */
}

/*
__kernel void tanimoto_similarity(__global int *tanimotoArray, __global int
*combinationsArray, __global int *Na, __global int *Nb, __global int *Nc) {

  int global_id = get_global_id(0);

  int comb = global_id / 2048;
  int mol1 = combinationsArray[comb * 2];
  int mol2 = combinationsArray[comb * 2 + 1];

  int actMol1 = mol1 * 2048 + (global_id % 2048);
  int actMol2 = mol2 * 2048 + (global_id % 2048);*/
/*
  printf("%d, %d, %d, %d, %d, %d", tanimotoArray[0], tanimotoArray[1],
  tanimotoArray[2], tanimotoArray[3], tanimotoArray[4], tanimotoArray[5]);

*//*
  if (comb == 0) {
    printf("%d ", tanimotoArray[actMol1]);
  }
*//*
  if (tanimotoArray[actMol1] == 1 && tanimotoArray[actMol2] == 1) {

    Nc[comb] = Nc[comb] + 1;
    //Na[mol1] = Na[mol1] + 1;
    //Nb[mol2] = Nb[mol2] + 1;

  } */
/*
else if (tanimotoArray[actMol1] == 1){

  Na[mol1] = Na[mol1] + 1;

} else {

  Nb[mol2] = Nb[mol2] + 1;

}
*/
/*
  if (tanimotoArray[actMol1] == 1) {
    Na[mol1] = Na[mol1] + 1;
  }

  if (tanimotoArray[actMol2] == 1) {
    Nb[mol2] = Nb[mol2] + 1;
  }
*/
