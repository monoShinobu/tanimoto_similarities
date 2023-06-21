import os, sys
import subprocess

groups = ["m61,m08,m17,m24,m60",
"m35,m36,m69,m70,m17,m24,m60,m05,m18,m19",
"m01,m02,m03,m37,m38,m39,m17,m24,m60,m05,m18,m19,m20,m21,m59",
"m31,m32,m33,m34,m65,m66,m67,m68,m17,m24,m60,m05,m18,m19,m20,m21,m22,m51,m58,m59",
"m04,m06,m07,m09,m10,m40,m41,m42,m43,m44,m17,m24,m60,m05,m18,m19,m20,m21,m22,m23,m51,m52,m53,m58,m59",
"m25,m26,m27,m28,m29,m30,m48,m49,m50,m62,m63,m64,m17,m24,m60,m05,m18,m19,m20,m21,m22,m23,m51,m52,m53,m71,m54,m55,m58,m59",
"m11,m12,m13,m14,m15,m25,m26,m27,m45,m46,m47,m48,m49,m50,m62,m63,m17,m24,m60,m05,m18,m19,m20,m21,m22,m23,m51,m52,m53,m71,m54,m55,m56,m58,m59",
"m61,m01,m02,m03,m04,m06,m07,m09,m10,m11,m27,m28,m29,m30,m31,m32,m33,m34,m35,m36,m08,m37,m38,m39,m40,m41,m42,m43,m44,m45,m50,m62,m63,m64,m65,m66,m67,m68,m69,m70"]

#number of the group
n = int(sys.argv[1]) - 1

molecules = groups[n].split(",")
l_pdb = [f"mols{os.sep}{mol}.pdb" for mol in molecules]
print(l_pdb)
for mol_pdb in l_pdb:
	subprocess.check_call(["cp",mol_pdb,"KL" + os.sep])
