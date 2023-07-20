import os
import numpy as np
os.environ["PYOPENCL_CTX"] = "0"
import pyopencl as cl

def tanimotoSmilarityGpu(fgrpsGpu, molSizes, nfgrps):
    cant = int(((nfgrps-1)*nfgrps)/2)

    #combinations
    combinations = np.transpose(np.triu_indices(nfgrps, k=1))
    arreglo1 = (combinations[:, 0]*2048).astype("uint16")
    arreglo2 = (combinations[:, 1]*2048).astype("uint16")

    #results
    tanimotoResult = np.zeros(cant, dtype=np.float32)

    #pyopencl configuration
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    mf = cl.mem_flags

    tanimotoArray = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=fgrpsGpu)
    molSizesGPU = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=molSizes)
    combinationsArray1 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo1)
    combinationsArray2 = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=arreglo2)
    tanimotoResultArray = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=tanimotoResult)

    f = open('tanimotoSimilarity5.cl', 'r', encoding='utf-8')
    kernels = ''.join(f.readlines())
    f.close()

    prg = cl.Program(ctx, kernels).build()

    knl = prg.tanimoto_similarity(queue, (cant,), None, tanimotoArray, combinationsArray1, combinationsArray2, tanimotoResultArray, molSizesGPU)

    cl.enqueue_copy(queue, tanimotoResult, tanimotoResultArray).wait()

    queue.finish()
    tanimotoArray.release()
    combinationsArray1.release()
    combinationsArray2.release()
    tanimotoResultArray.release()

    return tanimotoResult, cant, arreglo1