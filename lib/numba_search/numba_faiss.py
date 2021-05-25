
# -- python imports --
import numpy as np
import heapq

# -- numba imports --
from numba import jit, cuda, float32

# -- pytorch imports --
import torch

def search_raw_array_numba(res,noise_level,db,query,K):
    db_numpy = db.numpy()
    query_numpy = query.numpy()
    D,I = cuda_search_raw_array_numba(noise_level,db_numpy,query_numpy,K)
    return torch.FloatTensor(D.T),torch.LongTensor(I.T) 


@cuda.jit
def cuda_l2_norm(values,norms):
    L,D = values.shape

    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    x = tx + bx * bw

    if x < L:
        for y in range(D):
            norms[x] += values[x,y]
        norms[x] = norms[x] * norms[x]

@jit
def cuda_search_raw_array_numba(noise_level,db,query,K):
    global cuda_l2_norm

    # -- collect db and query info --
    D,P = db.shape
    Q,P = query.shape

    # -- init outputs --
    outDist = cuda.device_array(shape=(D, K), dtype=float32)
    outIndex = cuda.device_array(shape=(D, K), dtype=int)

    # -- tile to reduce memory requirement --
    tileDB = D
    tileQ = Q // 8 + (Q%8 != 0)
    numDB = D // tileDB
    numQ = Q // tileQ

    # -- compute norms --
    dbNorms = cuda.device_array(shape=(D,), dtype=float32)
    cuda_l2_norm[1,D](db,dbNorms)
    qNorms = cuda.device_array(shape=(D,), dtype=float32)
    cuda_l2_norm[1,D](query,qNorms)

    # -- create buffers --
    # qBuf = cuda.device_array(shape=(tileQ, K), dtype=float32)
    # dbBuf = cuda.device_array(shape=(tileDB, K), dtype=float32)
    outDistBuf = cuda.device_array(shape=(tileQ, tileDB), dtype=float32)
    outIndexBuf = cuda.device_array(shape=(tileQ, tileDB), dtype=int)

    for tile_q in range(0,numQ,tileQ):
        q_size = min([tile_q,tileQ-tile_q])
        q_start = tile_q
        q_end = q_start + q_size

        queryView = query[q_start:q_end]
        queryNormView = qNorms[q_start:q_end]
        # qBufView = qBuf[q_start:q_end]
        outDistBufRowView = outDistBuf[q_start:q_end]
        outIndexBufRowView = outIndexBuf[q_start:q_end]
        
        for tile_db in range(0,numDB,tileDB):
            db_size = min([tile_db,tileDB-tile_db])
            db_start = tile_db
            db_end = db_start + db_size

            dbView = db[db_start:db_end]
            outDistBufColView = outDistBufRowView[:,db_start:db_end]
            outIndexBufColView = outIndexBufRowView[:,db_start:db_end]

            griddim = 32, 64
            blockdim = 1, 32
            fast_matmul[griddim,blockdim](queryView, dbView, outDistBufColView)
            


    return outDist,outIndex
    # D,I = wrapper_compute_l2_distances_python_v1(M,db.numpy(),query.numpy(),K)
    # return compute_l2_distances_numba(db,query,K)
    # return compute_l2_distances_numbaCuda(db,query,K)
    

def wrapper_compute_l2_distances_python_v1(M,db,query,K):

    # -- matrix mult --
    dbNorms = np.sum(db**2,axis=1)

    # -- l2 --
    delta = M.T - dbNorms
    # delta = compute_l2_distances_python_v1(db,query,K)

    # -- top K --
    indices = np.argsort(delta,axis=0).astype(np.int)[:K,:]
    values = np.take_along_axis(delta, indices, axis=0)

    # -- return top K datums --
    return values,indices


@jit(nopython=True)
def compute_l2_distances_python_v1(db,query,K):

    
    # -- matrix mult --
    dbNorms = np.sum(db**2,axis=1)
    M = query @ db.T
    
    # -- l2 --
    delta = M.T - dbNorms
    
    return delta

def compute_l2_distances_python(db,query,K):
    # -- init heap --
    H = []
    heapq.heapify(H)

    # -- collect db and query info --
    D,P = db.shape
    Q,P = query.shape
    
    # -- matrix mult --
    qNorms = np.linalg.norm(q,axis=1)
    M = query @ db.T
    
    heapq.heappush(H,(dist,index))

    # -- add elem to heap --
    heapq.heappush(H,(dist,index))

    # -- return top K datums --
    return [heapq.heappop(H) for i in range(K)]

@jit(nopython=True)
def compute_l2_distances_numba(db,query,K):
    pass    
    

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 8

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
