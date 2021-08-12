
# -- python imports --
import tqdm,os
import numpy as np
from numba import jit,njit
from pathlib import Path
from joblib import delayed,dump,load
from sklearn.metrics import pairwise_distances as pwdist


# -- project imports --
from pyutils.parallel import ProgressParallel
from pyutils.mesh_gen import create_indexing_mesh,apply_indexing_mesh,gen_indexing_mesh_levels,select_indexing_sizes,compute_indexing_ranges

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#          Main Test Function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def test_suite_indexing_mesh():
    # trial_suite_level1()
    # trial_suite_level2()
    trial_suite_generator()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         "Test Level 1" Suite
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def run_level1_indexing_mesh(f_ranges,sizes):
    prod_sizes = np.product(sizes)
    mesh = np.c_[[x.ravel() for x in np.meshgrid(*f_ranges)]].T
    print("Standard mesh: ",mesh.shape)
    l1_mesh,l1_ranges = create_indexing_mesh(f_ranges,sizes)
    print("[Sizes] L1 Mesh [%d] | Standard Mesh [%d]" % (l1_mesh.size,mesh.size))
    exists = np.zeros((mesh.shape[0]),dtype=np.int)
    nframes = len(f_ranges)
    for index,l1_elem in enumerate(l1_mesh):
        sub_mesh = apply_indexing_mesh(f_ranges,l1_mesh,sizes,index)
        assert sub_mesh.shape[0] <= prod_sizes, "We should have no more than prod(sizes) elems"
        for e in sub_mesh:
            argx = np.where(np.sum(np.abs(e[None,:] - mesh),axis=1) == 0)[0]
            exists[argx] += 1
    assert np.all(exists == 1), "The entire meshgrid is covered!"
    print("All meshgrid indices exist!")
            
def trial_level1_1():
    f_ranges = [
        [0,1,2,3,4,5,6],
        [3,4,5,6],
        [3,4],
        [0,2,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [3,4,5,6],
        [3,4],
        [0,2,5,6],
        [0,1,2,3,4,5,6],
    ]
    sizes = [ 2, 2, 1, 1, 1, 2, 2, 2, 2, 2 ]
    run_level1_indexing_mesh(f_ranges,sizes)

def trial_level1_2():
    f_ranges = [
        [0,1,2,3,4,5,6],
        [3,4,5,6],
        [3,4],
        [0,2,5,6],
        [0,1,2,3,4,5,6]
    ]
    sizes = [ 2, 2, 1, 1, 1 ]
    run_level1_indexing_mesh(f_ranges,sizes)

def trial_suite_level1():
    # trial_level1_1()
    trial_level1_2()
    
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#         "Test Level 2" Suite
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def run_level2_indexing_mesh(f_ranges,l1_K,l2_K):
    
    nframes = len(f_ranges)
    
    l0_sizes = np.ones(nframes,dtype=np.int)
    l0_ranges = f_ranges
    l1_sizes = select_indexing_sizes(l0_ranges,l0_sizes,l1_K)
    l1_ranges = compute_indexing_ranges(l0_ranges,l1_sizes)
    l2_sizes = select_indexing_sizes(l1_ranges,l1_sizes,l2_K)
    
    prod_l2_sizes = np.product(l2_sizes)
    
    l1_mesh,l1_ranges = create_indexing_mesh(f_ranges,l1_sizes)
    l2_mesh,l2_ranges = create_indexing_mesh(l1_ranges,l2_sizes)
    
    print("l1_mesh.shape[0] ", l1_mesh.shape[0])
    print("l2_mesh.shape[0] ", l2_mesh.shape[0])
    print("l1_ranges ", l1_ranges)
    print("l2_ranges ", l2_ranges)

    mesh = np.c_[[x.ravel() for x in np.meshgrid(*f_ranges)]].T
    exists = np.zeros((mesh.shape[0]),dtype=np.int)
    
    # -- parallel execution with mesh dump --
    l2_mesh = joblib_memmap(l2_mesh,"l2_mesh","r")
    mesh = joblib_memmap(mesh,"mesh","r")
    exists = np_memmap(exists,"exists","w+")
    
    pPar = ProgressParallel(True,l2_mesh.shape[0],n_jobs=8)
    results = pPar(delayed(compare_level2_parallel)(l2_mesh,mesh,exists,
                                                    l2_sizes,l1_ranges,l1_sizes,
                                                    f_ranges,l2_index)
                   for l2_index in range(l2_mesh.shape[0]))
    print(exists)
    assert np.all(exists == 1), "The entire meshgrid is covered!"
    print("All meshgrid indices exist!")

def compare_level2_parallel(l2_mesh,mesh,exists,l2_sizes,
                            l1_ranges,l1_sizes,f_ranges,l2_index):
    prod_l2_sizes = np.product(l2_sizes)
    sub_l1_mesh = apply_indexing_mesh(l1_ranges,l2_mesh,l2_sizes,l2_index)
    # print("sub_l1_mesh.shape[0] ",sub_l1_mesh.shape[0])
    msg = "We should have no more than prod(sizes) elems"
    assert sub_l1_mesh.shape[0] <= prod_l2_sizes, msg
    for sub_l1_index in range(sub_l1_mesh.shape[0]):
        sub_mesh = apply_indexing_mesh(f_ranges,sub_l1_mesh,l1_sizes,sub_l1_index)
        # print("sub_mesh.shape[0] ",sub_mesh.shape[0])
        dists = pwdist(sub_mesh,mesh)
        x,y = np.where(dists == 0)
        numba_sum_unique_index(exists,y)
    
def trial_level2_1():
    f_ranges = [
        [0,1,2,3,4,5,6],
        [3,4,5,6],
        [3,4],
        [0,2,5,6],
        [0,1,2,3,4,5,6],
        [0,1,2,3,4,5,6],
        [3,4,5,6],
        [3,4],
    ]
    l1_K,l2_K = 5,5
    run_level2_indexing_mesh(f_ranges,l1_K,l2_K)

def trial_level2_2():
    f_ranges = [
        [0,1,2,3,4,5,6],
        [3,4,5,6],
        [3,4],
        [0,2,5,6],
        [0,1,2,3,4,5,6]
    ]
    l1_K,l2_K = 3,3
    run_level2_indexing_mesh(f_ranges,l1_K,l2_K)

def trial_suite_level2():
    trial_level2_1()
    trial_level2_2()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#          "Test Generator" Suite
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        
def run_gen_indexing_mesh_levels(f_ranges,levels_K,levels_H):
    
    # -- create generator --
    generator = gen_indexing_mesh_levels(f_ranges,levels_K,levels_H)
    
    # -- setup testing --
    mesh = np.c_[[x.ravel() for x in np.meshgrid(*f_ranges)]].T
    exists = np.zeros((mesh.shape[0]),dtype=np.int)
    for batch in generator:
        dists = pwdist(batch,mesh)
        x,y = np.where(dists == 0)
        numba_sum_unique_index(exists,y)
        #argx = np.where(dists == 0)[0]
        #exists[argx] += 1
    print("exists")
    print(exists)
    assert np.all(exists == 1), "The entire meshgrid is covered!"
    print("All meshgrid indices exist!")   

    # pPar = ProgressParallel(True,l2_mesh.shape[0],n_jobs=8)
    # results = pPar(delayed(compare_level2_parallel)(l2_mesh,mesh,exists,
    #                                                 l2_sizes,l1_ranges,l1_sizes,
    #                                                 f_ranges,l2_index)
    #                for l2_index in range(l2_mesh.shape[0]))
    # print(exists)
    # assert np.all(exists == 1), "The entire meshgrid is covered!"
    # print("All meshgrid indices exist!")



def trial_generator_1():
    f_ranges = [
      [0,1,2,3,4,5,6],
      [3,4,5,6],
      [3,4],
      [0,2,5,6],
      [0,1,2,3,4,5,6],
      [0,1,2,3,4,5,6],
      [3,4,5,6],
      [3,4],
      [0,2,5,6],
      [0,1,2,3,4,5,6],
    ]
    levels_K = [2]
    levels_H = [2,]*len(levels_K)
    run_gen_indexing_mesh_levels(f_ranges,levels_K,levels_H)

def trial_generator_2():
    f_ranges = [
      [0,1,2,3,4,5,6],
      [3,4,5,6],
      [3,4],
      [0,2,5,6],
      [0,1,2,3,4,5,6],
      [0,1,2,3,4,5,6],
      [3,4,5,6],
      [3,4],
    ]
    levels_K = [5,5,5,3]
    levels_H = [2,]*len(levels_K)
    run_gen_indexing_mesh_levels(f_ranges,levels_K,levels_H)

def trial_generator_3():
    f_ranges = [
      [0,1,2,3,4,5,6],
      [3,4,5,6],
      [3,4],
      [0,2,5,6],
      [0,1,2,3,4,5,6],
      [0,1,2,3,4,5,6],
      [3,4,5,6],
      [3,4],
    ]
    levels_K = [5,4,3,3]
    levels_H = [2,]*len(levels_K)
    run_gen_indexing_mesh_levels(f_ranges,levels_K,levels_H)
        

def trial_suite_generator():
    trial_generator_1()
    trial_generator_2()
    trial_generator_3()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#              Misc.
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def np_memmap(data,fn,mode="w+"):
    folder = Path("./tests/np_memmap")
    if not folder.exists(): folder.mkdir()
    data_fn_mm = folder / fn
    dtype = data.dtype
    shape = data.shape
    data = np.memmap(data_fn_mm,dtype=dtype,
                     shape=shape,mode=mode)
    return data

def joblib_memmap(data,fn,mode="r"):
    folder = Path("./tests/joblib_memmap")
    if not folder.exists(): folder.mkdir()
    data_fn_mm = folder / fn
    dump(data,data_fn_mm)
    data = load(data_fn_mm, mmap_mode=mode)
    return data
    

@njit
def numba_sum_unique_index(ndarray,indices):
    for idx in range(len(indices)):
        ndarray[indices[idx]] += 1

