# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import unittest
import faiss
import faiss_mod
import torch


def swig_ptr_from_FloatTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.float32
    return faiss.cast_integer_to_float_ptr(
        x.storage().data_ptr() + x.storage_offset() * 4)


def swig_ptr_from_LongTensor(x):
    assert x.is_contiguous()
    assert x.dtype == torch.int64, 'dtype=%s' % x.dtype
    return faiss.cast_integer_to_idx_t_ptr(
        x.storage().data_ptr() + x.storage_offset() * 8)


def search_index_pytorch(index, x, k, D=None, I=None):
    """call the search function of an index with pytorch tensor I/O (CPU
    and GPU supported)"""
    assert x.is_contiguous()
    n, d = x.size()
    assert d == index.d

    if D is None:
        D = torch.empty((n, k), dtype=torch.float32, device=x.device)
    else:
        assert D.size() == (n, k)

    if I is None:
        I = torch.empty((n, k), dtype=torch.int64, device=x.device)
    else:
        assert I.size() == (n, k)
    torch.cuda.synchronize()
    xptr = swig_ptr_from_FloatTensor(x)
    Iptr = swig_ptr_from_LongTensor(I)
    Dptr = swig_ptr_from_FloatTensor(D)
    index.search_c(n, xptr,
                   k, Dptr, Iptr)
    torch.cuda.synchronize()
    return D, I


def search_raw_array_pytorch(res, xb, xq, k, D=None, I=None,
                             metric=faiss.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)

    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)

    gpu_config = faiss.GpuDistanceParams()
    gpu_config.metric = metric
    gpu_config.k = k
    gpu_config.dims = d
    gpu_config.vectors = xb_ptr
    gpu_config.vectorsRowMajor = xb_row_major
    gpu_config.vectorType = faiss.DistanceDataType_F32
    gpu_config.numVectors = nb
    gpu_config.queries = xq_ptr
    gpu_config.queriesRowMajor = xq_row_major
    gpu_config.queryType = faiss.DistanceDataType_F32
    gpu_config.numQueries = nq
    gpu_config.outDistances = D_ptr
    gpu_config.outIndices = I_ptr
    gpu_config.outIndicesType = faiss.DistanceDataType_F32
    faiss.bfKnn(res, gpu_config)

    # faiss.bruteForceKnn(res, metric,
    #                     xb_ptr, xb_row_major, nb,
    #                     xq_ptr, xq_row_major, nq,
    #                     d, k, D_ptr, I_ptr)


    return D, I

def search_mod_raw_array_pytorch(res, noise_level, xb, xq, k, D=None, I=None,
                             metric=faiss_mod.METRIC_L2):
    assert xb.device == xq.device

    nq, d = xq.size()
    if xq.is_contiguous():
        xq_row_major = True
    elif xq.t().is_contiguous():
        xq = xq.t()    # I initially wrote xq:t(), Lua is still haunting me :-)
        xq_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')

    xq_ptr = swig_ptr_from_FloatTensor(xq)

    nb, d2 = xb.size()
    assert d2 == d
    if xb.is_contiguous():
        xb_row_major = True
    elif xb.t().is_contiguous():
        xb = xb.t()
        xb_row_major = False
    else:
        raise TypeError('matrix should be row or column-major')
    xb_ptr = swig_ptr_from_FloatTensor(xb)
    
    if D is None:
        D = torch.empty(nq, k, device=xb.device, dtype=torch.float32)
    else:
        assert D.shape == (nq, k)
        assert D.device == xb.device

    if I is None:
        I = torch.empty(nq, k, device=xb.device, dtype=torch.int64)
    else:
        assert I.shape == (nq, k)
        assert I.device == xb.device

    D_ptr = swig_ptr_from_FloatTensor(D)
    I_ptr = swig_ptr_from_LongTensor(I)
    # print("xb.means()",xb.mean(1).shape,xb.mean(1))
    # print("xq.means()",xq.mean(1).shape,xq.mean(1))
    # print("xb.stds()",xb.std(1).shape,xb.std(1)**2)
    # print("xq.stds()",xq.std(1).shape,xq.std(1)**2)
    # print("xb.norms().shape",xb.norm(dim=1,p=2))
    # print("xq.norms().shape",xq.norm(dim=1,p=2))
    # dist,ind = wasserstein_search(xb,xq,noise_level,k)
    # print("Test W Search")
    # print(dist,ind)

    gpu_config = faiss_mod.GpuDistanceParams()
    gpu_config.metric = metric
    gpu_config.k = k
    gpu_config.dims = d
    gpu_config.vectors = xb_ptr
    gpu_config.vectorsRowMajor = xb_row_major
    gpu_config.vectorType = faiss_mod.DistanceDataType_F32
    gpu_config.numVectors = nb
    gpu_config.queries = xq_ptr
    gpu_config.queriesRowMajor = xq_row_major
    gpu_config.queryType = faiss_mod.DistanceDataType_F32
    gpu_config.numQueries = nq
    gpu_config.outDistances = D_ptr
    gpu_config.outIndices = I_ptr
    gpu_config.outIndicesType = faiss_mod.DistanceDataType_F32
    gpu_config.ignoreOutDistances = False
    gpu_config.noise_level = 2*noise_level**2 / xb.shape[1]
    gpu_config.useWasserstein = True
    faiss_mod.bfKnn(res, gpu_config)

    # faiss_mod.bruteForceKnn(res, metric,
    #                     xb_ptr, xb_row_major, nb,
    #                     xq_ptr, xq_row_major, nq,
    #                     d, k, D_ptr, I_ptr)
    # print(D[:2],I[:2])


    return D, I

def wasserstein_search(database_g,query_g,noise_level,k):
    database = database_g.cpu().numpy()
    query = query_g.cpu().numpy()

    DB = database.shape[0]
    Q = 2

    dists = np.zeros((Q,k),dtype=np.float)
    indices = np.zeros((Q,k),dtype=np.int)

    mean_db = np.mean(database,axis=1)
    std_db = np.mean(database.T - mean_db,axis=0).T
    for q_index in range(Q): # only first 10 elements
        q = query[q_index]
        # mean_q = np.mean(q)
        # q_nmlz = q - mean_q
        # std_q = np.mean(q_nmlz)

        losses = np.zeros(DB)
        for db_index in range(DB):
            db = database[db_index]
            # db_nmlz = database[db_index] - mean_db[db_index]
            # m_loss = (mean_db[db_index] - mean_q)**2
            # s_loss = (std_db[db_index] - std_q - noise_level)**2
            # losses[db_index] = m_loss + s_loss
            res = q - db
            losses[db_index] = np.mean(res)**2 + (np.std(res)**2 - noise_level)**2
        args = np.argsort(losses)[:k]
        dists[q_index] = losses[args]
        indices[q_index] = args

    return dists,indices

class PytorchFaissInterop(unittest.TestCase):

    def test_interop(self):

        d = 16
        nq = 5
        nb = 20

        xq = faiss.randn(nq * d, 1234).reshape(nq, d)
        xb = faiss.randn(nb * d, 1235).reshape(nb, d)

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
        index.add(xb)

        # reference CPU result
        Dref, Iref = index.search(xq, 5)

        # query is pytorch tensor (CPU)
        xq_torch = torch.FloatTensor(xq)

        D2, I2 = search_index_pytorch(index, xq_torch, 5)

        assert np.all(Iref == I2.numpy())

        # query is pytorch tensor (GPU)
        xq_torch = xq_torch.cuda()
        # no need for a sync here

        D3, I3 = search_index_pytorch(index, xq_torch, 5)

        # D3 and I3 are on torch tensors on GPU as well.
        # this does a sync, which is useful because faiss and
        # pytorch use different Cuda streams.
        res.syncDefaultStreamCurrentDevice()

        assert np.all(Iref == I3.cpu().numpy())

    def test_raw_array_search(self):
        d = 32
        nb = 1024
        nq = 128
        k = 10

        # make GT on Faiss CPU

        xq = faiss.randn(nq * d, 1234).reshape(nq, d)
        xb = faiss.randn(nb * d, 1235).reshape(nb, d)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        gt_D, gt_I = index.search(xq, k)

        # resource object, can be re-used over calls
        res = faiss.StandardGpuResources()
        # put on same stream as pytorch to avoid synchronizing streams
        res.setDefaultNullStreamAllDevices()

        for xq_row_major in True, False:
            for xb_row_major in True, False:

                # move to pytorch & GPU
                xq_t = torch.from_numpy(xq).cuda()
                xb_t = torch.from_numpy(xb).cuda()

                if not xq_row_major:
                    xq_t = xq_t.t().clone().t()
                    assert not xq_t.is_contiguous()

                if not xb_row_major:
                    xb_t = xb_t.t().clone().t()
                    assert not xb_t.is_contiguous()

                D, I = search_raw_array_pytorch(res, xb_t, xq_t, k)

                # back to CPU for verification
                D = D.cpu().numpy()
                I = I.cpu().numpy()

                assert np.all(I == gt_I)
                assert np.all(np.abs(D - gt_D).max() < 1e-4)

                # test on subset
                try:
                    D, I = search_raw_array_pytorch(res, xb_t, xq_t[60:80], k)
                except TypeError:
                    if not xq_row_major:
                        # then it is expected
                        continue
                    # otherwise it is an error
                    raise

                # back to CPU for verification
                D = D.cpu().numpy()
                I = I.cpu().numpy()

                assert np.all(I == gt_I[60:80])
                assert np.all(np.abs(D - gt_D[60:80]).max() < 1e-4)
