from libcpp.algorithm cimport sort, copy_n, sample
from libcpp.vector cimport vector
from libcpp.execution cimport par


def genweights(float[:,:] weights, int nframes):
   cdef 

def bootcpp(float[:,:,:] samples, long[:][:] subsets, float[:,:] scores_t, long[:] counts_t):

    // -- var defs --
    cdef int i,p,t,s;
    cdef int subsize;
    cdef int nimages,nsubsets,nframes,npix;
    cdef float loss;
    cdef vector[float] weights;

    // -- sizes (DIFFERENT THAN YOU THINK!) --
    nsubsets = len(subsets);
    nimages = len(samples); // == len(scores_t)
    npix = len(samples[0]);
    nframes = len(samples[0][0]); // == len(scores_t[0])

    // -- set sizes --
    weights.resize(nframes);

    #pragma acc kernels loop default(present)
    for( s = 0; s < nsubsets; s++) {

        for (t = 0; t < nframes; t++){
            weights[t] = -1./nframes;
        }

	subsize = len(subsets[s]);
	for (t = 0; t < subsize; t++){
	    weights[subsets[s][t]] = 1./subsize - 1./nframes;
	    counts_t[subsets[s][t]] += 1;
	}

        for (i = 0; i < nimages; i++){
	    loss = 0;
	    for (p = 0; p < npix; p++){
	        pix_ave = 0;
                for (t = 0; t < nframes; t++){
	            pix_ave += weights[t]*samples[i][p][t];
		}
		loss += pix_ave**2;
	    }
	    loss /= npix;
            for (t = 0; t<nframes; t++){
                scores_t[i][t] += loss;
            }
        }
   }
