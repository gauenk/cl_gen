"""
Monte Carlo estimation of the noise floor

"""

# python imports
import sys
sys.path.append("./lib")
import numpy as np
import numpy.random as npr
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

# project imports
import settings

def batch_job(N,S):
    est = []
    for n in range(N):
        stddev = npr.uniform(0,50./255.,1)[0]
        noise = npr.normal(0,stddev,S)
        est.append(np.mean(noise**2))
    return est

def main():
    D = 6*10**3
    N = 15
    S = (32,32)
    est = []
    pjobs = Parallel(n_jobs=8)(delayed(batch_job)(N,S) for i in range(D))
    est = np.concatenate(pjobs)
    mean = np.mean(est)
    std = np.std(est)
    print("Estimated Noise Average: {:2.3e}".format(mean)) # 1.290*10^-2
    print("Estimated Noise Stddev: {:2.3e}".format(std)) # 1.148*10^-2
    sns.distplot(est)
    plt.savefig(f"{settings.ROOT_PATH}/reports/estimate_noise_level.png")
                
if __name__ == "__main__":
    print("Estimating the noise level for denosing experiments.")
    main()
