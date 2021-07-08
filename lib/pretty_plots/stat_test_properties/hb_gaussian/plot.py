
# -- python --
import numpy as np
import pandas as pd
import numpy.random as npr
from easydict import EasyDict as edict
from patsy import dmatrices,ModelDesc
import pprint

# -- python plotting --
import matplotlib
import seaborn as sns
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from matplotlib import cm as plt_cm
from sklearn import linear_model
from sklearn import metrics

# -- project imports --
from pyutils import add_legend,create_meshgrid,np_log,create_list_pairs,create_named_meshgrid

# -- local imports --
from pretty_plots.stat_test_properties.misc import skip_with_endpoints,aggregate_field_pairs
from pretty_plots.stat_test_properties.plot import plot_single_sim_group,plot_sim_test_pairs,stratify_contour_plots,stratify_line_plots



def inspect_ratios():
    T1 = [3,10,20,3,10,20,3,10,45]
    T2 = [50,50,50,10,35,35,35,20,50]
    D = [100,100,100,100,100,100,100,100,100]
    R = [67250, 563, 33.61, 119.347, 152.299, 9.075, 18175, 16.78, 1.922]
    df = pd.DataFrame({'T1':T1,'T2':T2,'R':R,'D':D})
    # df = df[df['T1'] != 45]
    #df = df.drop(0)
    # df = df[(df['T1'] != 3)]

    df['R_pow_T1/T2'] = np.power(df['R'],df['T1']/df['T2'])
    df['T1/T2'] = df['T1']/df['T2']
    # df['exp(T1/T2)'] = np.exp(df['T1']/df['T2'])
    df['sqrt(T1/T2)'] = np.power(df['T1']/df['T2'],.5)
    df['log(R)'] = np.log(df['R'])
    df = df.sort_values('log(R)')
    

    # -- model log(R) --
    print("-="*25 + "-")
    print("Modeling log(R)")
    alpha = 10
    beta = 1.
    # X = np.power(df['T1/T2'].to_numpy()[:,None],1./alpha)
    # X = np.c_[X,np.power(1./df['T1/T2'].to_numpy()[:,None],1./alpha)]
    # X = np.c_[X,np.power(df['T1'],1/beta),np.power(df['T2'],1/beta)]
    X = np.c_[np.power(np.log(df['T1']),1./beta),np.power(np.log(df['T2']),1./beta)]

    y = df['log(R)'].to_numpy()
    print(X.shape,y.shape)
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    preds = reg.predict(X)
    df['preds'] = preds
    mse = metrics.mean_squared_error(y,preds)
    print("MSE ~= %2.3e" % mse)
    print(reg.coef_)
    print(reg.intercept_)
    
    print(df[['T1','T2','log(R)','preds']])

    # -- model log(R) --
    print("-="*25 + "-")
    print("Modeling R_pow_T1/T2")
    alpha = 2.
    X = np.power(df['T1/T2'].to_numpy()[:,None],1./alpha)
    X = np.c_[X,np.power(1./df['T1/T2'].to_numpy()[:,None],1./alpha)]
    X = np.c_[X,np.power(df['T1'],2),np.power(df['T2'],2)]
    # X = np.c_[np.power(df['T1'],1./alpha),np.power(df['T2'],1./alpha)]

    y = df['R_pow_T1/T2'].to_numpy()
    print(X.shape,y.shape)
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    preds = reg.predict(X)
    df['preds'] = preds
    mse = metrics.mean_squared_error(y,preds)
    print("MSE ~= %2.3e" % mse)
    print(reg.coef_)
    print(reg.intercept_)
    
    print(df[['T1','T2','R_pow_T1/T2','preds']])


def contour_plots(sims,info,fname):
    mlevels = None
    obs = "noisy"
    title = "HB Gaussian"
    xform = None
    label = "hb-gaussian"
    zinfo = edict({'mean':obs,'std':obs,'title':title,
                   'label':label,'xform':xform})
    fields,sfield = ["eps","std"],["T","D"]
    stratify_contour_plots(None,fields,sfield,sims,info,mlevels,title,fname,zinfo)

def guess_and_check_eq(sims,info):
    errors,obs = [],"noisy"
    for index,row in sims.iterrows():
        T = row['T']
        gt = row['std']**2 / T - row['eps']**2 / T
        error = np.mean((row[obs] - gt)**2)
        errors.append(error)
    sims['errors'] = errors
    sims['inv_T'] = 1/sims['T']
    sims['inv_D'] = 1/sims['D']
    sims['inv_T2'] = 1/sims['T']**2
    sims['inv_D2'] = 1/sims['D']**2
    sims['sqrt_T'] = 1/np.sqrt(sims['T'])
    sims['sqrt_D'] = 1/np.sqrt(sims['D'])
    sims['std2'] = sims['std']**2
    sims['eps2'] = sims['eps']**2
    sims['eps2_div_T'] = sims['eps2'] / (sims['T']*sims['D'])**2
    sims['std2_div_T'] = sims['std2'] / (sims['T']*sims['D'])**2
    sims['eps2_std2'] = sims['std2'] * sims['eps2'] / sims['T']**2
    sims['sum_eps2_std2_div_T'] = sims['eps2'] + sims['std2'] / sims['T']**2
    sims['exp_sum_eps2_std2_div_T'] = np.exp((sims['eps2_div_T'] + sims['std2_div_T']))
    sims['sum_std2_prod_eps2_std2_div_T'] = sims['std2'] * (sims['eps2_div_T'] + sims['std2_div_T'])
    sims['div_D_prod_eps2_std2_div_T'] = (sims['eps2_div_T'] + sims['std2_div_T']) / sims['D']
    sims['eps2_prod_eps2_std2_div_T'] = (sims['eps2_div_T'] + sims['std2_div_T']) * sims['eps2']
    sims['std2_prod_eps2_std2_div_T'] = (sims['eps2_div_T'] + sims['std2_div_T']) * sims['std2']
    # sims['exp_eps2_std2_div_T'] = np.exp( ( sims['eps2'] + sims['std2'] ) / sims['T']**2)
    # sims = sims[sims['std'] > 0]
    # sims = sims[sims['T'] > 3]
    # sims = sims[sims['eps'] > 0]
    # model_desc = "errors ~ std2_div_T * eps2_div_T * sum_eps2_std2_div_T * T * D - D - T"
    model_desc = "errors ~ eps2_div_T * std2_div_T * inv_T2 * inv_D2 * std2_prod_eps2_std2_div_T * eps2_prod_eps2_std2_div_T"
    # model_desc = "errors ~ eps2_div_T * std2_div_T * inv_T * inv_D * exp_sum_eps2_std2_div_T"
    y,X = dmatrices(model_desc,sims)
    desc = ModelDesc.from_formula(model_desc)
    print(desc.describe())
    reg = linear_model.LinearRegression()
    #reg = linear_model.Ridge(alpha=1000)
    # reg = linear_model.Lasso(alpha=.1,max_iter=10000)
    reg.fit(X,y)
    print(reg.coef_)
    print(reg.intercept_)
    
    b = reg.intercept_
    A = reg.coef_
    print(X.shape,A.shape)
    preds = reg.predict(X)
    # preds_2 = X @ A.T + b
    # print(np.mean((preds - preds_2)**2))
    mse = np.mean((y - preds)**2)
    pp = pprint.PrettyPrinter(indent=4)

    results = {}
    for v,k in zip(reg.coef_[0],str(desc.describe()).split(" + ")):
        name = str(int(np_log(abs(v))/np_log(10)))
        if not(name in results.keys()): results[name] = []
        results[name].append(k)
    pp.pprint(results)

    fixed = {"T":10,"D":100}
    fsims = sims
    for name,value in fixed.items():
        fsims = fsims[fsims[name] == value]
    noisy = fsims['noisy'].mean()
    std = fsims['std'].mean()
    eps = fsims['eps'].mean()
    # print(noisy,std,eps)
    print("MSE ~= %2.3e" % mse)


def run_plot(sims,info,fname):

    # inspect_ratios()
    contour_plots(sims,info,fname)
    # guess_and_check_eq(sims,info)

    # -- aggregate over fields --
    # eps = sims['eps'].to_numpy()
    # std = sims['std'].to_numpy()
    # fields = ['eps','std']
    # names = ['f1','f2']
    # grids = [sorted(np.unique(eps)),sorted(np.unique(std))]
    # fgrid = create_named_meshgrid(grids,names)
    # zinfo = edict({'mean':obs,'std':obs})
    # agg,xdata,ydata,gt = aggregate_field_pairs(sims,fgrid,grids,fields,zinfo)

    # T_grid = sims['T'].to_numpy()
    # D_grid = sims['D'].to_numpy()
    # std_grid = sims['std'].to_numpy()
    # eps_grid = sims['eps'].to_numpy()
    # est = sims['noisy'].to_numpy()
    # names = ['x%d' % (idx) for idx in range(len(xdata[0]))] + ['y']
    # xdata = np.array(xdata)
    # ydata = np.array(ydata)
    # print(xdata.shape,ydata.shape)
    # data = np.c_[xdata,ydata]
    # df = pd.DataFrame(data,columns = names)
    # y,X = dmatrices('y ~ x1 * x2 * x3 * x4',df)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #
    # -- Guessing the functional form --
    #
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



    # # -- create figure --
    # figsize = (8,4)
    # fig,axes = plt.subplots(figsize=figsize,ncols=2)

    # # -- save --
    # DIR = Path(f"./output/pretty_plots/stat_test_properties/{fname}/")
    # if not DIR.exists(): DIR.mkdir()
    # fn =  DIR / f"./{obs}_histogram_of_D.png"
    # plt.savefig(fn,transparent=True,bbox_inches='tight',dpi=300)
    # plt.close('all')
    # print(f"Wrote plot to [{fn}]")

