from .grid import get_grid
from .sim import run_sim
from .plot import run_plot


def run_hb_gaussian():
    print("HI")
    grid,info = get_grid()
    sims = run_sim(grid,info)
    print(len(sims))
    fname = "bootstrap"
    run_plot(sims.bs,info,fname)
    fname = "frame_v_frame"
    run_plot(sims.fvf,info,fname)
    fname = "frame_v_mean"
    run_plot(sims.fvm,info,fname)

