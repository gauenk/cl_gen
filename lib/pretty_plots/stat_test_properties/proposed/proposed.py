

# -- [local] imports --
from .proposed_sim import get_proposed_sims
from .proposed_grid import create_proposed_parameter_grid
from .proposed_plot import plot_proposed_sims

def run_proposed():
    pgrid,lgrid = create_proposed_parameter_grid()
    plot_proposed_test(pgrid,lgrid)

def plot_proposed_test(pgrid,lgrid):
    sims = get_proposed_sims(pgrid,parallel=True)
    title = "Proposed Function"
    fname = "proposed"
    plot_proposed_sims(sims,lgrid,title,fname)

