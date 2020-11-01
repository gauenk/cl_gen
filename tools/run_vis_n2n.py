

# python imports
import sys
sys.path.append("./lib/")

# pytorch imports

# project imports
from n2n.vis_filters import run_vis_filters_grid
from n2n.vis_rec import run_vis_rec_grid

def main():
    # run_vis_filters_grid()
    run_vis_rec_grid()    

if __name__ == "__main__":
    main()
