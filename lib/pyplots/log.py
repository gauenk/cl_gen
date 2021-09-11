
import numpy as np
from matplotlib.ticker import MultipleLocator, FixedLocator, FuncFormatter

def get_matplotlib_formatters():
    # set tickmarks at multiples of 1.
    majorLocator = MultipleLocator(1.)
    # create custom minor ticklabels at logarithmic positions
    ra = np.array([ [n+(1.-np.log10(i))]
                    for n in range(10,20)
                    for i in [2,3,4,5,6,7,8,9][::-1]]).flatten()*-1.
    minorLocator = FixedLocator(ra)
    ###### Formatter for Y-axis (chose any of the following two)
    # show labels as powers of 10 (looks ugly)
    majorFormatter= FuncFormatter(lambda x,p: "{:.1e}".format(10**x) ) 
    # or using MathText (looks nice, but not conform to the rest of the layout)
    majorFormatter= FuncFormatter(lambda x,p: r"$10^{"+"{x:d}".format(x=int(x))+r"}$" ) 
    return majorFormatter,majorLocator,minorLocator
    
