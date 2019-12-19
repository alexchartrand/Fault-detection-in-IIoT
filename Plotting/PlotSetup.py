import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
#=========================================================== # Directory and filename; style file open #===========================================================
# Load style file
plt.style.use('D:\\OneDrive\\Documents\\ETS\\Memoire\\IoT\\Plotting\\PaperDoubleFig.mplstyle')
# Make some style choices for plotting
colourWheel =['#329932',
            '#ff6961',
            'b',
            '#6a3d9a',
            '#fb9a99',
            '#e31a1c',
            '#fdbf6f',
            '#ff7f00',
            '#cab2d6',
            '#6a3d9a',
            '#ffff99',
            '#b15928',
            '#67001f',
            '#b2182b',
            '#d6604d',
            '#f4a582',
            '#fddbc7',
            '#f7f7f7',
            '#d1e5f0',
            '#92c5de',
            '#4393c3',
            '#2166ac',
            '#053061']
dashesStyles = [[3,1],
            [1000,1],
            [2,1,10,1],
            [4, 1, 1, 1, 1, 1]]

def setupAx(ax):
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

def plotExemple():
    fig, ax = plt.subplots()

    x = np.linspace(-2*np.pi, 2*np.pi,100)
    sin = np.sin(x)
    cos = np.cos(x)

    alphaVal = 0.6
    linethick = 3.5
    ax.plot(x,
            sin,
            color=colourWheel[0],
            linestyle='-',
            dashes=dashesStyles[0],
            lw=linethick,
            label='sin',
            alpha=alphaVal)

    ax.plot(x,
            cos,
            color=colourWheel[1],
            linestyle='-',
            dashes=dashesStyles[1],
            lw=linethick,
            label='sin',
            alpha=alphaVal)

    ax.set_xlabel('')
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_label_coords(0.63, 1.01)
    #ax.yaxis.tick_right()
    nameOfPlot = 'Sin vs Cos function'
    plt.ylabel(nameOfPlot, rotation=0)
    ax.legend(frameon=False, loc='upper left', ncol=1, handlelength=4)
    #plt.savefig(os.path.join(dirFile, 'ProdCountries.tiff'), dpi=300)
    plt.show()


if __name__ == "__main__":
    plotExemple()

