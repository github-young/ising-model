# coding: utf-8
import numpy as np
import math
import matplotlib.pyplot as pl

def plotData(data, fileName):
    axName = ["Energy","Entropy","Specific heat","Magnaetization"]
    colors = ['red', 'green', 'blue', 'brown']
    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=[8,5], dpi=300)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    ax = ax.flatten()
    fig.suptitle("Ising model with size=%dx%d, iteration=%d" %(size, size, iteration))
    ax[0].set_ylim([-8.2,0.1])
    ax[1].set_ylim([-0.05,0.75])
    ax[1].axhline(y=np.log(2), xmin=0.05, xmax=0.95, linestyle=':', color='red', label="ln2")
    for i in range(0, 4):
        ax[i].plot(data[0], data[i+1], '.', color=colors[i], markersize=2, label=axName[i])
        ax[i].axvline(x=2.0/np.log(1+np.sqrt(2)), ymin=0.05, ymax=0.95, color='black', linestyle='--', label='$T_c=%.3f$' %(2.0/np.log(1+np.sqrt(2))))
        ax[i].grid()
        ax[i].legend(loc='best', fontsize=10)
    # fig.show()
    figName = fileName+".pdf"
    fig.savefig(figName, bbox_inches='tight', pad_inches=0)
    pl.close()

def main():
    size = 100
    iteration = 1000000
    fileName = "ESCM-T-size"+str(size)+"-iter"+str(iteration)
    dataFileName = "data-"+fileName+".csv"
    data = np.loadtxt(dataFileName, delimiter=',')
    plotData(data, fileName)

if __name__ == "__main__":
    main()