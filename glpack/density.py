import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Density:
    def __init__(self, L, W, dens):
        self.L = L
        self.W = W
        self.coords = []
        for x in range(L):
            for y in range(W):
                c = np.array([x, y])
                self.coords.append(c)
        self.coords = np.array(self.coords)
        self.dens = dens

    def plot(self, ax, scale=240, alpha=0.7, marker="s"):
        density=self.dens.reshape(self.L, self.W)
        # Plot density
        xes = []
        yes = []
        cs = []
        dmin = np.min(density)
        dmax = np.max(density)

        cmap = matplotlib.cm.get_cmap('gray_r')
        for x in range(self.L):
            for y in range(self.W):
                xes.append(x)
                yes.append(y)
                cval = (density[x, y] - dmin) / (dmax - dmin)
                c = cmap(cval)
                cs.append(c)
        ax.scatter(xes, yes, scale, c=cs, alpha=alpha, marker=marker)

    
    def average(self):
        avg = np.mean(self.dens.reshape(self.L, self.W), axis=1)
        return avg
