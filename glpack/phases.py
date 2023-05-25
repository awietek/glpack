import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cmocean
import matplotlib
import cmocean.cm as cmo
import colorcet as cc

class Phases:
    def __init__(self, L, W, phase_arr, lattice="direct", yperiodic=False):
        self.L = L
        self.W = W
        self.yperiodic = yperiodic
        if lattice in ["direct", "dual"]:
            self.lattice = lattice
        else:
            raise ValueError("lattice needs to be \"direct\" or \"dual\"")
        
        if phase_arr.shape[1] == 3:
            self.real = True
        elif phase_arr.shape[1] == 4:
            self.real = False
        else:
            raise ValueError("Input phase array must be 3 or 4 columns.")

        if lattice == "direct":
            self.phases = []
            self.coords = []
            self.phases_x = None
            self.coords_x = None
            self.phases_y = None
            self.coords_y = None
            for x in range(L):
                for y in range(W):
                    c = np.array([x, y])
                    
                    ids = np.where(np.all(np.abs(phase_arr[:,:2] - c) < 1e-6, axis=1))
                    assert len(ids) == 1
                    id = ids[0]
                    if self.real:
                        self.phases.append(phase_arr[id, 2])
                    else: 
                        self.phases.append(phase_arr[id, 2] + 1j * \
                                           phase_arr[id, 3])
                    self.coords.append(c)
            self.phases = np.array(self.phases)
            self.coords = np.array(self.coords)
                    
        if lattice == "dual":
            self.phases = []
            self.coords = []
            self.phases_x = []
            self.coords_x = []
            self.phases_y = []
            self.coords_y = []
            for x in range(L):
                for y in range(W):

                    # phases along x direction
                    if x != L - 1:
                        cr = np.array([x+0.5, y])
                        ids = np.where(np.all(np.abs(phase_arr[:,:2] - cr) < 1e-6, axis=1))
                        assert len(ids) == 1
                        id = ids[0]
                        
                        if self.real:
                            phase = phase_arr[id, 2]
                        else:
                            phase = phase_arr[id, 2] + 1j * phase_arr[id, 3]

                        self.phases.append(phase)
                        self.coords.append(cr)
                        self.phases_x.append(phase)
                        self.coords_x.append(cr)
                        
                    # phases along y direction
                    if y != W - 1 or yperiodic:
                        ct = np.array([x, y+0.5])
                        ids = np.where(np.all(np.abs(phase_arr[:,:2] - ct) < 1e-6, axis=1))
                        assert len(ids) == 1
                        id = ids[0]
                        
                        if self.real:
                            phase = phase_arr[id, 2]
                        else:
                            phase = phase_arr[id, 2] + 1j * phase_arr[id, 3]
                        self.phases.append(phase)
                        self.coords.append(ct)
                        self.phases_y.append(phase)
                        self.coords_y.append(ct)
            self.phases = np.array(self.phases)
            self.coords = np.array(self.coords)
            self.phases_x = np.array(self.phases_x)
            self.coords_x = np.array(self.coords_x)
            self.phases_y = np.array(self.phases_y)
            self.coords_y = np.array(self.coords_y)

            
    def plot(self, ax, scale=0.5):
        """ plots the phases to an ax """
        
        # Plot phases
        X = self.coords[:,0]
        Y = self.coords[:,1]

        # real line plot
        if self.real:
            if self.lattice == "direct":
                sizes = np.abs(self.phases) * 1000
                vmin = np.min(self.phases)
                vmax = np.max(self.phases)
                cmap = cm.get_cmap("RdBu")
                cval = (self.phases - vmin) / (vmax - vmin)
                colors = cmap(cval)
                ax.scatter(self.coords[:,0], self.coords[:,1], s=sizes,
                           c=colors, zorder=1000)
                
            if self.lattice == "dual":
                maxabs = np.max(np.abs(self.phases))
                vmin = -maxabs
                vmax = maxabs
                cmap = cm.get_cmap("RdBu")

                for p, c in zip(self.phases_x, self.coords_x):
                    cval = (p - vmin) / (vmax - vmin)
                    color = cmap(cval)
                    lw = np.abs(cval - 0.5) * 8
                    ax.plot([c[0] - 0.5, c[0] + 0.5], [c[1], c[1]], c=color, lw=lw)
                
                
                for p, c in zip(self.phases_y, self.coords_y):
                    cval = (p - vmin) / (vmax - vmin)
                    color = cmap(cval)
                    lw = np.abs(cval - 0.5) * 8
                    ax.plot([c[0], c[0]], [c[1]-0.5, c[1]+0.5], c=color, lw=lw)
        
        # complex phase plot
        else:
            real = np.real(self.phases)
            imag = np.imag(self.phases)
            absv = np.abs(self.phases)

            phases_plot = np.zeros_like(real)

            for idx in range(len(real)):
                p = real[idx] + 1j*imag[idx]
                if np.abs((Y[idx] % 1.0) - 0.5) < 1e-12:
                    phases_plot[idx] = np.angle(p * np.exp(1j*np.pi))
                else:
                    phases_plot[idx] = np.angle(p)

            # cmap = "cmo.phase"
            # cmap = "cet_colorwheel"
            # cmap = "cet_cyclic_mrybm_35_75_c68_s25"
            cmap = matplotlib.cm.get_cmap("cet_cyclic_mrybm_35_75_c68_s25")
            colors = []
            for phase in phases_plot:
                cval = (phase + np.pi) / (2* np.pi)
                c = cmap(cval)
                colors.append(c)

            ax.quiver(X, Y, real, imag, scale=scale,
                      scale_units='inches', pivot="middle", color=colors,
                      zorder=1000, width=0.005)

    def normalize(self):
        self.phases /= np.linalg.norm(self.phases)
            
    def dual(self, yfac=-1.0):
        """ returns phases on dual lattice, which are average of neighboring sites """
        def index(x, y):
            return x*self.W + y

        X = []
        Y = []
        phs = []
        for x in range(self.L):
            for y in range(self.W):

                if self.lattice == "direct":
 
                    # X bond
                    if x != self.L-1:
                        X.append(x+0.5)
                        Y.append(y)
                        ph = (self.phases[index(x, y)] + self.phases[index(x+1, y)]) / 2
                        phs.append(ph)


                    # Y bond
                    if (y != self.W-1) or self.yperiodic:
                        X.append(x)
                        Y.append(y+0.5)
                        ph = (self.phases[index(x, y)] + self.phases[index(x, (y+1) % self.W)]) / 2
                        phs.append(yfac * ph)


                elif self.lattice == "dual":
                    pt = np.array([x, y])
                    closebys = []
                    for c, p in zip(self.coords, self.phases):
                        if np.linalg.norm(c - pt) < 0.51:
                            closebys.append((c, p))

                    xvals = []
                    yvals = []

                    for c, p in closebys:
                        if np.abs(pt[1] - c[1])<1e-12:
                            xvals.append(p)
                        elif np.abs(pt[0] - c[0])<1e-12:
                            yvals.append(p)

                    pp = (np.mean(xvals) + yfac*np.mean(yvals)) / 2
                    X.append(x)
                    Y.append(y)
                    phs.append(pp)

        X = np.array(X)
        Y = np.array(Y)
        phs = np.array(phs).flatten()
        if self.real:
            phase_arr = np.zeros((len(phs), 3))
            phase_arr[:,0] = X
            phase_arr[:,1] = Y
            phase_arr[:,2] = phs
        else:
            phase_arr = np.zeros((len(phs), 4))
            phase_arr[:,0] = X
            phase_arr[:,1] = Y
            phase_arr[:,2] = np.real(phs)
            phase_arr[:,3] = np.imag(phs)

        if self.lattice == "direct":
            return Phases(self.L, self.W, phase_arr, lattice="dual",
                          yperiodic=self.yperiodic)
        elif self.lattice == "dual":
            return Phases(self.L, self.W, phase_arr, lattice="direct",
                          yperiodic=self.yperiodic)


    def average(self, yfac=-1.0):
        
        if self.lattice == "dual":
            ps = self.dual(yfac).phases
        elif self.lattice == "direct":
            ps = self.phases


        avg = np.mean(ps.reshape(self.L, self.W), axis=1)
        return avg

    def __mul__(self, factor):
        if self.real:
            phase_arr = np.zeros((self.phases.shape[0], 3))
            phase_arr[:,:2] = self.coords
            phase_arr[:,2] = (self.phases * factor).flatten()
        else:
            phase_arr = np.zeros((self.phases.shape[0], 4))
            phase_arr[:,:2] = self.coords
            phs = (self.phases * factor).flatten()
            phase_arr[:,2] = np.real(phs)
            phase_arr[:,3] = np.imag(phs)

        return Phases(self.L, self.W, phase_arr, lattice=self.lattice,
                      yperiodic=self.yperiodic)

    __rmul__ = __mul__

    def __truediv__(self, factor):
        return (1 / factor) * self
        
    def __add__(self, other):
        if self.L != other.L or self.W != other.W or self.lattice != other.lattice\
           or self.yperiodic != other.yperiodic:
            raise ValueError("Unequal lattices")
        
        if self.real and other.real:
            phase_arr = np.zeros((self.phases.shape[0], 3))
            phase_arr[:,:2] = self.coords
            phase_arr[:,2] = (self.phases + other.phases).flatten()
        else:
            phase_arr = np.zeros((self.phases.shape[0], 4))
            phase_arr[:,:2] = self.coords
            phs = (self.phases + other.phases).flatten()
            phase_arr[:,2] = np.real(phs)
            phase_arr[:,3] = np.imag(phs)

        return Phases(self.L, self.W, phase_arr, lattice=self.lattice,
                      yperiodic=self.yperiodic)

    def __sub__(self, other):
        return self + (-1.0) * other

    def __neg__(self):
        return (-1.0) * self
            
def init_phases(L, W, k, yperiodic, real):
    X = []
    Y = []
    phs = []
    for x in range(L):
        for y in range(W):
            X.append(x)
            Y.append(y)
            
            if (k==0):
                phs.append(1.0)
            else:
                phs.append(np.sin(2 * np.pi *  k * x / L))

    X = np.array(X)
    Y = np.array(Y)
    phs = np.array(phs)
    if real:
        phase_arr = np.zeros((len(phs), 3))
        phase_arr[:,0] = X
        phase_arr[:,1] = Y
        phase_arr[:,2] = phs
    else:
        phase_arr = np.zeros((len(phs), 4))
        phase_arr[:,0] = X
        phase_arr[:,1] = Y
        phase_arr[:,2] = np.real(phs)
        phase_arr[:,3] = np.imag(phs)
    return Phases(L, W, phase_arr, lattice="direct", yperiodic=yperiodic)

