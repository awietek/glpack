import os
import numpy as np

from .density import Density
from .phases import Phases

def datadir(dataset="hubbard"):
    if dataset == "hubbard":
        datadir = "/Users/awietek/Research/Projects/hubbard.hfield/dmrg_data/"
    elif dataset == "tj":
        datadir = "/Users/awietek/Research/Projects/hubbard.hfield/dmrg_data_tj/"
    else:
        raise ValueError("Invalid dataset")
    return datadir

def read_density(L, W, tp, nholes, alpha, mlink, yperiodic=False, dataset="hubbard"):
    if yperiodic:
        filename = "dens_tp({:.1f})_Nx({})_Ny({})_nholes({})_alpha_({:.1f})_mlink({})_P.txt".format(tp, L, W, nholes, alpha, mlink)
    else:
        filename = "dens_tp({:.1f})_Nx({})_Ny({})_nholes({})_alpha_({:.1f})_mlink({})_O.txt".format(tp, L, W, nholes, alpha, mlink)

    edens = np.loadtxt(os.path.join(datadir(dataset), filename))
    hdens = 1 - edens
    return Density(L, W, hdens)


def read_phases(L, W, tp, nholes, alpha, mlink, neig, yperiodic=True, dataset="hubbard"):
    if yperiodic:
        filename = "SC_tp({:.1f})_Nx({})_Ny({})_nholes({})_alpha_({:.1f})_mlink({})_neig({})_P.txt".format(tp, L, W, nholes, alpha, mlink, neig)
    else:
        filename = "SC_tp({:.1f})_Nx({})_Ny({})_nholes({})_alpha_({:.1f})_mlink({})_neig({})_O.txt".format(tp, L, W, nholes, alpha, mlink, neig)

    phase_arr = np.loadtxt(os.path.join(datadir(dataset), filename))
    phase_arr[:,0] -= 1
    phase_arr[:,1] -= 1


    
    # delete "periodic" data
    if not yperiodic:
        phase_new = []
        for i in range(phase_arr.shape[0]):
            if np.abs(phase_arr[i, 1] - (W-0.5)) > 1e-12:
                phase_new.append(phase_arr[i, :])
        phase_arr = np.array(phase_new)

    if phase_arr.shape[1] == 5:
        phase_arr = phase_arr[:, :4]

    return Phases(L, W, phase_arr, lattice="dual", yperiodic=yperiodic)
