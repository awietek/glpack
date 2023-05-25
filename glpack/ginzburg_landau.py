import numpy as np
from scipy.optimize import minimize
from .density import Density
from .phases import Phases

def gradient(psis, L, W, yperiodic=False):
    assert psis.shape == (L, W)
    gr = np.zeros((2, L, W), dtype=complex)
    if yperiodic:
        gr[0,:] = np.gradient(psis, axis=0)
        yp = psis.take(range(1, W+1), axis=1, mode="wrap")
        ym = psis.take(range(-1, W-1), axis=1, mode="wrap")
        gr[1,:] = (yp - ym) / 2.0
    else:
        gr[0,:] = np.gradient(psis, axis=0) 
        gr[1,:] = np.gradient(psis, axis=1)
    return gr

def _ginzburg_landau_functional(L, W, psis, density, alpha, beta, mstar,
                                estar, B, yperiodic):
    # create vector potential
    # x and y coordinates
    xes = np.arange(-L//2 + 0.5, L//2 + 0.5, 1.)
    yes = np.arange(-W//2 + 0.5, W//2 + 0.5, 1.)
    
    # Vector potential in Landau gauge
    A = np.zeros((2, L, W), dtype=complex)
    A[1, :, :] = B * np.outer(xes, np.ones_like(yes))

    A_times_psi = np.zeros_like(A)
    A_times_psi[0] = np.multiply(A[0], psis)
    A_times_psi[1] = np.multiply(A[1], psis)

    # V1 (only mass gets density)
    alphas = -alpha * density.dens.reshape(L, W)
    mass_term = np.sum(alphas * np.abs(psis)**2)
    int_term = (beta / 2) * np.sum(np.abs(psis)**4)

    # # V2 (mass and intgets density)
    # alphas = -alpha * density.dens.reshape(L, W)
    # betas = beta * density.dens.reshape(L, W) / 2
    # mass_term = np.sum(alphas * np.abs(psis)**2)
    # int_term = np.sum(betas  * np.abs(psis)**4)   
    
    grad_term = (1 / (2*mstar)) * \
        np.sum(np.abs(-1j * gradient(psis, L, W, yperiodic=yperiodic)
                      - estar * A_times_psi)**2)
    return mass_term + int_term + grad_term

def ginzburg_landau_functional(phases, density, alpha, beta, mstar, estar, B):
    L = phases.L
    W = phases.W
    yperiodic = phases.yperiodic

    if phases.lattice == "direct":
        psis = phases.phases.reshape(L, W)
    else:
        raise ValueError("GL functional only for phases on direct lattice")

    return _ginzburg_landau_functional(phases.L, phases.W, psis, density, alpha, beta, mstar, estar, B, yperiodic)

    
def solve_ginzburg_landau(phases0, density, alpha, beta, mstar, estar, B,
                          verbose=True, method="L-BFGS-B", tol=1e-12,
                          maxfun=100000):
    L = phases0.L
    W = phases0.W
    yperiodic = phases0.yperiodic
    if B == 0 and phases0.real:
        real = True
    else:
        real = False
    
    
    def objective(psis_ri):
        if real:
            psis = psis_ri
        else:
            N = len(psis_ri)
            psis = psis_ri[:N//2] + 1j * psis_ri[N//2:]
        return _ginzburg_landau_functional(L, W, psis.reshape(L, W), density, alpha, beta, mstar, estar, B, yperiodic)

    _psis0 = np.copy(phases0.phases)
    if real:
        psis0 = _psis0
    else:
        psis0 = np.concatenate((np.real(_psis0), np.imag(_psis0)))
    
    res = minimize(objective, psis0, method=method, tol=tol,
                   options={"maxfun": maxfun})
    print("fun        :", res.fun)

    if verbose:
        print("fun        :", res.fun)
        print("jac (norm) :", np.linalg.norm(res.jac))
        print("message    :", res.message)
        print("nfev       :", res.nfev)
        print("success    :", res.success)
        
    X = []
    Y = []
    for x in range(L):
        for y in range(W):
            X.append(x)
            Y.append(y)
                
    X = np.array(X)
    Y = np.array(Y)
    if real:
        phs = res.x
    else:
        N = len(res.x)
        phs = res.x[:N//2] + 1j * res.x[N//2:]
        
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





def _ginzburg_landau_functional_ext(L, W, psis, density,
                                    alpha1, alpha2, alpha3,
                                    beta1, beta2, beta3,
                                    mstar, estar, B, yperiodic):
    # create vector potential
    # x and y coordinates
    xes = np.arange(-L//2 + 0.5, L//2 + 0.5, 1.)
    yes = np.arange(-W//2 + 0.5, W//2 + 0.5, 1.)
    
    # Vector potential in Landau gauge
    A = np.zeros((2, L, W), dtype=complex)
    A[1, :, :] = B * np.outer(xes, np.ones_like(yes))

    A_times_psi = np.zeros_like(A)
    A_times_psi[0] = np.multiply(A[0], psis)
    A_times_psi[1] = np.multiply(A[1], psis)

    # V1 (only mass gets density)
    # alphas = -alpha * density.dens.reshape(L, W)
    # mass_term = np.sum(alphas * np.abs(psis)**2)
    # int_term = (beta / 2) * np.sum(np.abs(psis)**4)

    # V2 (mass and intgets density)
    alphas = -alpha1 -alpha2 * density.dens.reshape(L, W) \
        - alpha3 * density.dens.reshape(L, W)**2
    betas = (beta1 + beta2 * density.dens.reshape(L, W) + \
             beta3 * density.dens.reshape(L, W)) / 2
    mass_term = np.sum(alphas * np.abs(psis)**2)
    int_term = np.sum(betas  * np.abs(psis)**4)   
    
    grad_term = (1 / (2*mstar)) * \
        np.sum(np.abs(-1j * gradient(psis, L, W, yperiodic=yperiodic)
                      - estar * A_times_psi)**2)
    return mass_term + int_term + grad_term


def ginzburg_landau_functional_ext(phases, density, alpha1, alpha2, alpha3,
                                   beta1, beta2, beta3, mstar, estar, B):
    L = phases.L
    W = phases.W
    yperiodic = phases.yperiodic

    if phases.lattice == "direct":
        psis = phases.phases.reshape(L, W)
    else:
        raise ValueError("GL functional only for phases on direct lattice")

    return _ginzburg_landau_functional_ext(phases.L, phases.W, psis, density,
                                           alpha1, alpha2, alpha3,
                                           beta1, beta2, beta3,
                                           mstar, estar, B, yperiodic)

    
def solve_ginzburg_landau_ext(phases0, density, alpha1, alpha2, alpha3,
                              beta1, beta2, beta3, mstar, estar, B,
                              verbose=True, method="L-BFGS-B", tol=1e-12,
                              maxfun=100000):
    L = phases0.L
    W = phases0.W
    yperiodic = phases0.yperiodic
    if B == 0 and phases0.real:
        real = True
    else:
        real = False
    
    
    def objective(psis_ri):
        if real:
            psis = psis_ri
        else:
            N = len(psis_ri)
            psis = psis_ri[:N//2] + 1j * psis_ri[N//2:]
        return _ginzburg_landau_functional_ext(L, W, psis.reshape(L, W),
                                               density, alpha1, alpha2,
                                               alpha3, beta1, beta2, beta3,
                                               mstar, estar, B, yperiodic)

    _psis0 = np.copy(phases0.phases)
    if real:
        psis0 = _psis0
    else:
        psis0 = np.concatenate((np.real(_psis0), np.imag(_psis0)))
    
    res = minimize(objective, psis0, method=method, tol=tol,
                   options={"maxfun": maxfun})
    print("fun        :", res.fun)

    if verbose:
        print("fun        :", res.fun)
        print("jac (norm) :", np.linalg.norm(res.jac))
        print("message    :", res.message)
        print("nfev       :", res.nfev)
        print("success    :", res.success)
        
    X = []
    Y = []
    for x in range(L):
        for y in range(W):
            X.append(x)
            Y.append(y)
                
    X = np.array(X)
    Y = np.array(Y)
    if real:
        phs = res.x
    else:
        N = len(res.x)
        phs = res.x[:N//2] + 1j * res.x[N//2:]
        
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
