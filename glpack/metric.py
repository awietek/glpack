import numpy as np
from scipy.optimize import minimize_scalar


def metric_plain(phases1, phases2): 
    return np.linalg.norm(phases1.phases - phases2.phases)

def min_angle(phases1, phases2):
    def fun(theta):
        return metric_plain(phases1, np.exp(1j*theta) * phases2)
    res = minimize_scalar(fun, bounds=[0, 2*np.pi])
    return res.x

def metric(phases1, phases2):
    if phases1.real and phases2.real:
        m1 = metric_plain(phases1, phases2)
        m2 = metric_plain(phases1, -phases2)
        dist = min(m1, m2)
    else:
        theta = min_angle(phases1, phases2)
        dist = metric_plain(phases1, np.exp(1j*theta) * phases2) 
    return dist
