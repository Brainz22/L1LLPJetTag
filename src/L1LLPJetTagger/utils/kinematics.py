import numpy as np


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    if dphi < -np.pi:
        dphi += 2 * np.pi
    elif dphi > np.pi:
        dphi -= 2 * np.pi
    return dphi
