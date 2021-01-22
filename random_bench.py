
import itertools
import numpy as np


def Rot_xy(theta, phi):
    return np.array([
        [np.cos(theta/2.), -1j*np.exp(-1j*phi)*np.sin(theta/2.)],
        [-1j*np.exp(1j*phi)*np.sin(theta/2.), np.cos(theta/2.)]
    ])

def Rot_z(phi):
    return np.array([
        [np.exp(1j*phi/2.), 0.],
        [0., np.exp(-1j*phi/2.)]
    ])


class clifford_1q:
    """The single-qubit Clifford group decomposed into elementary gates.
    R(theta, phi) * Z(phi_z)
    """

    _paras = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, 0.5),
        (0.0, 0.0, -0.5),
        (1.0, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 0.0, 0.0),
        (0.5, 1.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.5, -0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.5, 1.0, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.0, -0.5),
        (0.5, 1.0, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.0, 1),
        (0.5, 1.0, 1),
        (0.5, 0.5, 1),
        (0.5, -0.5, 1)]

    _mats = []
    for x in _paras:
        _mats = np.dot(Rot_xy(x[0], x[1]), Rot_z(x[2]))

    def para(self, idx):
        if int(idx) > 23:
            raise ValueError("idx larger than 23")
        return self._paras[idx]
    
    def mat(self, idx):
        x = self.para(idx)
        return np.dot(Rot_xy(x[0]*np.pi, x[1]*np.pi), Rot_z(x[2]*np.pi))




