import numpy as np

pauli_x = np.zeros((2,2),dtype=complex)
pauli_x[0,1] = 1.0
pauli_x[1,0] = 1.0

pauli_y = np.zeros((2,2),dtype=complex)
pauli_y[0,1] = -1j
pauli_y[1,0] = 1j

pauli_z = np.zeros((2,2),dtype=complex)
pauli_z[0,0] = 1.0
pauli_z[1,1] = -1.0

def pauli_n(axis):
    """
    Pauli matrix along an arbitrary axis
    """
    axis = axis / np.linalg.norm(axis)
    return axis[0] * pauli_x + axis[1] * pauli_y + axis[2] * pauli_z
