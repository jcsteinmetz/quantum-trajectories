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
    axis = axis / np.linalg.norm(axis)
    pauli = np.zeros((2,2),dtype=complex)
    pauli[0,0] = axis[2]
    pauli[0,1] = axis[0]-1j*axis[1]
    pauli[1,0] = axis[0]+1j*axis[1]
    pauli[1,1] = -axis[2]
    return pauli