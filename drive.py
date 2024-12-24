import numpy as np
from typing import List
from utils import pauli_n

class Rabi:
    """
    Rabi drive, which rotates the qubit about a given axis.
    """
    def __init__(self, rotation_axis: List[float], rotation_angle: float, time_start: float, time_end: float):
        if time_start >= time_end:
            raise ValueError("Invalid start/end times.")

        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        self.rotation_angle = rotation_angle
        self.time_start = time_start
        self.time_end = time_end
        self.width = time_end - time_start

    def shape(self, time: np.ndarray) -> np.ndarray:
        """
        Sine squared pulse shape between time_start and time_end.
        """
        cutoff = (time >= self.time_start) & (time <= self.time_end)
        return cutoff * (self.rotation_angle * 2 / self.width) * np.sin((np.pi/self.width)*(time-np.pi/2))**2

    def operator(self, t: float, dt: float) -> np.ndarray:
        """
        Matrix operator to apply the drive to a density matrix.
        """
        return np.eye(2) - 1j*(self.shape(t)*dt/2) * pauli_n(self.rotation_axis)
        
class Measurement:
    """
    Measurement drive, which acts slowly compared to all other qubit evolution. Causes backaction, which kicks
    the qubit towards one of its eigenstates.
    """
    def __init__(self, measurement_axis: List[float], tau: float, efficiency: float = 1):
        self.measurement_axis = measurement_axis
        self.tau = tau
        self.efficiency = efficiency

    def operator(self, density_matrix: np.ndarray, dt: float, stochastic: bool = True) -> np.ndarray:
        """
        Matrix operator to apply the measurement backaction to the density matrix.
        """
        readout = self.readout(density_matrix, dt, stochastic)
        return np.eye(2) + (readout * dt / (2 * self.tau)) * pauli_n(self.measurement_axis)
    
    def readout(self, density_matrix: np.ndarray, dt: float, stochastic: bool = True) -> float:
        """
        Draws a single trajectory readout from overlapping Gaussian distributions.
        """
        # Current value of measured coord
        current_coord = np.real(np.trace(pauli_n(self.measurement_axis) @ density_matrix))

        # Draw from either P+ or P- using weights from the density matrix
        gaussian_mean = np.random.choice([1, -1], p=[(1+current_coord)/2, (1-current_coord)/2])

        if not stochastic:
            return np.random.normal(gaussian_mean, 0)
        else:
            return np.random.normal(gaussian_mean, np.sqrt(self.tau/dt))
    

# return np.cos(self.shape(t)*dt/2)*np.eye(2) - 1j*np.sin(self.shape(t)*dt/2)*pauli_n(self.rotation_axis)
# a = readout * np.sqrt(eta) * dt / (2.0 * self.tau)
# return np.cosh(a)*I+np.sinh(a)*sig_n