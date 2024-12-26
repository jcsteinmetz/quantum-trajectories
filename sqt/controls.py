"""
Contains various types of controls that can be applied to the qubit.
"""
from typing import List
from abc import ABC, abstractmethod
import numpy as np
from sqt.utils import pauli_n

class Control(ABC):
    """
    Abstract class for qubit controls
    """
    def __init__(self, direction):
        self.direction = direction

    @abstractmethod
    def operator(self, *args, **kwargs):
        """
        Matrix operator to apply the current control to the density matrix.
        """

class Rabi(Control):
    """
    Rabi drive, which rotates the qubit about a given axis.
    """
    def __init__(self,
                 direction: List[float],
                 angle: float,
                 time_start: float,
                 time_end: float):

        super().__init__(direction)

        if time_start >= time_end:
            raise ValueError("Invalid start/end times.")

        self.direction = direction / np.linalg.norm(direction)

        self.angle = angle
        self.time_start = time_start
        self.time_end = time_end
        self.width = time_end - time_start

    def shape(self, time: np.ndarray) -> np.ndarray:
        """
        Sine squared pulse shape between time_start and time_end.
        """
        cutoff = (time >= self.time_start) & (time <= self.time_end)
        amplitude = self.angle * 2 / self.width
        return cutoff * amplitude * np.sin((np.pi/self.width)*(time-np.pi/2))**2

    def operator(self, t: float, dt: float) -> np.ndarray:
        """
        Matrix operator to apply the drive to a density matrix.
        """
        return np.eye(2) - dt * pauli_n(self.direction) * 1j * (self.shape(t)/2)

class Measurement(Control):
    """
    Dispersive measurement, which acts slowly compared to all other qubit evolution. Causes
    backaction, which kicks the qubit towards one of its eigenstates.
    """
    def __init__(self,
                 direction: List[float],
                 tau: float,
                 efficiency: float = 1):

        super().__init__(direction)

        self.direction = direction
        self.tau = tau
        self.efficiency = efficiency

    def operator(self,
                 density_matrix: np.ndarray,
                 dt: float,
                 stochastic: bool = True) -> np.ndarray:
        """
        Matrix operator to apply the measurement backaction to the density matrix.
        """
        readout = self.readout(density_matrix, dt, stochastic)
        return np.eye(2) + dt * pauli_n(self.direction) * (readout / (2 * self.tau))

    def readout(self,
                density_matrix: np.ndarray,
                dt: float,
                stochastic: bool = True) -> float:
        """
        Draws a single trajectory readout from overlapping Gaussian distributions.
        """

        # Current value of measured coord
        current_coord = np.real(np.trace(pauli_n(self.direction) @ density_matrix))

        if not stochastic:
            return current_coord
        else:
             # Draw from either P+ or P- using weights from the density matrix
            gaussian_mean = np.random.choice([1, -1], p=[(1+current_coord)/2, (1-current_coord)/2])
            return np.random.normal(gaussian_mean, np.sqrt(self.tau/(dt*self.efficiency)))


# return np.cos(self.shape(t)*dt/2)*np.eye(2) - 1j*np.sin(self.shape(t)*dt/2)*pauli_n(self.direction)
# a = readout * np.sqrt(eta) * dt / (2.0 * self.tau)
# return np.cosh(a)*I+np.sinh(a)*sig_n
