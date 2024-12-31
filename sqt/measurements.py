from typing import List
import numpy as np
from sqt.utils import pauli_n

class WeakMeasurement:
    """
    Dispersive measurement, which acts slowly compared to all other qubit evolution. Causes
    backaction, which kicks the qubit towards one of its eigenstates.
    """
    def __init__(self,
                 direction: List[float],
                 tau: float,
                 efficiency: float = 1):

        if not 0 < efficiency <= 1:
            raise ValueError("Invalid measurement efficiency. Must be greater than 0 and up to 1.")

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