"""
Contains various types of controls that can be applied to the qubit.
"""
from typing import List
import numpy as np
from sqt.utils import pauli_n

class RabiDrive:
    """
    Rabi drive, which rotates the qubit about a given axis.
    """
    def __init__(self,
                 direction: List[float],
                 angle: float,
                 time_start: float,
                 time_end: float,
                 pulse_type: str = "sine squared"):

        if time_start >= time_end:
            raise ValueError("Invalid start/end times.")

        self.direction = direction / np.linalg.norm(direction)

        self.angle = angle
        self.time_start = time_start
        self.time_end = time_end
        self.width = time_end - time_start
        self.pulse_type = pulse_type

    def operator(self, time: float, dt: float) -> np.ndarray:
        """
        Matrix operator to apply the drive to a density matrix.
        """
        return np.eye(2) - dt * pauli_n(self.direction) * 1j * (self.pulse_shape(time)/2)

    def pulse_shape(self, time: float) -> float:
        """
        Method to select the type of pulse by name.

        Accepted names:
        "sine squared"
        """
        if self.pulse_type == "sine squared":
            return self.sine_squared(time)
        elif self.pulse_type == "oscillating":
            return self.oscillating(time)
        else:
            raise ValueError("Invalid pulse type.")
        
    def sine_squared(self, time: np.ndarray) -> np.ndarray:
        """
        Sine squared pulse shape between time_start and time_end.
        """
        cutoff = (time >= self.time_start) & (time <= self.time_end)
        amplitude = self.angle * 2 / self.width
        return cutoff * amplitude * np.sin((np.pi/self.width)*(time-np.pi/2))**2
    
    def oscillating(self, time: np.ndarray) -> np.ndarray:
        """
        Oscillating pulse shape between time_start and time_end.
        """
        cutoff = (time >= self.time_start) & (time <= self.time_end)
        amplitude = self.angle * 2 / self.width
        return cutoff * amplitude * np.sin((10*np.pi/self.width)*(time-np.pi/2))