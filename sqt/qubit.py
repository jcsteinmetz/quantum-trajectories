"""
Contains the Qubit class.
"""

from typing import List, Union, Tuple
import numpy as np
from sqt.drive import Rabi, Measurement
from sqt.utils import pauli_x, pauli_y, pauli_z

class Qubit:
    """
    A single qubit in a cavity.
    """
    def __init__(self, bloch_vector: List[float]):
        if len(bloch_vector) != 3:
            raise ValueError("Invalid Bloch vector.")

        self.bloch_vector = bloch_vector
        self.density_matrix = 0.5*(np.eye(2)
                                   + pauli_x*self.bloch_vector[0]
                                   + pauli_y*self.bloch_vector[1]
                                   + pauli_z*self.bloch_vector[2])

        self.drives = []

    def __repr__(self) -> str:
        return f"Qubit(bloch_vector={self.bloch_vector})"

    def normalize(self) -> None:
        """
        Normalize the Bloch vector and density matrix, which is needed in case a trajectory
        escapes the Bloch sphere.
        """
        norm = np.linalg.norm(self.bloch_vector)
        if norm != 0:
            self.bloch_vector /= np.linalg.norm(self.bloch_vector)

        self.density_matrix = 0.5*(np.eye(2)
                                   + pauli_x*self.bloch_vector[0]
                                   + pauli_y*self.bloch_vector[1]
                                   + pauli_z*self.bloch_vector[2])

    def set_drives(self, drives: List[Union[Rabi, Measurement]]) -> None:
        """
        Applies a list of Rabi drives and measurement drives to the qubit.
        """
        # Apply all measurement drives before all Rabi drives
        self.drives = sorted(drives, key=lambda drive: isinstance(drive, Rabi))

    def trajectory(self,
                   time_start: float,
                   time_end: float,
                   dt: float,
                   stochastic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a stochastic quantum trajectory.
        """
        if time_start >= time_end:
            raise ValueError("Invalid start/end times.")

        if dt > (time_end - time_start):
            raise ValueError("Time step must be smaller than the total time.")

        n_meas = int((time_end - time_start)/dt)

        time = np.arange(time_start, time_end, dt)
        trajectory_data = np.zeros((3, n_meas))

        # initial values
        trajectory_data[:,0] = self.bloch_vector

        for step, t in enumerate(time[:-1]): # compute the actual trajectories
            self.evolve(t, dt, stochastic) # updates density_matrix and bloch_vector
            trajectory_data[:, step+1] = self.bloch_vector

        return time, trajectory_data

    def evolve(self,
               current_time: float,
               dt: float,
               stochastic: bool = True) -> None:
        """
        Evolves the qubit forward by one time step using a Bayesian update.
        """
        evolution_operator = np.eye(2)

        # measurement
        for drive in self.drives:
            if isinstance(drive, Measurement):
                drive_operator = drive.operator(self.density_matrix, dt, stochastic)
            else:
                drive_operator = drive.operator(current_time, dt)

            evolution_operator = drive_operator @ evolution_operator

        evolution_operator_dagger = evolution_operator.conj().T
        numerator = evolution_operator @ self.density_matrix @ evolution_operator_dagger

        self.density_matrix = numerator/np.trace(numerator)
        self.bloch_vector = [np.real(np.trace(pauli_x @ self.density_matrix)),
                             np.real(np.trace(pauli_y @ self.density_matrix)),
                             np.real(np.trace(pauli_z @ self.density_matrix))]

        if not 0 <= np.linalg.norm(self.bloch_vector) <= 1:
            self.normalize()


    # U = U - dt*0.5*((G1/4)*np.dot((pauli_x+1j*pauli_y),(pauli_x-1j*pauli_y))+(Gphi/2)*np.dot(pauli_z,pauli_z))#+dt*np.sqrt(eta)*(pauli_x*rx/(4*tau_x)+pauli_z*rz/(4*tau_z))

    # measurement inefficiency
    # numerator += (1-eta)*(dt/(4*drive.tau))*np.dot(pauli_z,np.dot(density_matrix,pauli_z))
    # numerator += (1-eta)*(dt/(4*drive.tau))*np.dot(pauli_x,np.dot(density_matrix,pauli_x))

    # relaxation and dephasing
    # numerator += dt*((G1/4)*np.dot((pauli_x-1j*pauli_y),np.dot(density_matrix,(pauli_x+1j*pauli_y)))+(Gphi/2)*np.dot(pauli_z,np.dot(density_matrix,pauli_z)))
    
    # numerator[0,0] = (1-G1*dt)*numerator[0,0]
    # numerator[0,1] = (1-Gphi*dt-G1*dt/2.0)*numerator[0,1]
    # numerator[1,0] = (1-Gphi*dt-G1*dt/2.0)*numerator[1,0]
    # numerator[1,1] = numerator[1,1]+G1*dt*numerator[0,0]

    # return normal(gaussian_mean*np.sqrt(eta),np.sqrt(tau/dt))