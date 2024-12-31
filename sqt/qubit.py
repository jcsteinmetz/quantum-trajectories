"""
Contains the Qubit class.
"""

from typing import List, Union, Tuple
from copy import copy
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from sqt.controls import RabiDrive
from sqt.measurements import WeakMeasurement
from sqt.utils import pauli_x, pauli_y, pauli_z, pauli_n

sns.set_theme()

class Qubit:
    """
    A single qubit in a resonant cavity.
    """
    def __init__(self, initial_bloch_vector: List[float]):
        if len(initial_bloch_vector) != 3:
            raise ValueError("Invalid Bloch vector.")

        self.initial_bloch_vector = initial_bloch_vector
        self.current_bloch_vector = copy(self.initial_bloch_vector)
        self.current_density_matrix = 0.5*(np.eye(2)
                                   + pauli_x*self.current_bloch_vector[0]
                                   + pauli_y*self.current_bloch_vector[1]
                                   + pauli_z*self.current_bloch_vector[2])

        self.controls = []
        self.measurements = []

    def __repr__(self) -> str:
        return f"Qubit(bloch_vector={self.initial_bloch_vector})"

    def normalize(self) -> None:
        """
        Normalize the Bloch vector and density matrix, which is needed in case a trajectory
        escapes the Bloch sphere.
        """
        norm = np.linalg.norm(self.current_bloch_vector)
        if norm != 0:
            self.current_bloch_vector /= np.linalg.norm(self.current_bloch_vector)

        self.current_density_matrix = 0.5*(np.eye(2)
                                   + pauli_x*self.current_bloch_vector[0]
                                   + pauli_y*self.current_bloch_vector[1]
                                   + pauli_z*self.current_bloch_vector[2])

    def set_controls(self, controls: List[RabiDrive]) -> None:
        """
        Applies a list of control drives to the qubit.
        """
        self.controls = controls

    def set_measurements(self, measurements: List[WeakMeasurement]) -> None:
        """
        Applies a list of continuous measurements to the qubit.
        """
        self.measurements = measurements

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
        trajectory_data[:,0] = self.current_bloch_vector

        for step, t in enumerate(time[:-1]): # compute the actual trajectories
            self.evolve(t, dt, stochastic) # updates current_density_matrix and current_bloch_vector
            trajectory_data[:, step+1] = self.current_bloch_vector

        self.reset()

        return time, trajectory_data

    def evolve(self,
               current_time: float,
               dt: float,
               stochastic: bool = True) -> None:
        """
        Evolves the qubit forward by one time step using a Bayesian update.
        """
        total_operator = np.eye(2)

        # measurement
        for measurement in self.measurements:
            measurement_operator = measurement.operator(self.current_density_matrix, dt, stochastic)
            total_operator = measurement_operator @ total_operator

        for control in self.controls:
            control_operator = control.operator(current_time, dt)
            total_operator = control_operator @ total_operator

        total_operator_dagger = total_operator.conj().T
        numerator = total_operator @ self.current_density_matrix @ total_operator_dagger

        self.current_density_matrix = numerator/np.trace(numerator)
        self.current_bloch_vector = [np.real(np.trace(pauli_x @ self.current_density_matrix)),
                                     np.real(np.trace(pauli_y @ self.current_density_matrix)),
                                     np.real(np.trace(pauli_z @ self.current_density_matrix))]

        if not 0 <= np.linalg.norm(self.current_bloch_vector) <= 1:
            self.normalize()

    def reset(self):
        self.current_bloch_vector = copy(self.initial_bloch_vector)
        self.current_density_matrix = 0.5*(np.eye(2)
                                   + pauli_x*self.current_bloch_vector[0]
                                   + pauli_y*self.current_bloch_vector[1]
                                   + pauli_z*self.current_bloch_vector[2])

    def plot_trajectories(self,
                          n_trajectories: int,
                          time_start: float,
                          time_end: float,
                          dt: float,
                          stochastic: bool = True) -> Figure:

        n_meas = int((time_end - time_start)/dt)
        time = np.arange(time_start, time_end, dt)

        fig, axes = plt.subplots(3, 1, figsize=(8, 12))

        # Stochastic trajectories
        if stochastic:
            for traj in range(n_trajectories):
                _, trajectory_data = self.trajectory(time_start, time_end, dt, stochastic)

                axes[0].plot(time, trajectory_data[0])
                axes[1].plot(time, trajectory_data[1])
                axes[2].plot(time, trajectory_data[2])

        # Ensemble average
        _, ens_avg_data = self.trajectory(time_start, time_end, dt, False)
        axes[0].plot(time, ens_avg_data[0], 'k--')
        axes[1].plot(time, ens_avg_data[1], 'k--')
        axes[2].plot(time, ens_avg_data[2], 'k--')

        # Plot format
        axes[0].set_title("x")
        axes[1].set_title("y")
        axes[2].set_title("z")

        axes[0].set_ylim([-1.1, 1.1])
        axes[1].set_ylim([-1.1, 1.1])
        axes[2].set_ylim([-1.1, 1.1])

        return fig


    # U = U - dt*0.5*((G1/4)*np.dot((pauli_x+1j*pauli_y),(pauli_x-1j*pauli_y))+(Gphi/2)*np.dot(pauli_z,pauli_z))#+dt*np.sqrt(eta)*(pauli_x*rx/(4*tau_x)+pauli_z*rz/(4*tau_z))

    # measurement inefficiency
    # numerator += (1-eta)*(dt/(4*control.tau))*np.dot(pauli_z,np.dot(density_matrix,pauli_z))
    # numerator += (1-eta)*(dt/(4*control.tau))*np.dot(pauli_x,np.dot(density_matrix,pauli_x))

    # relaxation and dephasing
    # numerator += dt*((G1/4)*np.dot((pauli_x-1j*pauli_y),np.dot(density_matrix,(pauli_x+1j*pauli_y)))+(Gphi/2)*np.dot(pauli_z,np.dot(density_matrix,pauli_z)))

    # numerator[0,0] = (1-G1*dt)*numerator[0,0]
    # numerator[0,1] = (1-Gphi*dt-G1*dt/2.0)*numerator[0,1]
    # numerator[1,0] = (1-Gphi*dt-G1*dt/2.0)*numerator[1,0]
    # numerator[1,1] = numerator[1,1]+G1*dt*numerator[0,0]
