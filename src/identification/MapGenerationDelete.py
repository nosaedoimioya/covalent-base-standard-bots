# src/identification/MapGeneration.py
# This module is responsible for generating calibration maps.

import os
import pickle
import numpy as np

# Storred for backward compatibility

class CalibrationMap:
    def __init__(self, numPositions: int, axesCommanded: int, numJoints: int) -> None:
        # For storing calibration map raw values
        # =========================================================
        # List of all the numerators and denominators transfer functions for sine sweeps
        self.sineSweepNumerators = [[] for _ in range(numPositions)]
        self.sineSweepDenominators = [[] for _ in range(numPositions)]

        # Matrix of all the computed (median) natural frequencies at each position commanded
        self.allWn = np.zeros((numPositions, axesCommanded))

        # Matrix of all the computed (median) damping ratios at each position commanded
        self.allZeta = np.zeros((numPositions, axesCommanded))

        # Matrix of all the computed inertia matrices at each position commanded
        self.allInertia = np.zeros((numPositions, axesCommanded))

        # Vector of the computed (med.) natural frequencies that should be used for each of 
        # the joints when using an inverse dynamics controller
        self.jointWn = np.zeros((axesCommanded,))

        # Vector of the computed (med.) damping ratios that should be used for each of the
        # joints when using an inverse dynamics controller
        self.jointZeta = np.zeros((axesCommanded,))

        # Matrix of angular position at each position commanded (for neural network fitting)
        self.allV = np.zeros((numPositions, axesCommanded))

        # Matrix of radial position at each position commanded (for NN fitting)
        self.allR = np.zeros((numPositions, axesCommanded))

        # Matrix for storing initial joint positions
        self.initialPositions = np.zeros((numPositions, axesCommanded, numJoints))

        # For storing calibration map models (median natural frequency, damping ratio, and 
        # trained neural nets)
        # ===================================================================================
        # Vector of the median joint natural frequencies
        self.medianJointWn = np.zeros((axesCommanded,))

        # Vector of median joint damping ratios
        self.medianJointZeta = np.zeros((axesCommanded,))

        # Trained neural network parameters for natural frequency
        self.nnFunctionWn = [[]] * axesCommanded

        # Trained neural network parameters for damping ratio
        self.nnFunctionZeta = [[]] * axesCommanded
    
    def save_map(self, filename: str):
        """
        Save the class instance to a file using pickle.

        Parameters:
        filename (str): The name of the file to save the class instance to.
        """
        # Ensure the directory exists
        directory = os.path.dirname(filename)
        os.makedirs(directory, exist_ok=True)  # Create directories if they don't exist

        print(f"Saving {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load_map(self, filename: str) -> 'CalibrationMap':
        """
        Load the class instance from a file using pickle.

        Parameters:
        filename (str): The name of the file to load the class instance from.
        """
        print(f"Loading {filename}...")
        with open(filename, 'rb') as f:
            calibration_map_new = pickle.load(f)

        return calibration_map_new