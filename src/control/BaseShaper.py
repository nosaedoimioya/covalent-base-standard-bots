# src/control/InputShaper.py
# This module provides the InputShaper class for shaping input signals.

import numpy as np
from collections import deque
from typing import Tuple, List

class BaseShaper:
    def __init__(self, Ts: float):
        """
        Initialize the BaseShaper with the sampling time.
        The buffer is initialized with zeros.
        Parameters:
            Ts: sampling time [s]
        """
        self.Ts = Ts
        self.buffer = deque()       # holds past samples, newest at left
        self.M = 0                  # current filter length - 1

    def shape_sample(self, x_i: float, frf_params: np.ndarray) -> float:
        """
        Shape a single sample x_i given current dynamics (wn, zeta).
        Maintains internal buffer of past x's and re-computes
        the impulse vector I each step for OSA convolution.

        Parameters:
            x_i: input sample
            frf_params: array of natural frequencies and damping ratios

        Returns:
            x_shaped: shaped input sample
        """
        # Validate input parameters
        if not isinstance(frf_params, np.ndarray):
            raise ValueError("frf_params must be a numpy array")
        if frf_params.ndim != 2 or frf_params.shape[1] != 2:
            raise ValueError("frf_params must have shape (m, 2)")
        if frf_params.shape[0] == 0:
            raise ValueError("frf_params cannot be empty")
        
        # Compute new shaper and its length
        I, M_new = self.compute_zvd_shaper(params_array=frf_params)

        # If first call or filter just grew, seed buffer with x_i
        if not self.buffer or M_new != self.M:
            self.M = M_new
            # prefill with the same initial value so no discontinuity
            self.buffer = deque([x_i] * (self.M + 1), maxlen=self.M+1)
        else:
            # normal rolling: add newest, drop oldest automatically
            self.buffer.appendleft(x_i)

        # OSA convolution
        x_arr    = np.array(self.buffer)      # shape (M+1,)
        x_shaped = float(np.dot(I, x_arr)) 
        return x_shaped

    def shape_trajectory(self, x: np.ndarray, varying_params: List[np.ndarray]) -> np.ndarray:
        """
        Shape a full trajectory x (one axis), given varying dynamics (wn_i, zeta_i).

        Parameters:
            x: trajectory to shape
            varying_params: list of arrays of varying dynamics (wn_i, zeta_i) at each time step i

        Returns:
            x_shaped: shaped trajectory
        """
        # Check if x is a one-dimensional array
        if x.ndim != 1:
            raise ValueError("x must be a one-dimensional array")

        # Check if varying_params is a list of numpy arrays
        if not isinstance(varying_params, list):
            raise ValueError("varying_params must be a list of numpy arrays")
        
        # Check if varying_params has the same number of list elements as x
        if len(varying_params) != x.shape[0]:
            raise ValueError("varying_params must have the same number of list elements as x")
        
        # Check if varying_params has exactly two columns
        if varying_params[0].shape[1] != 2:
            raise ValueError("varying_params must have 2 columns")
        
        x_shaped = np.zeros_like(x)
        for i in range(x.shape[0]):
            x_shaped[i] = self.shape_sample(x[i], frf_params=varying_params[i])
        return x_shaped

    def compute_zvd_shaper(self, params_array: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        ZVD shaper created via convolution of multiple single-mode ZV filters.

        Parameters:
          params_array: array of natural frequencies and damping ratios

        Returns:
          I: 1D array of length M+1 with combined impulse gains
          M: filter order (max delay index)
        """
        # Validate params_array
        if not isinstance(params_array, np.ndarray):
            raise ValueError("params_array must be a numpy array")
        if params_array.ndim != 2 or params_array.shape[1] != 2:
            raise ValueError("params_array must have shape (m, 2)")
        if params_array.shape[0] == 0:
            raise ValueError("params_array cannot be empty")
        
        # Initialize with first mode
        I, _ = self.__compute_single_mode(wn=params_array[0, 0], zeta=params_array[0, 1])

        # Convolve the remaining modes
        for i in range(1, params_array.shape[0]):
            I_new, _ = self.__compute_single_mode(wn=params_array[i, 0], zeta=params_array[i, 1])
            I = np.convolve(I, I_new)

        M = len(I) - 1
        return I, M

    def __compute_single_mode(self, wn: float, zeta: float) -> Tuple[np.ndarray, int]:
        """
        Build a single-mode shaper impulse train

        Parameters:
          wn: natural frequency [rad/s]
          zeta: damping ratio (0 < zeta < 1)

        Returns:
          impulse: array of length (2*d+1) with impulse amplitudes
          d:  integer delay index
        """
        if wn <= 0:
            raise ValueError("Natural frequency must be positive")
        if zeta <= 0 or zeta >= 1:
            raise ValueError("Damping ratio must be between 0 and 1")
        
        wd = wn * np.sqrt(1 - zeta**2)
        k  = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
        Td = 2 * np.pi / wd                     # half-period of damped oscillation
        d  = int(round(0.5 * Td / self.Ts)) # samples

        norm = 1 + 2*k + k**2
        impulse   = np.zeros(2*d + 1)
        impulse[0]     = 1.0 / norm
        impulse[d]     = 2.0 * k / norm
        impulse[2*d]   = k**2 / norm

        return impulse, d
    
    