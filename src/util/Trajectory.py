# src/util/Trajectory.py
# This module defines the Trajectory class for handling robot trajectories.

import math
import numpy as np
import copy
from typing import List, Tuple
from util.Utility import MotionProfile, SystemIdParams, TrajParams #linspace_step_size

DEFAULT_FREQ_SPACING = 0.5 # [Hz] - default frequrncy range spacing

class Trajectory:
    def __init__(self, t_params: TrajParams, s_params: SystemIdParams):
        self.configuration = t_params.configuration
        self.max_displacement = t_params.max_displacement
        self.max_velocity = t_params.max_velocity
        self.max_acceleration = t_params.max_acceleration
        self.sysid_type = t_params.sysid_type
        self.single_traj_run_time = t_params.single_pt_run_time
        self.nV = s_params.nV
        self.nR = s_params.nR
        
        self.trajectoryTime = []
        self.perturbation = []
        self.endIndices = []
        self.positionTrajectory = []
        self.velocityTrajectory = []
        self.accelerationTrajectory = []