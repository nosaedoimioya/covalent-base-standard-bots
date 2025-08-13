# src/util/RobotDynamics.py
# This module defines the RobotDynamics class for handling robot dynamics calculations.

import os
import numpy as np
import sympy as sym
import pickle

from typing import List
from util.urdf_to_dh.generate_dh import GenerateDhParams

from util.Utility import deg2rad, rotation_matrix_to_quaternion
class Dynamics():
    def __init__(self, robot_directory: str, urdf_file: str, initialize: bool = False, initial_angles: List = [], robot_object = None) -> None:
        '''
        Initialize the Dynamics class with the URDF file and optional initial angles.
        Args:
            urdf_file (str): Path to the URDF file.
            initialize (bool): Whether to initialize the robot model.
            initial_angles (List): Initial joint angles for the robot.
            robot_object: Optional robot object for additional functionality.
        '''
        # Initialize URDF location
        self.robot_module_dir = robot_directory
        self.urdf_file = urdf_file

        # Generate DH parameters from the URDF file
        if not os.path.exists(self.urdf_file):
            raise FileNotFoundError(f"URDF file not found at {self.urdf_file}")
        
        self.params_node = GenerateDhParams(self.urdf_file)
        self.robot_object = robot_object # The robot object for additional functionality
        self._know_params_node : bool = False 
        self.compute_params()

        self.num_joints = len(self.params_node.urdf_joints)

        # Check if the symbolic parameters are already computed

        # Symbolic functions - USE ONLY FOR CALIBRATION (not for real-time control)
        self.sym_q = sym.symbols(f'q0:{self.num_joints}')
        self.sym_p : list[sym.Matrix] = []                  # Link position vectors (static)
        self.sym_p_com : list[sym.Matrix] = []              # Center of mass position vectors (dynamic)
        self.sym_R : list[sym.Matrix] = []                  # Link rotation matrices (dynamic)
        self.sym_R_total = sym.eye(3)                       # Total rotation matrix (dynamic)
        self.sym_T : list[sym.Matrix] = []                  # Link transformation matrices (dynamic)
        self.sym_T_total = sym.eye(4)                       # Total transformation matrix (dynamic)
        self.sym_J = sym.zeros(6,self.num_joints)           # Jacobian matrix (dynamic)
        self.sym_Jw = sym.zeros(3,self.num_joints)          # Angular Jacobian (dynamic)
        self.sym_Jv = sym.zeros(3,self.num_joints)          # Linear Jacobian (dynamic)
        self.sym_M = sym.zeros(self.num_joints)             # Mass matrix (dynamic)

        # Joint positions (dynamic)
        self.current_joints : list[float] = []

        if initialize:
            if len(initial_angles) == 0:
                initial_angles = [0]*self.num_joints
                self.initialize_model()
                self.current_joints = initial_angles
            elif len(initial_angles) == self.num_joints:
                self.initialize_model()
                self.current_joints = initial_angles
            else:
                print(f"Warning: could not initialize robot model. Number of joints specified by URDF ({self.num_joints}) \
                       does not match the initial joint angles size ({len(initial_angles)}): {initial_angles}")

    def compute_params(self):
        if not self._know_params_node:
            self.params_node.parse_urdf()
            self.params_node.calculate_tfs_in_world_frame()
            self.params_node.calculate_params()
            self._know_params_node = True 

    def initialize_model(self) -> None:
        print(f"Initializing robot model from URDF file: {self.urdf_file}")
        # Always start clean before any recompute/caching path
        self.sym_R_total = sym.eye(3)
        self.sym_T_total = sym.eye(4)
        self.sym_J      = sym.zeros(6, self.num_joints)
        self.sym_Jw     = sym.zeros(3, self.num_joints)
        self.sym_Jv     = sym.zeros(3, self.num_joints)
        self.sym_M      = sym.zeros(self.num_joints, self.num_joints)

        # Compute position vectors for rotation and transformation matrices
        self.__calculate_position_vectors()

        # Save symbolic functions if not already saved
        sym_attributes = ["sym_R_total", "sym_p_com", "sym_T_total", "sym_J", "sym_M"]
        for attr in sym_attributes:
            cache_name = f"{self.robot_module_dir}/dynamics/{self.urdf_file.split('.')[-2].split('/')[-1]}_{attr}.pkl"
            if not os.path.isfile(f"{cache_name}"): # Model is not cached
                # Compute symbolic matrix
                getattr(self, f"_{attr}_compute")()
                # Save file in cache
                self.__cache_symbolic_matrix(cache_name=cache_name, symbolic_matrix=getattr(self, f"{attr}"))
                # Load the cached symbolic matrix
                self.__load_cached_model(cache_name=cache_name, attribute=attr)
            
            elif attr == "sym_R_total":
                # Load the cached symbolic rotation matrix
                self.__load_cached_model(cache_name=cache_name, attribute=attr)
                # Compute individual rotation matrices (lambdified)
                self.__sym_R_compute()
            
            elif attr == "sym_T_total":
                # Load the cached symbolic transformation matrix
                self.__load_cached_model(cache_name=cache_name, attribute=attr)
                # Compute individual transformation matrices (lambdified)
                self.__sym_T_compute()
            
            else:
                # Load the cached symbolic matrix
                self.__load_cached_model(cache_name=cache_name, attribute=attr)
    
    def get_mass_matrix(self, joint_angles: List) -> np.ndarray:
        if self._know_params_node == False:
            raise Exception("The robot parameters have not been initialized yet. \
                       Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        
        return self.sym_M(*joint_angles)
    
    def get_jacobian_matrix(self, joint_angles: List) -> np.ndarray:
        '''
            Returns a Jacobian matrix for a given list of joint angles
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                       Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        return self.sym_J(*joint_angles)
    
    def get_transformation_matrix(self, joint_angles: List) -> np.ndarray:
        '''
            Returns the transformation matrix for a given list of joint angles
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                       Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        
        return self.sym_T_total(*joint_angles)
    
    def get_individual_transformation_matrix(self, i: int, joint_angle: float) -> List[np.ndarray]:
        '''
            Returns the transformation matrix for a given joint index and list of joint angles
            Args:
                i: joint index
                joint_angle: joint angle for the specified joint
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                       Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        
        return self.sym_T[i](joint_angle)
    
    def get_rotation_matrix(self, joint_angles: List) -> np.ndarray:
        '''
            Returns the rotation matrix for a given list of joint angles
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                            Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        
        return self.sym_R_total(*joint_angles)
    
    def get_individual_rotation_matrix(self, i: int, joint_angle: float) -> np.ndarray:
        '''
            Returns the rotation matrix for a given joint index and list of joint angles
            Args:
                i: joint index
                joint_angle: joint angle for the specified joint
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                            Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        
        return self.sym_R[i](joint_angle)
    
    def get_position_vectors(self, joint_angles: List) -> List[np.ndarray]:
        '''
            Returns the position vectors for a given list of joint angles
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                       Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")
        
        return [p(*joint_angles) for p in self.sym_p]
    
    def get_forward_kinematics(self, joint_angles: List) -> tuple[np.ndarray, np.ndarray]:
        '''
            Returns the position and orientation of the end-effector in the world frame
            given a list of joint angles.
        '''
        if self._know_params_node == False:
            raise Exception("Warning: The robot parameters have not been initialized yet. \
                      Run Dynamics.initialize_model(joint_angles [rad]) to initialize model.")

        # Get the end-effector position and orientation
        position = self.sym_T_total(*joint_angles)[:3, 3]
        orientation = rotation_matrix_to_quaternion(rotation_mat=self.sym_R_total(*joint_angles))

        return position, orientation

    def __cache_symbolic_matrix(self, cache_name: str, symbolic_matrix: sym.Matrix):
        """
        __CACHE_SYMBOLIC_MATRIX saves the symbolic matrix so that 
        it doesn't need to be recomputed again.
        """
        print(f"Saving symbolic matrix to {cache_name}...")

        with open(f"{cache_name}", "wb") as f:
            pickle.dump(symbolic_matrix, f)
        
        print(f'{cache_name} saved.')

    def __load_cached_model(self, cache_name: str, attribute: str):
        """
        __LOAD_CACHED_MODEL loads the cached symbolic matrix from the file.
        """
        print(f"Loading cached model from {cache_name} for attribute {attribute}...")

        if attribute in ("sym_M"):
            print(f"Mass matrix takes a while to load, please wait...")
    
        with open(cache_name, "rb") as f:
            expr = pickle.load(f)

        # Keep symbolic (no lambdify) for structures you still differentiate or that are lists
        if attribute in ("sym_p_com",) or isinstance(expr, list):
            setattr(self, attribute, expr)
            return

        # Lambdify matrices/scalars only
        fn = sym.lambdify(self.sym_q, expr, modules='numpy')
        setattr(self, attribute, fn)

    def __calculate_position_vectors(self):
        # Compute link position vectors
        self.sym_p = [sym.Matrix(self.__position_vector(i)) for i in range(self.num_joints)]
    
    def __position_vector(self, i: int):
        # alpha and theta are usually saved in degrees
        alpha = deg2rad(self.params_node.dh_dict['alpha'][i])
        a = self.params_node.dh_dict['r'][i]
        d = self.params_node.dh_dict['d'][i]

        # Return the position vector in the form of [a, -sin(alpha)*d, cos(alpha)*d]
        return [a, -sym.sin(alpha)*d, sym.cos(alpha)*d]

    def _sym_R_total_compute(self):
        # Update rotation matrix list
        self.__sym_R_compute()
        Rt = sym.eye(3)
        for i in range(self.num_joints):
            Rt = Rt * self.sym_R[i]
        self.sym_R_total = Rt

    def __sym_R_compute(self):
        sym_R_matrices = [self.__sym_rotation_matrix(i, self.sym_q[i]) for i in range(self.num_joints)]

        self.sym_R = [
            sym.lambdify(self.sym_q[j], sym_R_matrices[j], modules='numpy')
            for j in range(self.num_joints)
        ]
    
    def __sym_rotation_matrix(self, i: int, q: sym.Symbol) -> np.ndarray:
        # alpha and theta saved in degrees
        alpha = np.deg2rad(self.params_node.dh_dict['alpha'][i])

        return sym.Matrix([
            [sym.cos(q), -sym.sin(q), 0],
            [sym.sin(q)*sym.cos(alpha), sym.cos(q)*sym.cos(alpha), -sym.sin(alpha)],
            [sym.sin(q)*sym.sin(alpha), sym.cos(q)*sym.sin(alpha), sym.cos(alpha)]
        ])
    
    def _sym_p_com_compute(self):
        # reset before append
        self.sym_p_com = []
        # Update center of mass position vectors
        current_com = self.sym_p[0] # Start with the initial base position
        current_rotation = self.sym_R[0] # Start with the first rotation matrix

        for i in range(self.num_joints):
            pc = sym.Matrix(self.params_node.inertia_dict['com'][i])
            if i > 0:
                current_com += current_rotation * self.sym_p[i]
                current_rotation = current_rotation * self.sym_R[i]
            self.sym_p_com.append(current_com + (current_rotation * pc))

    def _sym_T_total_compute(self):
        self.__sym_T_compute()
        Tt = sym.eye(4)
        for i in range(self.num_joints):
            Tt = Tt * self.sym_T[i]
        self.sym_T_total = Tt

    def __sym_T_compute(self):
        sym_T_matrices = [self.__sym_transformation_matrix(i,self.sym_q[i]) for i in range(self.num_joints)]

        self.sym_T = [
            sym.lambdify(self.sym_q[j], sym_T_matrices[j], modules='numpy')
            for j in range(self.num_joints)
        ]
    
    def __sym_transformation_matrix(self, i: int, q: sym.Symbol) -> np.ndarray:
        '''
        Returns the homogeneous transformation matrix for joint i
        given the joint positions using the DH parameters

        Args:
            i: joint number (column of joint angles)
            q: joint angle [radians]
        '''
        T_a = sym.eye(4)
        T_a[0,3] = self.params_node.dh_dict['r'][i]
        T_d = sym.eye(4)
        T_d[2,3] = self.params_node.dh_dict['d'][i]

        # alpha and theta saved in degrees
        alpha = np.deg2rad(self.params_node.dh_dict['alpha'][i])
        
        R_zt = sym.Matrix([[sym.cos(q),-sym.sin(q), 0,  0],
                           [sym.sin(q),sym.cos(q),  0,  0],
                           [0,        0,          1,  0],
                           [0,        0,          0,  1]])
        
        R_xa = sym.Matrix([[1, 0,            0,              0],
                           [0, sym.cos(alpha),-sym.sin(alpha), 0],
                           [0, sym.sin(alpha),sym.cos(alpha),  0],
                           [0, 0,            0,              1]])
        
        return T_d * R_zt * T_a * R_xa
    
    def _sym_J_compute(self):
        # reset
        self.sym_Jv = sym.zeros(3, self.num_joints)
        self.sym_Jw = sym.zeros(3, self.num_joints)
        q = sym.Matrix([[self.sym_q[i]] for i in range(self.num_joints)])
        
        # Linear part
        for i in range(self.num_joints):
            self.sym_Jv[:,i] = (self.sym_p_com[self.num_joints-1].jacobian(q))[:,i]
        
        # Angular part
        R_current = self.sym_R[0] # Start with the base rotation matrix
        Jo : sym.Matrix = self.sym_R[0][:,2] # Start the Jacobian with the first column

        for i in range(1,self.num_joints):
            # Compute cumulative rotation
            R_current = R_current * self.sym_R[i] 
            # Add current column to the Jacobian
            Jo = Jo.row_join(R_current[:, 2])
        
        self.sym_Jw = Jo

        self.sym_J = sym.Matrix.vstack(self.sym_Jv,self.sym_Jw)
    
    def _sym_M_compute(self):
        # reset
        self.sym_M = sym.zeros(self.num_joints, self.num_joints)

        # Compiled rotation matrix
        R_o = sym.eye(3)

        Jv_stack = sym.zeros(self.sym_Jv.shape[0], self.sym_Jv.shape[1])
        Jw_stack = sym.zeros(self.sym_Jw.shape[0], self.sym_Jw.shape[1])

        # print(f"self.Jv[:,[0]]: {self.Jv[:,[0]]}")
        for i in range(self.num_joints):
            # Stacking rotation matrix
            R_o = R_o * self.sym_R[i]

            # Stacking Jacobian matrices
            Jv_stack[:,i] = self.sym_Jv[:,i]
            Jw_stack[:,i] = self.sym_Jw[:,i]

            # Add to mass matrix
            self.sym_M += (Jv_stack.T * (self.params_node.inertia_dict['mass'][i] * sym.eye(3)) * Jv_stack) + \
                           Jw_stack.T * R_o * self.params_node.inertia_dict['inertia_tensor'][i] * R_o.T * Jw_stack
    