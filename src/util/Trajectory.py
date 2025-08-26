# src/util/Trajectory.py
# This module defines the Trajectory class for handling robot trajectories.

import numpy as np
import copy
from typing import List, Tuple
from util.Utility import MotionProfile, SystemIdParams, TrajParams, linspace_step_size

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

    def generate_sine_sweep_trajectory(self, start_position: List[float], move_axis: int, Ts: float, \
                                       freq_range: np.ndarray = np.arange(1,20.5,DEFAULT_FREQ_SPACING), \
                                        settling_time: float = 1.0) -> None:
        # Generate sine perturbation
        sineProfile = self.sine_sweep(Ts, start_position, move_axis, freq_range, settling_time)

        self.__store_trajectory_data(sineProfile, start_position, move_axis, Ts)
        self._generated_sine_sweep_trajectory = True

    def sine_sweep(self, Ts: float, initial_position: List[float], move_axis: int, freq_range: np.ndarray, settling_time: float) -> MotionProfile:
        fs = 1/Ts
        sin_amp_vector = []
        q = np.array([])
        t = np.array([])
        freq_vec = np.ones(np.ceil(settling_time*fs) + 1)*freq_range[0]

        q0 = initial_position[move_axis]

        for freq in freq_range:
            sin_frequency = freq
            sin_amplitude = min(2*np.pi*sin_frequency*np.sqrt(self.max_displacement), self.max_acceleration)

            sin_amp_vector.append(sin_amplitude)
            q_sin, t_sin = self.table_traj_gen_function(sin_amplitude,sin_frequency,Ts)
            
            if len(q)>0:
               q = np.append(q, q_sin + q[-1])
            else:
                q = np.append(q, q_sin)

            t = np.append(t, t_sin + t[-1]) if len(t) > 0 else np.append(t, t_sin)

            q_dwell = np.ones(np.ceil(settling_time*fs))*q[-1]
            q = np.append(q,q_dwell)
            freq_temp = np.ones(len(q_sin)+len(q_dwell))*sin_frequency
            freq_vec = np.append(freq_vec,freq_temp)
            t_dwell = np.arange(0,settling_time,Ts)
            t = np.append(t,t_dwell + t[-1])
        
        q = q + q0
         
        # return
        return MotionProfile(t, q, [len(q)])

    def table_traj_gen_function(self, A: float, freq: float, Ts: float) -> Tuple[np.ndarray, np.ndarray]:
        p = 5 # cycles
        q_temp_end = 0

        # ramp up and ramp down point-to-point motion
        p2p_acc = 0.5 # [rad/s^2]
        p2p_vel = 0.05 # [rad/s]

        t_temp = []
        for j in range(0,np.ceil(1/freq*p/Ts)):
            t_temp.append(j*Ts)

        omega = 2*np.pi*freq
        t_temp = np.array(t_temp)
        v_temp = -(A/(2*np.pi*freq)) * np.cos(omega*t_temp)
        q_temp = -(A/np.pow(omega,2)) * np.sin(omega*t_temp)

        ramp_up_steps = range(0,np.ceil(abs(v_temp[0]/0.5)/Ts))
        ramp_down_steps = range(0,np.ceil(abs(v_temp[-1]/0.5)/Ts))
        q_ramp_up = np.zeros(ramp_up_steps.stop)
        q_ramp_down = np.zeros(ramp_down_steps.stop)
        for k in ramp_up_steps:
            q_ramp_up[k] = np.sign(v_temp[0])*0.5*0.5*np.pow(k*Ts,2) # [rad]

        for k in ramp_down_steps:
            q_ramp_down[k] = -np.sign(v_temp[-1])*0.5*0.5*np.pow(k*Ts,2) + v_temp[-1]*(k*Ts) # [rad]

        q_ramp_up = q_ramp_up - q_ramp_up[-1]
        q_ramp_down = q_ramp_down + q_temp[-1]

        q_ramp_start = q_ramp_up[0] # [rad]
        q_return = []
        return_time = np.arange(0, abs((q_ramp_start - q_temp_end))/(p2p_vel) + 0.2, Ts)
        for t in return_time:
            q_return_temp, _ = self.point_to_point_motion_jerk_limit(p2p_acc,q_temp_end,q_ramp_start,p2p_vel,t,Ts) #rad
            q_return.append(q_return_temp)

        q_ramp_up = np.concatenate((np.asarray(q_return),q_ramp_up)) # [rad]
        q_temp = np.concatenate((q_ramp_up[0:-1],q_temp))
        q_temp = np.append(q_temp,q_ramp_down[1:])
        q_temp_end = q_temp[-1]
        del q_ramp_up, q_ramp_down

        q_sin = q_temp

        q_return = []
        return_time = np.arange(0, abs((q_ramp_start - q_temp_end))/(p2p_vel) + 0.2, Ts)
        for t in return_time:
            q_return_temp, _ = self.point_to_point_motion_jerk_limit(p2p_acc,q_temp_end,0,p2p_vel,t,Ts) #rad
            q_return.append(q_return_temp)

        q_sin = np.append(q_sin,np.asarray(q_return))
        t_sin = np.array(linspace_step_size(0,((q_sin.shape[0])-1)*Ts,Ts))

        return q_sin, t_sin

    def point_to_point_motion_jerk_limit(self, acc: float, start_point: float, end_point: float, scan_v: float, t: float, Ts: float) -> Tuple[float, int]:
        dt = Ts
        j_limit = 10^3 # jerk limit [rad/s^3]
        t_dwell = 0.1 # dwell time [s]
        
        end_flag = 0

        dist = abs(start_point - end_point) # distance between two points [rad]
        dir = np.sign(end_point - start_point) # direction

        tj = acc/j_limit # jerk limited acceleratio time [s]
        ta = scan_v/acc + tj # acceleration time [s]

        la = (1/6)*j_limit*np.pow(tj,3) + (1/2)*acc*np.pow((ta-2*tj),2) + (1/2)*j_limit*np.pow(tj,2)*(ta-2*tj) + (1/6)*-j_limit*np.pow(tj,3) + (1/2)*acc*np.pow(tj,2) + (acc*(ta-2*tj) + (1/2)*j_limit*np.pow(tj,2))*tj

        td = ta
        ld = la

        tb = (dist-la-ld) / scan_v
        if tb < 0:
            tb = 0 # in case scan_v cannot be reached because of too short stroke
        
        lb = tb*scan_v # travel length for constant velocity [s]
        scan_v_p = scan_v # new scan velocity

        acc_p = acc

        if tb == 0:
            ta = np.sqrt(dist/acc) # spend half the time accelerating
            na = np.ceil(ta/dt) # num of acceleration steps
            ta = na*dt # new acceleration time [s]
            acc_p = dist/(np.pow(ta,2)) # new acceleration [rad/s^2]

            td = np.sqrt(dist/acc)
            nd = np.ceil(td/dt)
            td = nd*dt

        tP = (t_dwell+ta+tb+td+t_dwell) # motion period
        if tb > 0:
            T = np.zeros(9)
            T[0] = t_dwell
            T[1] = t_dwell + tj
            T[2] = t_dwell + ta - tj
            T[3] = t_dwell + ta
            T[4] = t_dwell + ta + tb
            T[5] = t_dwell + ta + tb + tj 
            T[6] = t_dwell + ta + tb + td - tj 
            T[7] = t_dwell + ta + tb + td 
            T[8] = t_dwell + ta + tb + td + t_dwell

            q_temp = 0
            if t <= T[0]: # dwell
                q_temp = start_point
            elif t > T[0] and t <= T[1]: # initial acceleration with limited jerk
                q_temp = start_point + (1/6)*dir*j_limit*np.pow((t-T[0]),3)
            elif t > T[1] and t <= T[2]: # constant accel.
                q_temp = start_point + (1/6)*dir*j_limit*np.pow((T[1]-T[0]),3) + (1/2)*dir*acc*np.pow(t-T[1],2) + (1/2)*dir*j_limit*np.pow(T[1]-T[0],2)*(t-T[1])
            elif t > T[2] and t <= T[3]: # reduce accel. w/ limited jerk
                q_temp = start_point + (1/6)*dir*j_limit*np.pow((T[1]-T[0]),3) + (1/2)*dir*acc*np.pow(T[2]-T[1],2) + (1/2)*dir*j_limit*np.pow(T[1]-T[0],2)*(T[2]-T[1]) + \
                 (-1/6)*dir*j_limit*np.pow(t-T[2],3) + (1/2)*dir*acc*np.pow(t-T[2],2) + (dir*acc*(T[2]-T[1]) + (1/2)*dir*j_limit*np.pow((T[1]-T[0]),2))*(T[3]-T[2]) + \
                    dir*scan_v_p*(t-T[3])
            elif t > T[4] and t <= T[5]: # deceleration with limited jerk
                q_temp = start_point + dir*(la + lb) + (-1/6)*dir*j_limit*np.pow(t-T[4],3) + dir*scan_v_p*(t-T[4])
            elif t > T[5] and t <= T[6]: # deceleration with constant deceleration
                q_temp = start_point + dir*(la + lb) + (-1/6)*dir*j_limit*np.pow(T[5]-T[4],3) + dir*scan_v_p*(T[5]-T[4]) + \
                    (-1/2)*dir*acc*np.pow(t-T[5],2) + ((-1/2)*dir*j_limit*np.pow(T[5]-T[4],2) + dir*scan_v_p)*(t-T[5])
            elif t > T[6] and t <= T[7]: # reduce deceleration with limited jerk
                q_temp = start_point + dir*(la + lb) + (-1/6)*dir*j_limit*np.pow(T[5]-T[4],3) + dir*scan_v_p*(T[5]-T[4]) + \
                    (-1/2)*dir*acc*np.pow(T[6]-T[5],2) + ((-1/2)*dir*j_limit*np.pow(T[5]-T[4],2) + dir*scan_v_p)*(T[6]-T[5]) + \
                        (1/6)*dir*j_limit*np.pow(t-T[6],3) + (-1/2)*dir*acc*np.pow(t-T[6],2) + (-dir*acc*(T[6]-T[5]) + (-1/2)*dir*j_limit*np.pow(T[5]-T[4],2) + dir*scan_v_p)*(t-T[6])
            elif t > T[7] and t <= T[8]: # dwell
                q_temp = end_point
            elif t > T[8]:
                q_temp = end_point
                end_flag = 1
        else:
            T = np.zeros(4)
            T[0] = t_dwell
            T[1] = t_dwell + ta # initial acceleration
            T[2] = t_dwell + ta + tb + td # slow down
            T[3] = t_dwell + ta + tb + td + t_dwell # dwell

            q_temp = 0
            if t <= T[0]: # dwell
                q_temp = start_point
            elif t > T[0] and t <= T[1]: # initial acceleration
                q_temp = start_point + (1/2)*dir*acc_p*np.pow((t-T[0]),2)
            elif t > T[1] and t <= T[2]: # slow down
                q_temp = start_point + (1/2)*dir*acc_p*np.pow((T[1]-T[0]),2) + (-1/2)*dir*acc_p*np.pow(t-T[1],2) + (dir*acc_p*(T[1]-T[0]))*(t-T[1])
            elif t > T[2] and t <= T[3]: # dwell
                q_temp = end_point
            elif t > T[3]:
                q_temp = end_point
                end_flag = 1

        return q_temp, end_flag

    def __store_trajectory_data(self, motionProfile: MotionProfile, start_position: List[float], move_axis: int, Ts: float) -> None:
        qdot = np.gradient(motionProfile.q, Ts)
        qddot = np.gradient(qdot, Ts)

        self.trajectoryTime = motionProfile.t
        self.perturbation = motionProfile.q
        self.endIndices = motionProfile.endIndex

        # Clear vectors
        self.positionTrajectory.clear()
        self.velocityTrajectory.clear()
        self.accelerationTrajectory.clear()

        # Create current vectors
        currentPosition = copy.copy(start_position)
        currentVelocity = [0.0] * len(start_position)
        currentAcceleration = [0.0] * len(start_position)

        for i in range(len(motionProfile.q)):
            currentPosition[move_axis] = motionProfile.q[i]
            currentVelocity[move_axis] = qdot[i]
            currentAcceleration[move_axis] = qddot[i]

            self.positionTrajectory.append(currentPosition.copy())
            self.velocityTrajectory.append(currentVelocity.copy())
            self.accelerationTrajectory.append(currentAcceleration.copy())