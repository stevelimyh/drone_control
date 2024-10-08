#!/usr/bin/python3
import gym
import numpy as np
import mujoco
import mujoco_viewer
import os
import random
import copy
from custom_utils import *
from icecream import ic

ic.configureOutput(includeContext=True, contextAbsPath=True)
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)
# Global variables
RANGE_LIMIT = 10
RESOLUTION = 10
GOAL_TOLERANCE = 0.1  # if the drone is within 0.1m of the goal, it is considered to have reached the goal
FULL_THROTTLE = 7
DRONE_MODEL_PATH = os.path.join(os.getcwd(), "asset/skydio_x2/scene.xml")
ROLL_TARGET = 5 # degrees
PITCH_TARGET = 5 # degrees
ROLL_THRESHOLD = 90 # degrees
PITCH_THRESHOLD = 90 # degrees
HEIGHT_LOWER_LIMIT = 0.05 # m
IDLE_POSITION_THRESHOLD = 0.05
GYRO_SMOOTH_THRESHOLD = 0.05  # Threshold for angular velocity (rad/s)
ACC_SMOOTH_THRESHOLD = 0.1    # Threshold for linear acceleration (m/s^2)


ACTION_COST, IDLE_COST, GOAL_REWARD, OUT_OF_BOUND_REWARD, FLIPPED_REWARD, SMOOTH_MOTION_REWARD = -0.1, -0.2, 1.0, -1.0, -2.0, 0.2
ACTIONS = [
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
]


class DroneControlGym(gym.Env):
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(DRONE_MODEL_PATH)
        self.drone = mujoco.MjData(self.model)
        # Get the index of the IMU sensors (accelerometer and gyro) from the XML definition
        self.imu_gyro_start = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_gyro')]
        self.imu_acc_start = self.model.sensor_adr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'body_linacc')]
        
        # Initialize IMU readings
        self.imu_acc_data = []
        self.imu_gyro_data = []
        self.sensor_attributes = []
        
        # Print the IMU data
        ic(self.imu_acc_data)
        ic(self.imu_gyro_data)
        mujoco.mj_step(self.model, self.drone)
        self.motor_states = [
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
            [0] * RESOLUTION,
        ]
        self.last_motor_states = copy.deepcopy(self.motor_states)
        self.goal_pose = [
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.0, RANGE_LIMIT),
            random.uniform(0.1, RANGE_LIMIT),
        ]  # randomly initialize goal pose (x, y, z) within RANGE_LIMIT
        self.drone_pose = None
        self.last_drone_pose = [0.0, 0.0, 0.1]
        self.drone_rpy = None
        self.drone_motor_thrust = None
        self.distance_to_goal = None
        self.drone_last_ctrl = copy.deepcopy(self.drone.ctrl)
        self.step_count = 1

    def _update_drone_pose_from_sim(self):
        rpy_angles = np.rad2deg(quaternion_to_rpy(self.drone.xquat[1]))
        self.drone_rpy = np.array(rpy_angles)
        self.drone_pose = np.array(self.drone.xpos[1])

    def _update_motor_thrust(self):
        duty_cycle = [0] * 4
        thrust = [0] * 4
        for index, individual_motor_state in enumerate(self.motor_states):
            duty_cycle[index] = average(individual_motor_state)
            thrust[index] = duty_cycle[index] * FULL_THROTTLE
        self.drone.ctrl[:] = np.array(thrust)
    
    def _update_imu_data(self):
        self.imu_acc_data = self.drone.sensordata[self.imu_acc_start:self.imu_acc_start + 3]  # Accelerometer (x, y, z)
        self.imu_gyro_data = self.drone.sensordata[self.imu_gyro_start:self.imu_gyro_start + 3]  # Gyroscope (x, y, z)
        return np.array([self.imu_acc_data, self.imu_gyro_data])
        
    def _calculate_goal_attributes(self):
        # calculate normalised unit vector [dx, dy, dz] and distance d
        # between self.drone_pose and self.goal_pose
        self.distance_to_goal = distance(self.drone_pose, self.goal_pose)
        dx, dy, dz = 0, 0, 0  # placeholder, please implement this
        return [dx, dy, dz, self.distance_to_goal]

    def _calculate_reward(self):
        # calculate reward based on the current state of the drone and if the drone has reached the goal
        # Check if the roll and pitch are close to zero (upright)
        if self.distance_to_goal < GOAL_TOLERANCE and abs(self.drone_rpy[0]) < ROLL_TARGET and abs(self.drone_rpy[1]) < PITCH_TARGET:
            print("The drone is staying upright at the goal.")
            return True, GOAL_REWARD
        else:
            reward = 0
            # if drone is flipped (or collided?), return FLIPPED_REWARD
            if abs(self.drone_rpy[0]) > ROLL_THRESHOLD or abs(self.drone_rpy[1]) > PITCH_THRESHOLD:
                print("drone has flipped")
                reward += FLIPPED_REWARD
            # if drone is out of RANGE_LIMIT return OUT_OF_BOUND_REWARD, not sure need this or not
            if self.drone_pose[0]> RANGE_LIMIT or self.drone_pose[1]>RANGE_LIMIT:
                print("drone is out of bound")
                reward += OUT_OF_BOUND_REWARD
            # check if current duty cycle is equal to previous duty cycle and position is same
            if np.array_equal(self.drone.ctrl, self.drone_last_ctrl) and np.all(np.abs(self.drone_pose - self.last_drone_pose) < IDLE_POSITION_THRESHOLD):
                print("drone is idle in place")
                reward += IDLE_COST
            else: 
                print("drone took valid action")
                reward += ACTION_COST
            
                # Check for smooth angular motion (low angular velocity)
                if np.all(np.abs(self.sensor_attributes[1]) < GYRO_SMOOTH_THRESHOLD):
                    print("drone is rotating smoothly")
                    reward += SMOOTH_MOTION_REWARD
                
                # Check for smooth linear motion (low acceleration)
                if np.all(np.abs(self.sensor_attributes[0]) < ACC_SMOOTH_THRESHOLD):
                    print("drone is moving smoothly")
                    reward += SMOOTH_MOTION_REWARD
            return False, reward

    def get_pose(self):
        self._update_drone_pose_from_sim()
        return {
            "x": self.drone_pose[0],
            "y": self.drone_pose[1],
            "z": self.drone_pose[2],
            "roll": self.drone_rpy[0],
            "pitch": self.drone_rpy[1],
            "yaw": self.drone_rpy[2],
        }

    def get_goal(self):
        return {"x": self.goal_pose[0], "y": self.goal_pose[1], "z": self.goal_pose[2]}

    def reset(self):
        pass

    def step(self, action):
        # using action given, pop the first motor state and append the new motor state for each motor
        for index, individual_action in enumerate(action):
            self.motor_states[index].pop(0)
            self.motor_states[index].append(individual_action)
        self._update_motor_thrust()
        mujoco.mj_step(self.model, self.drone)
        self.step_count += 1

        self._update_drone_pose_from_sim()
        self.drone_motor_thrust = list(self.drone.actuator_force)

        print(f"quadcopter xyz: {self.drone_pose}")
        print(f"quadcopter rpy: {self.drone_rpy}")

        self.goal_attributes = self._calculate_goal_attributes()  # return list of [dx, dy, dz, d]
        self.sensor_attributes = self._update_imu_data() # return array of [[accx, accy, accz], [velr, velp, vely]]
        self.has_finished, self.reward = self._calculate_reward()  # return bool and float
        
        self.drone_last_ctrl = copy.deepcopy(self.drone.ctrl)
        self.last_drone_pose = copy.deepcopy(self.drone_pose)

        # eg 2.0, True, [0.3, 0.4, 0.1, 5.0], [33.0, 44.0, 55.0], [3.5, 3.5, 3.6, 3.7]
        return self.reward, self.has_finished, self.goal_attributes, self.drone_rpy, self.drone_motor_thrust

    def get_last_state(self):
        return self.reward, self.has_finished, self.goal_attributes, self.drone_rpy, self.drone_motor_thrust

    def render(self):
        viewer = mujoco_viewer.MujocoViewer(self.model, self.drone)
        print("Request to render image, press Ecs to continue")
        while viewer.is_alive:
            viewer.render()


if __name__ == "__main__":
    # Sample Usage
    gym_env = DroneControlGym()

    print(f"step: {gym_env.step_count}")
    while gym_env.step_count < 25:
        print(f"step: {gym_env.step_count}, drone pose: {gym_env.get_pose()} state: {gym_env.step(ACTIONS[0])}")
        ic(gym_env.sensor_attributes)
        ic(gym_env.last_drone_pose)
    gym_env.render()

    print(f"step: {gym_env.step_count}")
    while gym_env.step_count < 70:
        print(f"step: {gym_env.step_count}, drone pose: {gym_env.get_pose()} , state: {gym_env.step(ACTIONS[15])}")
        ic(gym_env.sensor_attributes)
        ic(gym_env.last_drone_pose)
    gym_env.render()
