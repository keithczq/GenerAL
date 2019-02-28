#!/usr/bin/env python
import copy
import time

import roslib

roslib.load_manifest('my_seed_hand')

import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from diagnostic_msgs.msg import DiagnosticArray
import numpy as np


def map_to_sim_frame(joint_name, value):
    joint_angle = copy.copy(value)
    if joint_name == "forearm__base":
        joint_angle /= 2 * np.pi
        joint_angle *= np.pi
        joint_angle -= np.pi / 2
    elif joint_name == "palm_axis__forearm":
        joint_angle /= 2 * np.pi
        joint_angle *= 2 * 0.785398163397
        joint_angle = 2 * 0.785398163397 - joint_angle
        joint_angle -= 0.785398163397
    elif joint_name == "palm__palm_axis":
        joint_angle /= 2 * np.pi
        joint_angle *= 2 * 0.785398163397
        joint_angle -= 0.785398163397
    elif joint_name == "Rproximal__palm":
        joint_angle /= 2 * np.pi
        joint_angle *= 1.4835298642
        joint_angle -= 0
    elif joint_name == "Mproximal__palm":
        joint_angle /= 2 * np.pi
        joint_angle *= 1.4835298642
        joint_angle -= 0
    elif joint_name == "palm__thumb_base":
        joint_angle /= 2 * np.pi
        joint_angle *= np.pi / 2
        joint_angle -= 0
    elif joint_name == "Tproximal__thumb_base":
        joint_angle /= 2 * np.pi
        joint_angle *= np.pi / 2
        joint_angle -= 0
    elif joint_name == "Iproximal__palm":
        joint_angle /= 2 * np.pi
        joint_angle *= 1.4835298642
        joint_angle -= 0
    return joint_angle


def map_to_real_frame(joint_name, value):
    joint_angle = copy.copy(value)
    if joint_name == "forearm__base":
        joint_angle += np.pi / 2
        joint_angle /= np.pi
        joint_angle *= 2 * np.pi
    elif joint_name == "palm_axis__forearm":
        joint_angle += 0.785398163397
        joint_angle = 2 * 0.785398163397 - joint_angle
        joint_angle /= 2 * 0.785398163397
        joint_angle *= 2 * np.pi
    elif joint_name == "palm__palm_axis":
        joint_angle += 0.785398163397
        joint_angle /= 2 * 0.785398163397
        joint_angle *= 2 * np.pi
    elif joint_name == "Rproximal__palm":
        joint_angle += 0
        joint_angle /= 1.4835298642
        joint_angle *= 2 * np.pi
    elif joint_name == "Mproximal__palm":
        joint_angle += 0
        joint_angle /= 1.4835298642
        joint_angle *= 2 * np.pi
    elif joint_name == "palm__thumb_base":
        joint_angle += 0
        joint_angle /= np.pi / 2
        joint_angle *= 2 * np.pi
    elif joint_name == "Tproximal__thumb_base":
        joint_angle += 0
        joint_angle /= np.pi / 2
        joint_angle *= 2 * np.pi
    elif joint_name == "Iproximal__palm":
        joint_angle += 0
        joint_angle /= 1.4835298642
        joint_angle *= 2 * np.pi
    return joint_angle


class Client:
    def __init__(self, motor_name, simulation_input=False, ros_node=True, logging=True):
        self.logging = logging
        if ros_node:
            rospy.init_node('trajectory_client', anonymous=True)
        self.name = motor_name
        self.jta = actionlib.SimpleActionClient('/' + self.name + '/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        self.sub = rospy.Subscriber('/diagnostics', DiagnosticArray, self.joint_status_callback)
        self.joint_names = ['wrist_rotation', 'wrist_adduction', 'wrist_flexion', 'thumb_adduction', 'thumb_flexion',
                            'index_flexion', 'middle_flexion', 'ring_and_pinky_flexion']
        self.raw_joint_names = ['forearm__base', 'palm_axis__forearm', 'palm__palm_axis', 'palm__thumb_base',
                                'Tproximal__thumb_base', 'Iproximal__palm', 'Mproximal__palm', 'Rproximal__palm']

        self.simulation_input = simulation_input
        self.goal_joint_angles = [0, 0, 0, 0, 0, 0, 0, 0]
        self.current_joint_state = list((np.ones(8) * 0.5))
        self.prev_joint_state = [0, 0, 0, 0, 0, 0, 0, 0]
        self.default_joint_maxima = [2 * np.pi, 2 * np.pi, 2 * np.pi, 4 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi,
                                     4 * np.pi]
        self.joint_state_maxima = None

        self.reset_joint_maxima()

        self.OVERLOAD_THRESHOLD = 0.2
        self.PULLBACK_OFFSET = 0.15
        self.PUSHFORWARD_OFFSET = 5
        self.CALLBACK_INTERVAL = 0
        self.last_callback_time = time.time()
        self.PROTECT_FROM_OVERLOAD = True

        self.jta.wait_for_server()

    def compute_joint_maxima(self, slow):
        if not self.PROTECT_FROM_OVERLOAD:
            self.reset_joint_maxima()
        else:
            for i in range(3, len(self.current_joint_state)):
                # NOTE: if the finger didn't move enough, the hand is either pulled back or against obstacles
                if 0 < self.current_joint_state[i] - self.prev_joint_state[i] <= self.OVERLOAD_THRESHOLD:
                    # NOTE: if the hand is against obstacles, pull back
                    if self.joint_state_maxima[i] > self.current_joint_state[i]:
                        self.joint_state_maxima[i] = self.current_joint_state[i] - self.PULLBACK_OFFSET
                    # NOTE: if the hand is stopped, make it move
                    else:
                        self.joint_state_maxima[i] = self.current_joint_state[i] + self.PUSHFORWARD_OFFSET * (slow + 1)
                else:
                    self.joint_state_maxima[i] = self.current_joint_state[i] + self.PUSHFORWARD_OFFSET * (slow + 1)
        self.prev_joint_state = copy.copy(self.current_joint_state)

    def reset_joint_maxima(self):
        print('resetting joint maxima')
        self.joint_state_maxima = copy.copy(self.default_joint_maxima)

    def map_to_real_frames(self, angles):
        joint_angles = []
        for i in range(len(angles)):
            joint_angle = map_to_real_frame(self.raw_joint_names[i], angles[i])
            joint_angles.append(joint_angle)
        return joint_angles

    def map_to_simulated_frames(self, angles):
        joint_angles = []
        for i in range(len(angles)):
            joint_angle = map_to_sim_frame(self.raw_joint_names[i], angles[i])
            joint_angles.append(joint_angle)
        return joint_angles

    def move_joint_till_match(self, angles):
        self.PROTECT_FROM_OVERLOAD = False
        valid_angles = np.clip(angles, 0, 2 * np.pi)
        virtual_angles = np.array(valid_angles)
        count = 0
        while not np.allclose(self.current_joint_state[:4], valid_angles[:4], atol=1e-2, rtol=1e-2):
            delta = np.array(valid_angles) - np.array(self.current_joint_state)
            print(self.current_joint_state, valid_angles, virtual_angles, delta)

            delta = np.array(valid_angles) - np.array(self.current_joint_state)
            virtual_angles = np.clip(virtual_angles + delta, -np.pi, 2 * np.pi)
            self.move_joint(virtual_angles)
            count += 1
            if count > 30:
                self.PROTECT_FROM_OVERLOAD = True
                return False

        delta = np.array(valid_angles) - np.array(self.current_joint_state)
        print(self.current_joint_state, valid_angles, delta)
        self.PROTECT_FROM_OVERLOAD = True
        return True

    def move_joint(self, angles, slow=False):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names[:]

        joint_angles = []

        self.compute_joint_maxima(slow)

        i = 0
        for joint_name in self.raw_joint_names:
            joint_angle = angles[i]

            if self.simulation_input:
                joint_angle = map_to_real_frame(joint_name, joint_angle)
            joint_angle = min(joint_angle, self.joint_state_maxima[i])
            joint_angle = max(0, joint_angle)
            joint_angles.append(joint_angle)
            i += 1

        point = JointTrajectoryPoint()
        point.positions = joint_angles

        point.time_from_start = rospy.Duration(1.0)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)
        time.sleep(.5)

    def joint_status_callback(self, message):
        if time.time() - self.last_callback_time < self.CALLBACK_INTERVAL:
            return
        if self.logging:
            rospy.loginfo("joint_status_callback! current: %.1f  old: %.1f  delta: %.5f  maxima: %.1f" % (
                self.current_joint_state[-3], self.prev_joint_state[-3],
                self.current_joint_state[-3] - self.prev_joint_state[-3], self.joint_state_maxima[-3]))
            print(self.current_joint_state)
        self.last_callback_time = time.time()
        for joint_status in message.status:
            for i in range(len(self.joint_names)):
                if self.joint_names[i] in joint_status.name:
                    for item in joint_status.values:
                        if item.key == "Position":
                            self.current_joint_state[i] = float(item.value)
                            break
                    break


def example():
    hand = Client('seed_hand_controller',
                  simulation_input=False)  # if input values are in radians [0,2pi] disable simulation input else

    i = -0.5
    offset = 5

    while True:
        print("Step {}".format(i))
        hand.move_joint([3.14, 3.14, 3.14, 10, 0, i, 0, 0])
        # ipdb.set_trace()
        # client.move_joint([5.0]*8)
        i += offset  # if i >= 5.0 or i <= 0:  #     offset = -offset


if __name__ == '__main__':
    example()
