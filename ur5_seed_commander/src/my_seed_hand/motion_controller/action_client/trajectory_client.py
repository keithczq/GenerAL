#!/usr/bin/env python
import roslib
roslib.load_manifest('my_seed_hand')

import rospy
import actionlib
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryAction, JointTrajectoryGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import JointState
import numpy as np
import time

class Client:
    def __init__(self, motor_name, simulation_input=False):
        # arm_name should be b_arm or f_arm
        rospy.init_node('trajectory_client', anonymous=True)
        self.name = motor_name
        self.jta = actionlib.SimpleActionClient('/' + self.name + '/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        rospy.Subscriber('/diagnostics', DiagnosticArray, self.joint_status_callback)
        rospy.Subscriber("/teleop/joint_states", JointState, self.teleop_callback)

        self.simulation_input = simulation_input

        self.joint_names = ['wrist_rotation', 'wrist_adduction', 'wrist_flexion', 'thumb_adduction', 'thumb_flexion',
                            'index_flexion', 'middle_flexion', 'ring_and_pinky_flexion']
        self.raw_joint_names = ['forearm__base', 'palm_axis__forearm', 'palm__palm_axis', 'palm__thumb_base',
                                'Tproximal__thumb_base', 'Iproximal__palm', 'Mproximal__palm', 'Rproximal__palm']

        # internal state of client (slave client)
        self.current_joint_state = [1,1,1,1,1,1,1,1]
        self.prev_joint_state = [0,0,0,0,0,0,0,0]

        # state of commands (master client)
        self.prev_joint_command = [0,0,0,0,0,0,0,0]
        self.curr_joint_command = [0,0,0,0,0,0,0,0]

        self.default_joint_maxima = [5,5,5,5,5,5,5,5]
        self.joint_state_maxima = None
        self.reset_joint_maxima()

        self.OVERLOAD_THRESHOLD = 0.1
        self.PULLBACK_OFFSET = 0.25
        self.RELEASE_THRESHOLD = 0.3

        self.collision = False

        rospy.loginfo('Waiting for joint trajectory action')
        self.jta.wait_for_server()
        rospy.loginfo('Found joint trajectory action!')

        self.time = time.time()
        self.SLEEP = 1

    def run(self, sensitivity=5.0):     # the lower the sensitivity the greater the precision of the real time client commands
        while not rospy.is_shutdown():
            now = time.time()
            if now - self.time > self.SLEEP:
                self.time = now
                if self.command_error(self.curr_joint_command, self.prev_joint_command) > sensitivity:
                    # print("Moving to "+", ".join(map(str, [self.curr_joint_command[5]])))
                    self.move_joint(self.curr_joint_command)
                    self.prev_joint_command = self.curr_joint_command[:]
            else:
                self.move_joint(self.prev_joint_command)

    def compute_joint_maxima(self):
        collision = False
        for i in range(3, len(self.current_joint_state)):
            delta = self.current_joint_state[i] - self.prev_joint_state[i]
            if delta < 0:
                self.reset_joint_maxima()
                self.collision = False
                break
            lower_bound = 0.0 if self.collision else 0.01
            if lower_bound <= self.current_joint_state[i] - self.prev_joint_state[i] <= self.OVERLOAD_THRESHOLD:
                self.joint_state_maxima[i] = self.current_joint_state[i] - self.PULLBACK_OFFSET
                if not self.collision:
                    self.collision = True
                collision = True
        self.collision = collision

    def reset_joint_maxima(self):
        self.joint_state_maxima = self.default_joint_maxima[:]

    def move_joint(self, angles):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names[:]
        joint_angles = []
        self.compute_joint_maxima()

        i = 0
        for joint_name in self.raw_joint_names:
            # if joint_name == "Iproximal__palm":
            joint_angle = angles[i]
            joint_angle = min(joint_angle, self.joint_state_maxima[i])
            joint_angle = max(0.1, joint_angle)
            joint_angles.append(joint_angle)
            i += 1

        point = JointTrajectoryPoint()
        point.positions = joint_angles
        point.time_from_start = rospy.Duration(1.0)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)
        # rospy.loginfo("current: %.1f  old: %.1f  delta: %.5f  maxima: %.1f" % (self.current_joint_state[5], self.prev_joint_state[5], self.current_joint_state[5]-self.prev_joint_state[5], self.joint_state_maxima[5]))
        self.reset_joint_maxima()
        # rospy.loginfo("Moving to ["+", ".join(map(str,point.positions))+"]")

    def command_error(self, current_command, prev_command):
        return np.linalg.norm(np.array(current_command)-np.array(prev_command))

    def teleop_callback(self, message):
        joint_angles = []
        for i in range(len(self.raw_joint_names)):
            joint_angle = message.position[message.name.index(self.raw_joint_names[i])]
            if self.simulation_input:
                if self.raw_joint_names[i] == "forearm__base":
                    joint_angle += 1.57
                    joint_angle /= 2*1.57
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "palm_axis__forearm":
                    joint_angle += 0.79
                    joint_angle /= 2*0.79
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "palm__palm_axis":
                    joint_angle += 0.79
                    joint_angle /= 2*0.79
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "Rproximal__palm":
                    joint_angle += 0
                    joint_angle /= 1.48
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "Mproximal__palm":
                    joint_angle += 0
                    joint_angle /= 1.48
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "palm__thumb_base":
                    joint_angle += 0
                    joint_angle /= 1.57
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "Tproximal__thumb_base":
                    joint_angle += 0
                    joint_angle /= 1.57
                    joint_angle *= 6.28
                elif self.raw_joint_names[i] == "Iproximal__palm":
                    joint_angle += 0
                    joint_angle /= 1.48
                    joint_angle *= 6.28
            joint_angles.append(joint_angle)

        self.curr_joint_command = joint_angles[:]

    def joint_status_callback(self, message):
        self.prev_joint_state = self.current_joint_state[:]
        for joint_status in message.status:
            pos = None
            for i in range(len(self.joint_names)):
                if self.joint_names[i] in joint_status.name:
                    pos = i
                    break
            if pos:
                for item in joint_status.values:
                    if item.key == "Position":
                        self.current_joint_state[pos] = float(item.value)
                        break


if __name__ == '__main__':
    hand = Client('seed_hand_controller', simulation_input=True)
    hand.run()


