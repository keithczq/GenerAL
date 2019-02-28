#!/usr/bin/env python
import roslib
roslib.load_manifest('my_seed_hand')

import rospy
import actionlib
from std_msgs.msg import Float64
import trajectory_msgs.msg 
import control_msgs.msg  
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryAction, JointTrajectoryGoal, FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState


class Joint:
    def __init__(self, motor_name):
        #arm_name should be b_arm or f_arm
        rospy.init_node('trajectory_client', anonymous=True)
        self.name = motor_name           
        self.jta = actionlib.SimpleActionClient('/'+self.name+'/follow_joint_trajectory', FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for joint trajectory action')
        self.jta.wait_for_server()
        rospy.loginfo('Found joint trajectory action!')
        rospy.Subscriber("/joint_states", JointState, self.callback)
        rospy.spin()

    def move_joint(self, angles):
        goal = FollowJointTrajectoryGoal()                  
        char = self.name[0] #either 'f' or 'b'
        goal.trajectory.joint_names = ['wrist_rotation', 'wrist_adduction', 'wrist_flexion', 'thumb_adduction', 'thumb_flexion', 'index_flexion', 'middle_flexion', 'ring_and_pinky_flexion']
        # ['claw_1f'+char, 'traction_1f'+char,'joint_1f'+char]
        point = JointTrajectoryPoint()
        point.positions = angles
        # point.time_from_start = rospy.Duration(2)                   
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)
        # rospy.loginfo('moving...')
        rospy.loginfo(angles)


    def callback(self, message):
        joint_names = ['wrist_rotation', 'wrist_adduction', 'wrist_flexion', 'thumb_adduction', 'thumb_flexion', 'index_flexion', 'middle_flexion', 'ring_and_pinky_flexion']
        raw_joint_names = ['forearm__base', 'palm_axis__forearm', 'palm__palm_axis', 'palm__thumb_base', 'Tproximal__thumb_base', 'Iproximal__palm', 'Mproximal__palm', 'Rproximal__palm']
        joint_angles = []
        
        for joint_name in raw_joint_names:
            joint_angle = message.position[message.name.index(joint_name)]

            if joint_name == "forearm__base":
                joint_angle += 1.57
                joint_angle /= 2*1.57
                joint_angle *= 6.28
            elif joint_name == "palm_axis__forearm":
                joint_angle += 0.79
                joint_angle /= 2*0.79
                joint_angle *= 6.28
            elif joint_name == "palm__palm_axis":
                joint_angle += 0.79
                joint_angle /= 2*0.79
                joint_angle *= 6.28
            elif joint_name == "Rproximal__palm":
                joint_angle += 0
                joint_angle /= 1.48
                joint_angle *= 6.28
            elif joint_name == "Mproximal__palm":
                joint_angle += 0
                joint_angle /= 1.48
                joint_angle *= 6.28
            elif joint_name == "palm__thumb_base":
                joint_angle += 0
                joint_angle /= 1.57
                joint_angle *= 6.28
            elif joint_name == "Tproximal__thumb_base":
                joint_angle += 0
                joint_angle /= 1.57
                joint_angle *= 6.28
            elif joint_name == "Iproximal__palm":
                joint_angle += 0
                joint_angle /= 1.48
                joint_angle *= 6.28

            joint_angle = min(joint_angle, 5.0)
            joint_angle = max(0.1, joint_angle)
            joint_angles.append(joint_angle)

        self.move_joint(joint_angles)



def main():
    arm = Joint('f_arm_controller')
    arm.move_joint([0.0,5.0,0.0,0.0,0.0,0.0,0.0,0.0])
    # arm.move_joint([6.28,3.14,6.28])


if __name__ == '__main__':
    hand = Joint('f_arm_controller')

