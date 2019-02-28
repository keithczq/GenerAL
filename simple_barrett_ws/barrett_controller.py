#!/usr/bin/env python
from threading import Lock

import moveit_python
import rospy
import tf2_kdl
import tf2_ros
import tf_conversions
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import MoveItErrorCodes, PositionIKRequest
from moveit_msgs.srv import GetPositionIK


class StaubliBarrettController(object):
    TIP_LINK = "staubli_rx60l_link7"
    BASE_LINK = "staubli_rx60l_link1"
    ARM = "StaubliArm"
    GRIPPER = "BarrettHand"

    OPEN_POSITION = [0] * 2
    CLOSED_POSITION = [1.3] * 2
    FINGER_MAXTURN = 1.3

    def __init__(self):
        self._lock = Lock()

        self.arm_move_group = moveit_python.MoveGroupInterface(self.ARM, self.BASE_LINK)
        self.gripper_move_group = moveit_python.MoveGroupInterface(self.GRIPPER, self.BASE_LINK)
        self.arm_move_group.setPlannerId('RRTConnectkConfigDefault')

        self.compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        rospy.loginfo('Waiting for controllers...')
        rospy.loginfo('connected to controllers.')

    def __del__(self):
        # cancel moveit goals
        self.arm_move_group.get_move_action().cancel_all_goals()
        self.gripper_move_group.get_move_action().cancel_all_goals()

    def get_ik(self, position, orientation, planner_time_limit=0.5):

        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.BASE_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = Pose(Point(*position), Quaternion(*orientation))

        service_request = PositionIKRequest()
        service_request.group_name = self.ARM
        service_request.ik_link_name = self.TIP_LINK
        service_request.pose_stamped = gripper_pose_stamped
        service_request.timeout.secs = planner_time_limit
        service_request.avoid_collisions = True

        try:
            resp = self.compute_ik(ik_request=service_request)
            return resp
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def move_arm_joint_values(self, joint_values, wait=True):
        self.arm_move_group.moveToJointPosition(
            joints=['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'], positions=joint_values,
            wait=wait)
        result = self.arm_move_group.get_move_action().get_result()

    def set_end_effector_pose(self, position, orientation, wait=True):
        # create pose

        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.BASE_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = Pose(Point(*position), Quaternion(*orientation))

        # Move gripper frame to the pose specified
        self.arm_move_group.moveToPose(gripper_pose_stamped, self.TIP_LINK, wait=wait)
        result = self.arm_move_group.get_move_action().get_result()

        # error checking
        if result:
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                rospy.logerr("Arm goal in state: %s", self.arm_move_group.get_move_action().get_state())
        else:
            rospy.logerr("MoveIt! failure no result returned.")

    def set_finger_positions(self, finger_positions):
        """Send a gripper goal to the action server."""

        result = self.gripper_move_group.moveToJointPosition(
            joints=['m1n6s200_joint_finger_1', 'm1n6s200_joint_finger_2'], positions=finger_positions)
        # error checking
        if result:
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                rospy.logerr("Gripper goal in state: %s", self.gripper_move_group.get_move_action().get_state())
        else:
            rospy.logerr("MoveIt! failure no result returned.")

    def get_transform(self, reference_frame, target_frame):
        try:
            transform_stamped = self.tfBuffer.lookup_transform(reference_frame, target_frame, rospy.Time(0),
                                                               rospy.Duration(1.0))
            translation_rotation = tf_conversions.toTf(tf2_kdl.transform_to_kdl(transform_stamped))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException), e:
            print(e)
            rospy.logerr('FAILED TO GET TRANSFORM FROM %s to %s' % (reference_frame, target_frame))
            return None
        return translation_rotation

    def open_hand(self):
        rospy.loginfo('Opening gripper...')
        self.set_finger_positions(self.OPEN_POSITION)
        rospy.loginfo('Gripper open.')

    def close_hand(self):
        rospy.loginfo('Closing gripper...')
        self.set_finger_positions(self.CLOSED_POSITION)
        rospy.loginfo('Gripper closed.')

    # Stop all movement
    def cancel_move(self):
        self.arm_move_group.get_move_action().cancel_all_goals()

    def move_gripper(self, pos, rot):
        rospy.loginfo('Moving gripper...')
        self.set_end_effector_pose(pos, rot)
        rospy.loginfo('Gripper moved.')


if __name__ == '__main__':
    rospy.init_node('test_node')

    staubli_controller = StaubliBarrettController()
    import ipdb

    ipdb.set_trace()

    # position = [0.215, -0.203, 0.419]
    # orientation = [0.682, -0.229, 0.688, 0.090]
    #
    position = [-0.317, 0.020, 0.694]
    orientation = [0.000, 0.354, 0.000, 0.935]
    staubli_controller.set_end_effector_pose(position, orientation)
    import ipdb

    ipdb.set_trace()

    staubli_controller.set_end_effector_pose([0.145, -0.083, 0.803], [-0.539, -0.181, 0.573, 0.590])
    staubli_controller.get_transform('staubli_rx60l_link1', 'staubli_rx60l_link7')

    # staubli_controller.open_hand()  # staubli_controller.close_hand()
