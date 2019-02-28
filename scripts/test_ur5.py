import geometry_msgs
import moveit_commander
import rospy
import world_manager
from geometry_msgs.msg import PoseStamped


def get_pose(x_pos, y_pos, z_pos, x_ori, y_ori, z_ori, w_ori):
    pose = geometry_msgs.msg.Pose()
    pose.position.x = x_pos
    pose.position.y = y_pos
    pose.position.z = z_pos
    pose.orientation.x = x_ori
    pose.orientation.y = y_ori
    pose.orientation.z = z_ori
    pose.orientation.w = w_ori
    return pose


BASE_LINK = "base_link"
ARM = "ur5"
arm_commander = moveit_commander.MoveGroupCommander(ARM)
rospy.init_node('test_node')
gripper_pose_stamped = PoseStamped()
gripper_pose_stamped.header.frame_id = BASE_LINK
gripper_pose_stamped.header.stamp = rospy.Time.now()
gripper_pose_stamped.pose = get_pose(
    *[-0.456343782494, -0.604612734285, -0.076200921763, -0.244298804265, 0.202824907276, 0.863450201957,
      0.391961605216])
world_manager.world_manager_client.add_tf('Grasp', gripper_pose_stamped)
arm_commander.set_pose_target(gripper_pose_stamped)
result = arm_commander.plan()
import ipdb

ipdb.set_trace()
