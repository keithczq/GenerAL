#!/usr/bin/env python
import copy
import math
import operator
import os
import random
import time
import traceback
from threading import Lock

import actionlib
import bhand_controller.msg
import bhand_controller.srv
import curvox.cloud_to_mesh_conversions
import ipdb
import moveit_commander
import moveit_python
import numpy as np
import ros_numpy
import rospy
import sensor_msgs.point_cloud2 as pcl2
import std_msgs.msg
import tf2_ros
import world_manager
import world_manager.world_manager_client as wm_client
from bhand_controller.msg import TactileArray
from control_msgs.msg import FollowJointTrajectoryAction
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform, TransformStamped
from moveit_msgs.msg import MoveItErrorCodes, PositionIKRequest, RobotState, RobotTrajectory
from moveit_msgs.srv import ExecuteKnownTrajectory, GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectoryPoint

import kinect_rgbd_standalone
from config import d, g
from pybullet_commander.pybullet_barrett_commander import PybulletBarrettCommander
from staubli_barrett_ws.src.staubli_barrett_meta_package.barrett_hand_ros.rqt_bhand.src.rqt_bhand.tact_maps import \
    AVERAGE_HISTORY, THRESHOLD_VALUE
from utils import bilinear_interpolation, get_point_clouds_from_depth, get_pose, pose2PosOri, qua_to_dirvec, \
    vector2GeometryMsgs

try:
    from queue import Queue
except:
    from Queue import Queue


class StabuliBarrettCommander(PybulletBarrettCommander):
    WORLD_LINK = "staubli_rx60l_link1"
    ARM_TIP_LINK = "staubli_rx60l_link7"
    ARM = "StaubliArm"
    GRIPPER = "BarrettHand"
    ARM_GRIPPER = "StaubliArm"
    TIP_LINK_TO_GRIPPER_BASE_CM = 0.127
    HAND_BASE_LINK = 'bh_base_link'

    ARM_JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', ]
    OPEN_POSITION = [0] * 6
    OPEN_GRIP_POSITION = [0.01] * 3
    CLOSED_POSITION = np.array([2.44, 0.84, 2.44, 0.84, 2.44, 0.84])
    FINGER_JOINTS = ['bh_j12_joint', 'bh_j13_joint', 'bh_j22_joint', 'bh_j23_joint', 'bh_j32_joint', 'bh_j33_joint']
    SPREAD_JOINTS = ['bh_j11_joint', 'bh_j21_joint']
    GRIP_JOINTS = ['bh_j12_joint', 'bh_j22_joint', 'bh_j32_joint']
    TIP_JOINTS = ['bh_j13_joint', 'bh_j23_joint', 'bh_j33_joint']
    HAND_JOINTS = ['bh_j11_joint', 'bh_j12_joint', 'bh_j13_joint', 'bh_j21_joint', 'bh_j22_joint', 'bh_j23_joint',
                   'bh_j32_joint', 'bh_j33_joint']
    PLANNING_TIME_LIMIT = 0.8
    MAX_FINGER_1_JOINT = MAX_FINGER_2_JOINT = MAX_FINGER_3_JOINT = 2.44
    LIFT_HEIGHT = 0.25

    def __init__(self, id=0):
        self.id = id
        self.baseline_efforts = np.zeros(3)
        self.baseline_tactile_contacts = np.zeros(self.num_tactile_links)
        self.keep_grasping = False
        self.slow_grasping = True
        self.last_timestamp = time.time()
        if not self.is_master() or g.hand == 'seed':
            os.environ['ROS_MASTER_URI'] = 'http://localhost:%s' % (11311 + self.id)
        else:
            os.environ['ROS_MASTER_URI'] = 'http://bali.local:%s' % (11311 + self.id)
        os.environ['ROS_HOSTNAME'] = 'delfino.local'
        if not rospy.core.is_initialized():
            rospy.init_node('test_node_%s' % id)
        self.read_ros_params()
        self._lock = Lock()
        self.arm_move_group = moveit_python.MoveGroupInterface(self.ARM, self.WORLD_LINK)
        self.gripper_move_group = moveit_python.MoveGroupInterface(self.GRIPPER, self.WORLD_LINK)
        self.scene = moveit_python.PlanningSceneInterface(self.ARM)
        self.arm_move_group.setPlannerId('RRTConnectkConfigDefault')
        self.arm_move_group.setPlanningTime(self.PLANNING_TIME_LIMIT)
        self.gripper_move_group.setPlanningTime(self.PLANNING_TIME_LIMIT)

        self.compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        if self.is_master():
            self.hand_commander = moveit_commander.MoveGroupCommander(self.GRIPPER)
        self.arm_commander = moveit_commander.MoveGroupCommander(self.ARM)
        if g.hand == 'seed':
            self.arm_commander.set_max_velocity_scaling_factor(0.07 if g.eval_single_object else 0.04)
        self.arm_commander.set_planner_id('RRTConnectkConfigDefault')
        self.arm_gripper_commander = moveit_commander.MoveGroupCommander(self.ARM_GRIPPER)
        self.robot_commander = moveit_commander.RobotCommander()
        self.robot_commander.get_current_state()
        self.arm_commander.set_planning_time(self.PLANNING_TIME_LIMIT)

        self.graspable_bodies = {}
        self.execute_trajectory_service = rospy.ServiceProxy('/execute_kinematic_path', ExecuteKnownTrajectory)

        if self.is_master():
            self.rgbd_setup()
            self.init_finger_joint_states()
            self.init_tactile()
            self.init_tactile_cartesian()

        self.h = np.arange(int(d.camera_h))[:, np.newaxis]
        self.w = np.arange(int(d.camera_w))[np.newaxis, :]

        self.rqt_pub = rospy.Publisher('cache', String)
        self._tact_data = TactileArray()
        self.tact_data_queue = Queue(maxsize=AVERAGE_HISTORY)
        self.auto_grasped = False
        self.arm_preempt_client = actionlib.SimpleActionClient("setFollowTrajectory", FollowJointTrajectoryAction)

    def saveState(self):
        pass

    def rgbd_setup(self):
        kinect_rgbd_standalone.init_service("/camera/rgb/image_color", "/camera/depth_registered/image_raw",
                                            '/camera/depth_registered/points', 'camera_link', 'camera_link')

        self.cloud_pub = rospy.Publisher('point_cloud_debug', PointCloud2, queue_size=10)
        rate = rospy.Rate(10)
        while not kinect_rgbd_standalone.ready():
            print('not ready', self.id)
            rate.sleep()
        if rospy.is_shutdown():
            exit()

    def init_finger_joint_states(self):
        self._topic_joint_states_connected = False
        self.joint_state_pointer = {}
        for i in range(len(self.joint_ids)):
            self.joint_state_pointer[self.joint_ids[i]] = {'joint': self.joint_names[i], 'values': [0, 0, 0]}
        try:
            self._joint_subscriber = rospy.Subscriber('/joint_states', JointState, self._receive_joints_data)
        except ValueError, e:
            rospy.logerr('BHandGUI: Error connecting topic (%s)' % e)

    def init_tactile(self):
        self._tact_topic = '/%s/tact_array' % self.bhand_node_name
        self._tact_data = None
        try:
            self._tact_subscriber = rospy.Subscriber(self._tact_topic, TactileArray, self._receive_tact_data)
        except ValueError, e:
            rospy.logerr('BHandGUI: Error connecting topic (%s)' % e)

    def init_tactile_cartesian(self):
        self._tact_cartesian_topic = '/tf'
        self._tact_cartesian_data = {}
        try:
            self._tact_cartesian_subscriber = rospy.Subscriber(self._tact_cartesian_topic, TransformStamped,
                                                               self._receive_tact_cartesian_data)
        except ValueError, e:
            rospy.logerr('BHandGUI: Error connecting topic (%s)' % e)

    def _receive_tact_cartesian_data(self, msg):
        transforms = msg.transforms
        for transform in transforms:
            if transform.header.frame_id in ['bh_base_link', 'bh_finger_14_link', 'bh_finger_24_link',
                                             'bh_finger_34_link']:
                self._tact_cartesian_data[transform.child_frame_id] = ros_numpy.numpify(transform.transform)

    def _receive_tact_data(self, msg):
        if self._tact_data is None:
            self._tact_data = TactileArray()

        self._tact_data = msg
        self._tact_data.finger1 = list(
            np.array(self._tact_data.finger1) + self.joint_state_pointer['F1_TIP']['values'][2])
        self._tact_data.finger2 = list(
            np.array(self._tact_data.finger2) + self.joint_state_pointer['F2_TIP']['values'][2])
        self._tact_data.finger3 = list(
            np.array(self._tact_data.finger3) + self.joint_state_pointer['F3_TIP']['values'][2])
        if self.tact_data_queue.full():
            self.tact_data_queue.get()
        self.tact_data_queue.put(self._tact_data)

    def read_ros_params(self):
        '''
            Read ROS params from servers
        '''
        _name = rospy.get_name()

        self.bhand_node_name = rospy.get_param('%s/bhand_node_name' % _name, 'bhand_node')
        # Reads the configuration of the joints ids
        self.joint_ids = rospy.get_param('%s/joint_ids' % (self.bhand_node_name),
                                         ['F1', 'F1_TIP', 'F2', 'F2_TIP', 'F3', 'F3_TIP', 'SPREAD_1', 'SPREAD_2'])
        self.joint_names = rospy.get_param('%s/joint_names' % (self.bhand_node_name),
                                           self.FINGER_JOINTS + self.SPREAD_JOINTS)

        rospy.loginfo('%s::read_ros_params: bhand_node_name = %s' % (_name, self.bhand_node_name))
        rospy.loginfo('%s::read_ros_params: joint_ids = %s' % (_name, self.joint_ids))
        rospy.loginfo('%s::read_ros_params: joint_names = %s' % (_name, self.joint_names))

    def _receive_joints_data(self, msg):
        '''
            Handler for Joint States
        '''
        self._joints_data = msg

        self._topic_joint_states_timer = time.time()

        for i in range(len(msg.name)):
            for j in self.joint_state_pointer:
                if self.joint_state_pointer[j]['joint'] == msg.name[i]:
                    self.joint_state_pointer[j]['values'] = [msg.position[i], msg.velocity[i], msg.effort[i]]

        if not self._topic_joint_states_connected:
            rospy.loginfo('Bhand: connection established with /joint_states')
            self._topic_joint_states_connected = True
        if self.keep_grasping:
            now = time.time()
            if now > self.last_timestamp + 5:
                self.autoGrasp(slow=True)
                self.last_timestamp = now

    def avoidTrayCollision(self, lift=True):
        self.scene.removeCollisionObject('scene_mesh')
        self.cache_rqt()

    def execute(self, result, wait):
        self.execute_trajectory_service(result, wait)

    def lift_a_bit(self, delta=0.001):
        self.arm_commander.set_start_state(self.robot_commander.get_current_state())
        pose = self.getArmTipPose()
        pose.position.z += delta
        print('adjusting', pose.position.z, self.baseline_efforts)
        result, _ = self.arm_commander.compute_cartesian_path(waypoints=[pose], eef_step=0.1, jump_threshold=0)
        self.execute(result, True)

    def drop_a_bit(self, delta=0.001):
        self.arm_commander.set_start_state(self.robot_commander.get_current_state())
        pose = self.getArmTipPose()
        pose.position.z -= delta
        result, _ = self.arm_commander.compute_cartesian_path(waypoints=[pose], eef_step=0.1, jump_threshold=0)
        self.execute(result, True)

    def getTargetJointValues(self, finger1, finger2, finger3, delta):
        joint_values = dict(zip(self.HAND_JOINTS, self.hand_commander.get_current_joint_values()))
        target_joint_values = copy.copy(joint_values)
        target_joint_values['bh_j12_joint'] = min(target_joint_values['bh_j12_joint'] + delta * finger1,
                                                  self.MAX_FINGER_1_JOINT - 0.1)
        target_joint_values['bh_j22_joint'] = min(target_joint_values['bh_j22_joint'] + delta * finger2,
                                                  self.MAX_FINGER_2_JOINT - 0.1)
        target_joint_values['bh_j32_joint'] = min(target_joint_values['bh_j32_joint'] + delta * finger3,
                                                  self.MAX_FINGER_3_JOINT - 0.1)
        target_joint_values['bh_j13_joint'] = target_joint_values['bh_j12_joint'] * self.PROXIMAL_TIP_RATIO
        target_joint_values['bh_j23_joint'] = target_joint_values['bh_j22_joint'] * self.PROXIMAL_TIP_RATIO
        target_joint_values['bh_j33_joint'] = target_joint_values['bh_j32_joint'] * self.PROXIMAL_TIP_RATIO
        target_joint_value_array = [target_joint_values[j] for j in (self.FINGER_JOINTS + self.SPREAD_JOINTS)]
        return target_joint_values, target_joint_value_array

    def incrementalGrasp(self, finger1, finger2, finger3, delta):
        try:
            print('incrementalGrasp before execute: %s ' % self.correct_safe_close_position)
            target_joint_value_dict, target_joint_value_array = self.getTargetJointValues(finger1, finger2, finger3,
                                                                                          delta)

            while not self.check_whether_finger_joint_values_are_safe(target_joint_value_array):
                if self.check_whether_finger_joint_values_are_safe_without_table(target_joint_value_array):
                    self.lift_a_bit()
                else:
                    if not self.auto_grasped:
                        self.auto_grasped = True
                        self.autoGrasp()
                    self.updateForceHistory()
                    return

            hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions', bhand_controller.srv.CustomActions)
            hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY,
                               [target_joint_value_dict['bh_j12_joint'], target_joint_value_dict['bh_j22_joint'],
                                target_joint_value_dict['bh_j32_joint'], 0.2 * finger1, 0.2 * finger2, 0.2 * finger3])

            target_joint_values_array = np.array([target_joint_value_dict[joint] for joint in self.HAND_JOINTS])

            while True:
                current_joint_values = np.array(self.hand_commander.get_current_joint_values())[[1, 4, 6]]
                time.sleep(0.2)
                if np.allclose(current_joint_values,
                               np.array(self.hand_commander.get_current_joint_values())[[1, 4, 6]], rtol=1e-4,
                               atol=1e-4):
                    break
                print(np.array(target_joint_values_array)[[1, 4, 6]],
                      np.array(self.hand_commander.get_current_joint_values())[[1, 4, 6]], [finger1, finger2, finger3])
            self.updateForceHistory()
        except:
            traceback.print_exc()
            ipdb.set_trace()

    def process_and_save_pcl_as_mesh(self, cloud_points, mesh_filepath):
        if cloud_points is not None:
            cloud_points = cloud_points[~np.isnan(cloud_points).any(axis=1)]
            ply_data = curvox.cloud_to_mesh_conversions.marching_cubes_completion(cloud_points,
                                                                                  percent_offset=(.5, .5, .5),
                                                                                  smooth=False, patch_size=200)
            ply_data.write(open(mesh_filepath, 'wb'))

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'camera_rgb_optical_frame'
        pose_stamped.pose = Pose(Point(0, 0, -0.01), Quaternion(0, 0, 0, 1))
        wm_client.add_mesh('scene_mesh', mesh_filepath, pose_stamped)

    def getCurrentSpread(self):
        current_joint_values = self.hand_commander.get_current_joint_values()
        return current_joint_values[0]

    def move_single_joint(self, joint_idx=0, total_joint_movement=-1.25, total_time_taken=10., num_waypoints=10):
        current_jvs = self.robot_commander.get_current_state().joint_state.position
        rt = RobotTrajectory()
        rt.joint_trajectory.header.frame_id = self.WORLD_LINK
        rt.joint_trajectory.joint_names = self.ARM_JOINT_NAMES

        jv_steps = total_joint_movement / (num_waypoints + 1)
        duration_steps = total_time_taken / (num_waypoints + 1)

        for i in range(num_waypoints):
            pnt = np.array(current_jvs)
            pnt[joint_idx] += i * jv_steps

            jtp = JointTrajectoryPoint()
            jtp.positions = pnt.tolist()
            jtp.time_from_start = rospy.Duration(i * duration_steps)

            rt.joint_trajectory.points.append(jtp)

        self.execute(rt, True)

    def dynamicOpen(self):
        self.auto_grasped = False
        self.forceRobotDof(speed=-0.2, avoid_collision=True)

    def adjustBasePosition(self, pos_delta):
        final_target_hand_pose = self.getArmTipPose()
        final_target_hand_pose.position.x += pos_delta[0]
        final_target_hand_pose.position.y += pos_delta[1]

        current_spread = self.getCurrentSpread()
        while not np.allclose(pose2PosOri(final_target_hand_pose)[0][:2], pose2PosOri(self.getArmTipPose())[0][:2],
                              atol=1e-3, rtol=1e-3):
            self.liftUntilSafe(current_spread)
            current_hand_pose = self.getHandBasePose()
            final_target_hand_pose.position.z = current_hand_pose.position.z
            dirvec = pose2PosOri(final_target_hand_pose)[0] - pose2PosOri(current_hand_pose)[0]

            unit_dirvec = dirvec / np.linalg.norm(dirvec)
            if np.linalg.norm(dirvec) > 0.02:
                target_dirvec = unit_dirvec * 0.02
            else:
                target_dirvec = dirvec
            current_hand_pose.position.x += target_dirvec[0]
            current_hand_pose.position.y += target_dirvec[1]
            current_hand_pose.orientation = self.getArmTipPose().orientation
            a = raw_input('continue...')
            if a == 'y':
                break
            self.setHandPose(hand_pose=current_hand_pose, post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                             spread=0, virtual=True, cartesian=True, skip_execution=False)

    def liftUntilSafe(self, current_spread):
        lifted = False
        while max(self.getForces()['efforts']) >= 0.05 or not np.allclose(current_spread, self.getCurrentSpread(),
                                                                          atol=2e-2, rtol=2e-2):
            self.lift_a_bit()
            lifted = True
        if lifted:
            raw_input('continue...')

    def adjustHandOrientation(self, ori_rad, safe_lift=True):
        original_wrist_joint_value = self.get_wrist_value()
        current_wrist_joint_value = self.get_wrist_value()

        target_value = float(original_wrist_joint_value + ori_rad)
        target_value -= np.pi / 2

        current_spread = self.getCurrentSpread()
        while True:
            if not np.allclose(original_wrist_joint_value, current_wrist_joint_value, atol=.001, rtol=.001):
                break
            target_value = random.random() * 2 * math.pi - math.pi
            while target_value >= 2 * math.pi:
                target_value -= 2 * math.pi
            while target_value <= -2 * math.pi:
                target_value += 2 * math.pi

            if safe_lift:
                while not np.allclose(self.get_wrist_value(), target_value, atol=2e-2, rtol=2e-2):
                    self.liftUntilSafe(current_spread)
                    a = raw_input('continue... ori')
                    if a == 'y':
                        break
                    while a == 'n':
                        self.lift_a_bit()
                        a = raw_input('continue... ori')

                    current_joint_value = self.get_wrist_value()
                    joint_values = self.arm_commander.get_current_joint_values()
                    joint_values[-1] = current_joint_value + np.clip(target_value - current_joint_value, -0.1, 0.1)

                    target_arm_tip_pose = self.compute_fk_client(self.arm_commander, joint_values, [self.ARM_TIP_LINK])[
                        0].pose
                    final_target_hand_pose = self.getHandBasePoseFromArmTipPose(target_arm_tip_pose)

                    self.setHandPose(hand_pose=final_target_hand_pose, post_point_cloud=False, finger_1=0, finger_2=0,
                                     finger_3=0, spread=0, virtual=True, cartesian=True, skip_execution=False)
            else:
                joint_values = self.arm_commander.get_current_joint_values()
                joint_values[-1] = target_value
                target_arm_tip_pose = self.compute_fk_client(self.arm_commander, joint_values, [self.ARM_TIP_LINK])[
                    0].pose
                hand_pose = self.getHandBasePoseFromArmTipPose(target_arm_tip_pose)
                self.setHandPose(hand_pose=hand_pose, post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                                 spread=0, virtual=True, cartesian=True, skip_execution=False)

            current_wrist_joint_value = self.arm_commander.get_current_joint_values()[-1]
            print(original_wrist_joint_value, '->', target_value, '==', current_wrist_joint_value)

    def get_wrist_value(self):
        return self.arm_commander.get_current_joint_values()[-1]

    def getArmTipPose(self):
        return self.getLinkPose(target_frame=self.WORLD_LINK, source_frame=self.ARM_TIP_LINK)

    def getLinkPose(self, source_frame, target_frame):
        while True:
            try:
                transform = self.tfBuffer.lookup_transform(target_frame=target_frame, source_frame=source_frame,
                                                           time=rospy.Time.now(), timeout=rospy.Duration(3.0)).transform
                return self.transform_to_pose(transform)
            except:
                pass

    def getSafeZHandPose(self, hand_pose):
        hand_base_pos, ori = pose2PosOri(hand_pose)

        robot_dirvec = qua_to_dirvec(ori[3], ori[0], ori[1], ori[2])

        if hand_base_pos[2] < g.safe_real_z:
            scale = (g.safe_real_z - hand_base_pos[2]) / robot_dirvec[2]
            hand_base_pos += scale * robot_dirvec
        return vector2GeometryMsgs(hand_base_pos, ori)

    def checkNoExecute(self, hand_pose, skip_execution):
        curr_hand_base_pos, _ = pose2PosOri(self.getHandBasePose())
        if self.is_master():
            print(hand_pose)
        hand_base_pos, ori = pose2PosOri(hand_pose)
        if np.allclose(curr_hand_base_pos - hand_base_pos, 0, rtol=0.12, atol=0.12) and skip_execution:
            print('no execute', 'THRESHOLD_VALUE', THRESHOLD_VALUE)
            return True
        else:
            return False

    def getHandBasePose(self):
        return self.getLinkPose(target_frame=self.WORLD_LINK, source_frame=self.HAND_BASE_LINK)

    def compute_fk_client(self, group, joint_values, links):
        rospy.wait_for_service('compute_fk')
        try:
            compute_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
            header = Header()
            header.stamp = rospy.Time.now()
            header.frame_id = group.get_pose_reference_frame()

            rs = RobotState()
            rs.joint_state.header = header
            rs.joint_state.name = group.get_active_joints()
            rs.joint_state.position = joint_values

            res = compute_fk(header, links, rs)

            return res.pose_stamped
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def pose_to_transform(self, pose):
        transform = Transform()
        transform.translation.x = pose.position.x
        transform.translation.y = pose.position.y
        transform.translation.z = pose.position.z
        transform.rotation = pose.orientation
        return transform

    def transform_to_pose(self, transform):
        pose = Pose()
        pose.position.x = transform.translation.x
        pose.position.y = transform.translation.y
        pose.position.z = transform.translation.z
        pose.orientation = transform.rotation
        return pose

    def clearWorld(self):
        pass

    def toggleAllCollisions(self, collision):
        pass

    def loadWorld(self):
        pass

    def importObstacle(self, obstacle_name, pose=None):
        pass

    def importRobot(self, robot_name, pose=None, use_fixed_base=True, novel=None):
        return 0

    def getHandBasePoseFromArmTipPose(self, arm_tip_pose):
        arm_tip_pos, arm_tip_ori = pose2PosOri(arm_tip_pose)
        arm_tip_dirvec = qua_to_dirvec(arm_tip_ori[3], arm_tip_ori[0], arm_tip_ori[1], arm_tip_ori[2])
        hand_base_pos = arm_tip_pos + arm_tip_dirvec * self.TIP_LINK_TO_GRIPPER_BASE_CM
        return vector2GeometryMsgs(hand_base_pos, arm_tip_ori)

    def postTactileCartesian(self, cartesian, name):
        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.WORLD_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = Pose(Point(*cartesian), Quaternion(0, 0, 0, 1))
        world_manager.world_manager_client.add_tf(name, gripper_pose_stamped)

    def post_fk_tf(self, pose):
        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.WORLD_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = pose
        world_manager.world_manager_client.add_tf('FK', gripper_pose_stamped)
        return gripper_pose_stamped

    def post_grasp_tf(self, pose):
        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.WORLD_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = pose
        world_manager.world_manager_client.add_tf('Grasp', gripper_pose_stamped)
        return gripper_pose_stamped

    def planHandBasePose(self, pose, stop=False):
        self.arm_commander.clear_pose_targets()
        gripper_pose_stamped = self.post_grasp_tf(pose)

        input = False
        while input != 'y':
            pos, ori = pose2PosOri(pose)
            print(pos)
            if self.get_ik(pos, ori, self.HAND_BASE_LINK) is None:
                print('ik does not exist')

            self.arm_commander.set_pose_target(gripper_pose_stamped)
            result = self.arm_commander.plan()
            if len(result.joint_trajectory.points) == 0:
                raise NotImplementedError
            if len(result.joint_trajectory.points) > (70 - 10 * g.eval_single_object) and self.is_master():
                continue

            end_position = result.joint_trajectory.points[-1].positions
            target_arm_tip_pose = self.compute_fk_client(self.arm_commander, end_position, [self.HAND_BASE_LINK])[
                0].pose
            self.post_fk_tf(target_arm_tip_pose)
            intended_pos, intended_ori = pose2PosOri(pose)
            actual_pos, actual_ori = pose2PosOri(target_arm_tip_pose)
            if not np.allclose(intended_pos, actual_pos, atol=1e-3, rtol=1e-3):
                print(intended_pos, actual_pos)
                raise NotImplementedError
            if not np.allclose(intended_ori, actual_ori, atol=1e-3, rtol=1e-3) and not np.allclose(-intended_ori,
                                                                                                   actual_ori,
                                                                                                   atol=1e-3,
                                                                                                   rtol=1e-3):
                print(intended_ori, actual_ori)
                raise NotImplementedError

            if self.is_master():
                input = raw_input('setHandPose before execute %s' % len(result.joint_trajectory.points))
            else:
                input = 'y'

        return result

    def planArmTipPose(self, pose, stop=False):
        gripper_pose_stamped = self.post_grasp_tf(pose)

        input = False
        while input != 'y':
            self.arm_commander.set_pose_target(gripper_pose_stamped)
            result = self.arm_commander.plan()
            if len(result.joint_trajectory.points) == 0:
                raise NotImplementedError

            if self.is_master():
                input = raw_input('setHandPose before execute')
            else:
                input = 'y'

        return result

    def post_point_cloud(self):
        if self.is_master():
            if hasattr(self, 'rgbd'):
                print('posting point cloud')
                self.process_and_save_pcl_as_mesh(cloud_points=self.rgbd.point_cloud_array,
                                                  mesh_filepath=os.path.abspath('./test_2.ply'))
            else:
                raise NotImplementedError
        else:
            print('posting point cloud slave')
            self.process_and_save_pcl_as_mesh(cloud_points=None, mesh_filepath=os.path.abspath('./test_2.ply'))

    def getRobotState(self):
        return self.robot_commander.get_current_state()

    def needToMoveFingers(self, finger_1, finger_2, finger_3, finger_4):
        return finger_1 is not None and finger_2 is not None and finger_3 is not None

    def setHandPose(self, hand_pose, spread=None, finger_1=None, finger_2=None, finger_3=None, finger_4=None,
                    skip_execution=True, post_point_cloud=True, virtual=False, stop=False, execute=True,
                    get_safe_finger_positions=False, cartesian=False, wait=True, robot_state=None):
        if self.checkNoExecute(hand_pose, skip_execution):
            return

        if self.is_master():
            print(hand_pose)

        hand_base_pos, ori = pose2PosOri(self.getSafeZHandPose(hand_pose))

        robot_dirvec = qua_to_dirvec(ori[3], ori[0], ori[1], ori[2])

        arm_tip_pos = hand_base_pos - robot_dirvec * self.TIP_LINK_TO_GRIPPER_BASE_CM

        if robot_state is None:
            robot_state = self.robot_commander.get_current_state()
        new_position = list(robot_state.joint_state.position)
        if spread is not None:
            new_position[6] = spread
            new_position[9] = spread
        if self.needToMoveFingers(finger_1, finger_2, finger_3, finger_4):
            new_position[7] = finger_1
            new_position[8] = finger_1 * 0.84 / 2.44
            new_position[10] = finger_2
            new_position[11] = finger_2 * .84 / 2.44
            new_position[12] = finger_3
            new_position[13] = finger_3 * .84 / 2.44
        robot_state.joint_state.position = tuple(new_position)
        self.arm_commander.set_start_state(robot_state)

        if cartesian:
            result, _ = self.arm_commander.compute_cartesian_path(
                waypoints=[Pose(Point(*arm_tip_pos), Quaternion(*ori))], eef_step=0.01, jump_threshold=0)
        else:
            result = self.planArmTipPose(Pose(Point(*arm_tip_pos), Quaternion(*ori)), stop)

        if not execute:
            return (hand_pose, 0, spread, finger_1, finger_2, finger_3)

        if spread is not None and not virtual:
            self.forceSpread(spread)
            time.sleep(.5)
        if self.needToMoveFingers(finger_1, finger_2, finger_3, finger_4) and not virtual:
            print('CLOSE_HAND_VELOCITY %s' % str([finger_1, finger_2, finger_3]))
            hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions', bhand_controller.srv.CustomActions)
            hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY, [finger_1, finger_2, finger_3, 0.2])

        if get_safe_finger_positions:
            joint_values = dict(zip(self.HAND_JOINTS, self.hand_commander.get_current_joint_values()))
            self.correct_safe_close_position = self.get_safe_finger_joints(joint_values)
        self.execute(result, wait)
        self.cache_rqt()

    def setRobotPose(self, pose, id=0):
        pass

    def importGraspableBody(self, body_name, pose=None):
        return 1

    def publish_point_cloud(self, point_cloud):
        print('min z = ', np.min(np.array(point_cloud)[:, 2]))
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        debug_pointcloud = pcl2.create_cloud_xyz32(header, point_cloud)
        debug_pointcloud.header.frame_id = self.WORLD_LINK

        self.cloud_pub.publish(debug_pointcloud)

    def findInitialContact(self, pose):
        self.setHandPose(pose, skip_execution=False)

    def fill_depth_with_bilinear_interpolation(self, depth, original_y, original_x):
        h, w, _ = depth.shape
        low_x = original_x
        while low_x > 0 and depth[original_y, low_x, 0] == 0:
            low_x -= 1
        high_x = original_x
        while high_x < w - 1 and depth[original_y, high_x, 0] == 0:
            high_x += 1
        low_y = original_y
        while low_y > 0 and depth[low_y, original_x, 0] == 0:
            low_y -= 1
        high_y = original_y
        while high_y > h - 1 and depth[high_y, original_x, 0] == 0:
            high_y += 1

        low_y_low_x = depth[low_y, low_x, 0]
        high_y_low_x = depth[high_y, low_x, 0]
        low_y_high_x = depth[low_y, high_x, 0]
        high_y_high_x = depth[high_y, high_x, 0]
        try:
            return bilinear_interpolation(original_x, original_y,
                                          points=[[low_x, low_y, low_y_low_x], [high_x, low_y, low_y_high_x],
                                                  [low_x, high_y, high_y_low_x], [high_x, high_y, high_y_high_x]])
        except:
            return np.mean([low_y_low_x, low_y_high_x, high_y_low_x, high_y_high_x])

    def getImageDisk(self, host, port, get_rgb=False):
        w, h = int(d.camera_w), int(d.camera_h)
        self.rgbd = kinect_rgbd_standalone.get_rgbd()
        depth = self.rgbd.depth_image[:, :, np.newaxis] / 1000.0
        assert depth.shape == (h, w, 1), (depth.shape, [h, w, 1])
        max = np.max(depth)
        fixed_depth = np.zeros_like(depth)
        for i in range(h):
            for j in range(w):
                if depth[i, j, 0] == 0:
                    fixed_depth[i, j, 0] = max if (
                            float(i) / h < 0.2 or float(i) / h > 0.8 or float(j) / w < 0.2 or float(
                        j) / w > 0.8) else self.fill_depth_with_bilinear_interpolation(depth, i, j)
                else:
                    fixed_depth[i, j, 0] = depth[i, j, 0]
        point_cloud = get_point_clouds_from_depth(fixed_depth, w=self.w, h=self.h)
        self.publish_point_cloud(point_cloud)

        return fixed_depth, self.rgbd.rgb_image

    def findInitialContact(self, hand_pose):
        self.setHandPose(hand_pose)

    def check_hand_collision(self):
        pass

    def rememberRobotPose(self):
        pass

    def move_object_to_bin(self):
        self.setHandPose(get_pose(*g.real_home), post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0, finger_4=0,
                         spread=0, virtual=True)

    def move_object_to_bin_cartesian(self):
        current_tip_position = self.getArmTipPose().position
        position = [current_tip_position.x, current_tip_position.y]
        while not np.allclose(position, g.real_home[:2], atol=2e-2, rtol=2e-2):
            current_tip_pose = self.getArmTipPose()
            current_tip_pose.position.x = -0.5
            self.setHandPose(current_tip_pose, post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0, spread=0,
                             virtual=True, cartesian=True, skip_execution=False)
            self.adjustHandOrientation((random.random() - 0.5) * np.pi, safe_lift=False)

            previous_base_position = pose2PosOri(self.getHandBasePose())[0]
            self.setHandPose(get_pose(*g.real_home), post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                             spread=0, virtual=True, cartesian=True)
            if np.allclose(previous_base_position, pose2PosOri(self.getHandBasePose())[0], atol=1e-3, rtol=1e-3):
                self.setHandPose(get_pose(*g.real_home), post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                                 spread=0, virtual=True, cartesian=False)
            current_tip_position = self.getArmTipPose().position
            position = [current_tip_position.x, current_tip_position.y]
            print('cartesian', position, g.real_home[:2])

    def incrementalLift(self):
        self.auto_grasped = False
        self.autoGrasp(slow=True)
        raw_input('before lift')
        self.lift()
        return True

    def computeIncrementalQuality(self, delete):
        pass

    def lift(self):
        joints = self.hand_commander.get_joints()[1:]
        print(dict(zip(joints, self.hand_commander.get_current_joint_values())))
        target_tip_pose = self.getArmTipPose()
        done = False
        x = target_tip_pose.position.x
        z = target_tip_pose.position.z
        target_tip_pose.position.z = z + self.LIFT_HEIGHT
        while not done:
            print(target_tip_pose)
            try:
                self.keep_grasping = True
                self.setHandPose(target_tip_pose, post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                                 finger_4=0, spread=0, virtual=True, cartesian=True)
                done = True
            except:
                pass
            target_tip_pose.position.x = x + random.uniform(0.1, 0.15)
            target_tip_pose.position.z = z + random.uniform(0.05, 0.20)

        if g.eval_single_object:
            self.keep_grasping = False
            self.forceRobotDof()
            self.move_object_to_bin_cartesian()
        else:
            self.move_object_to_bin_cartesian()
            self.keep_grasping = False
            self.forceRobotDof()
            if g.hand == 'seed':
                self.slow_grasping = True

    def forceRobotDof(self, dofs=None, speed=-.8, avoid_collision=False):
        old_joint_values = np.array(self.hand_commander.get_current_joint_values())[[1, 4, 6]]
        while True:
            do_random = 1 - reduce(operator.mul, np.greater_equal(old_joint_values, np.array(
                self.hand_commander.get_current_joint_values())[[1, 4, 6]] + 0.0002), 1)

            real_speed = speed - random.random() * do_random
            if reduce(operator.mul, np.less_equal(np.array(self.hand_commander.get_current_joint_values())[[1, 4, 6]],
                                                  self.OPEN_GRIP_POSITION), 1):
                break
            hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions', bhand_controller.srv.CustomActions)
            hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY, self.OPEN_GRIP_POSITION + [real_speed])
            print('trying to open', real_speed, do_random)
            old_joint_values = np.array(self.hand_commander.get_current_joint_values())[[1, 4, 6]]
            time.sleep(.15)
            if avoid_collision and (min(self.getForces()['efforts']) < -0.1):
                self.lift_a_bit()
                hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions',
                                                        bhand_controller.srv.CustomActions)
                hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY, self.OPEN_GRIP_POSITION + [speed])

        self.cache_rqt()

    def computeQuality(self, delete=False):
        self.lift()
        return 0

    def forceSpread(self, spread):
        current_spread = self.getCurrentSpread()
        if np.allclose(current_spread, spread, atol=1e-2, rtol=1e-2):
            return
        time.sleep(0.2)
        spread = spread * 0.45 / np.pi
        print('forceSpread before execute %s' % spread)
        hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions', bhand_controller.srv.CustomActions)
        try:
            hand_service_proxy(bhand_controller.msg.Service.SET_SPREAD, [spread, spread])
        except:
            pass
        time.sleep(0.2)

    def autoOpen(self):
        pass

    def check_whether_finger_joint_values_are_safe_without_table(self, positions):
        robot_state = self.robot_commander.get_current_state()
        new_position = list(robot_state.joint_state.position)
        for i in range(6):
            new_position[i] = g.home_joint_position[i]
        robot_state.joint_state.position = tuple(new_position)
        finger_joint_angles = np.array(positions)
        self.hand_commander.set_start_state(robot_state)
        finger_joint_angles[[1, 3, 5]] = finger_joint_angles[[0, 2, 4]] * 0.84 / 2.44
        try:
            self.hand_commander.set_joint_value_target(
                dict(zip(self.FINGER_JOINTS + self.SPREAD_JOINTS, finger_joint_angles)))
        except:
            return False
        result = self.hand_commander.plan()
        self.hand_commander.set_start_state(self.robot_commander.get_current_state())
        self.arm_commander.set_start_state(self.robot_commander.get_current_state())
        return len(result.joint_trajectory.points) > 0

    def check_whether_finger_joint_values_are_safe(self, positions):
        finger_joint_angles = np.array(positions)
        finger_joint_angles[[1, 3, 5]] = finger_joint_angles[[0, 2, 4]] * 0.84 / 2.44
        result = self.gripper_move_group.moveToJointPosition(plan_only=True,
                                                             joints=self.FINGER_JOINTS + self.SPREAD_JOINTS,
                                                             positions=finger_joint_angles, wait=True, tolerance=1e-4)
        ret = (result and (result.error_code.val == MoveItErrorCodes.SUCCESS))
        return ret

    def get_safe_finger_joints(self, current_joint_values, finger1=True, finger2=True, finger3=True,
                               binary_search_steps=10):
        print(current_joint_values)
        max_finger_joint_angles = self.CLOSED_POSITION
        if not finger1:
            max_finger_joint_angles[0] = current_joint_values['bh_j12_joint']
            max_finger_joint_angles[1] = current_joint_values['bh_j13_joint']
        if not finger2:
            max_finger_joint_angles[2] = current_joint_values['bh_j22_joint']
            max_finger_joint_angles[3] = current_joint_values['bh_j23_joint']
        if not finger3:
            max_finger_joint_angles[4] = current_joint_values['bh_j32_joint']
            max_finger_joint_angles[5] = current_joint_values['bh_j33_joint']
        min_finger_joint_angles = np.clip(np.array([current_joint_values[i] for i in self.FINGER_JOINTS]), 0, None)
        safe_finger_joint_positions = min_finger_joint_angles.tolist() + [current_joint_values['bh_j11_joint'],
                                                                          current_joint_values['bh_j21_joint']]

        # binary search to self-collision free client posture
        for i in range(binary_search_steps):
            finger_joint_angles = max_finger_joint_angles - (max_finger_joint_angles - min_finger_joint_angles) / 2
            finger_joint_angles[[1, 3, 5]] = finger_joint_angles[[0, 2, 4]] * 0.84 / 2.44

            positions = finger_joint_angles.tolist() + [current_joint_values['bh_j11_joint'],
                                                        current_joint_values['bh_j21_joint']]
            start = time.time()
            if self.check_whether_finger_joint_values_are_safe(positions):
                safe_finger_joint_positions = positions
                min_finger_joint_angles = finger_joint_angles.copy()
                print('min_finger_joint_angles', min_finger_joint_angles)
            else:
                max_finger_joint_angles = finger_joint_angles.copy()
        return safe_finger_joint_positions

    def autoGrasp(self, slow=False):
        self.scene.removeCollisionObject('scene_mesh')
        joint_values = dict(zip(self.HAND_JOINTS, self.hand_commander.get_current_joint_values()))
        safe_close_position = self.get_safe_finger_joints(joint_values)

        if not g.video:
            raw_input('autoGrasp before execute: %s ' % safe_close_position)

        hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions', bhand_controller.srv.CustomActions)
        if slow:
            no_touching = reduce(operator.mul, self.getForces()['efforts'] < 0.2, 1)
            velocities = 0.02 + no_touching * 0.15
            hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY,
                               [safe_close_position[0], safe_close_position[2], safe_close_position[4], velocities])
            print('finger velocities', velocities, no_touching)
        else:
            hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY,
                               [safe_close_position[0], safe_close_position[2], safe_close_position[4], 0.2])

    def readTactile(self):
        raw_data = list(self.tact_data_queue.queue)
        ret = [np.array(list(tactile.palm) + list(tactile.finger1) + list(tactile.finger2) + list(tactile.finger3)) for
               tactile in raw_data]

        return np.mean(ret, axis=0)

    def getEfforts(self):
        return np.array([self.joint_state_pointer[i]['values'][2] for i in ['F1_TIP', 'F2_TIP', 'F3_TIP']])

    def getForces(self):
        if self._tact_data is not None:
            current_contacts = (
                    (self.readTactile() - self.baseline_tactile_contacts) > g.real_tactile_threshold).astype(np.int64)
        else:
            current_contacts = np.zeros(self.num_tactile_links)
        ret = {'current_contacts': current_contacts}
        ret['efforts'] = self.getEfforts() - self.baseline_efforts
        if g.tactile_positions:
            ret['current_positions'] = self.getTactileJointPositions()
        if g.tactile_velocities:
            raise NotImplementedError
        if g.tactile_forces:
            raise NotImplementedError
        if g.tactile_cartesians:
            ret['current_cartesians'] = self.getTactileCartesianPositions(ret['current_contacts'])
        return ret

    def getWristReactionForces(self):
        return np.zeros(3)

    def getWristReactionTorques(self):
        return np.zeros(3)

    def getTactileJointPositions(self):
        return np.array([self.joint_state_pointer[i]['values'][0] for i in
                         ['F1', 'F2', 'F3', 'F1_TIP', 'F2_TIP', 'F3_TIP', 'SPREAD_1', 'SPREAD_2']])

    def getCurrentHandPosition(self):
        translation = self.tfBuffer.lookup_transform(target_frame=self.WORLD_LINK, source_frame=self.HAND_BASE_LINK,
                                                     time=rospy.Time.now(),
                                                     timeout=rospy.Duration(1.0)).transform.translation
        return np.array([translation.x, translation.y, translation.z])

    def getTactileCartesianPositions(self, contacts):
        try:
            ret = []
            ee_position = self.getCurrentHandPosition()

            for link_index in range(len(self.tactile_link_names)):
                link = self.tactile_link_names[link_index]
                for i in range(24):
                    link_name = 'bh_%s_sensor%s%s_link' % (link, '_' if link == 'palm' else '', i + 1)
                    if contacts[link_index * 24 + i]:
                        translation = self.tfBuffer.lookup_transform(target_frame=self.WORLD_LINK,
                                                                     source_frame=link_name, time=rospy.Time.now(),
                                                                     timeout=rospy.Duration(1.0)).transform.translation
                        ret.append([translation.x, translation.y, translation.z])
                    else:
                        ret.append(ee_position)

            ret = np.array(ret) - ee_position
            cartesians = (ret * contacts[:, np.newaxis])
            return cartesians.flatten()
        except:
            traceback.print_exc()
            ipdb.set_trace()

    def cache_rqt(self):
        self.rqt_pub.publish('1')
        self.baseline_efforts = self.getEfforts()
        self.baseline_tactile_contacts = self.readTactile()

    def is_master(self):
        return self.id == 0

    def get_ik(self, position, orientation, ik_link_name):
        try:
            gripper_pose_stamped = PoseStamped()
            gripper_pose_stamped.header.frame_id = self.WORLD_LINK
            gripper_pose_stamped.header.stamp = rospy.Time.now()
            gripper_pose_stamped.pose = Pose(Point(*position), Quaternion(*orientation))

            service_request = PositionIKRequest()
            service_request.group_name = self.ARM_GRIPPER
            service_request.ik_link_name = ik_link_name
            service_request.pose_stamped = gripper_pose_stamped
            service_request.timeout.secs = self.PLANNING_TIME_LIMIT
            service_request.avoid_collisions = True
            resp = self.compute_ik(ik_request=service_request)
            return resp
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def get_transformation_matrix(self, source_frame, target_frame):
        return ros_numpy.numpify(self.getLinkPose(source_frame, target_frame))
