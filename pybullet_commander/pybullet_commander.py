import gc
import json
import math
import os
import pybullet as p
import random
import traceback

import geometry_msgs
import matplotlib.pyplot as plt
import numpy as np
import pybullet_data
from matplotlib.font_manager import FontProperties
from pyquaternion import Quaternion

from config import d, g
from utils import pose2PosOri, qua_to_dirvec, quaternion_to_directional_vector_ori, wxyz_to_xyzw, xyzw_to_wxyz


class PybulletCommander(object):
    PROXIMAL_TIP_RATIO = None
    tactile_link_names = None

    def __init__(self, test):
        self.full_cartesians_queue = None
        self.custom_scale = None
        self.lift_hand_height = None
        self.target_grasp_velocities = None
        self.lift_arm_joint_names = None
        self.dampingRatio = None
        self.arm_joint_names = None
        self.grip_joint_names = None
        self.finger_joint_names = None
        self.spread_joint_names = None
        self.palm_joint_names = None
        self.end_effector_link_name = None
        self.auto_grasp_num_iters = None
        self.test = test
        self.connect()
        self.robots = {}
        self.graspable_bodies = {}
        self.obstacles = {}
        self.grasp_time = 1
        self.gravity = 9.8
        self.far = 1000
        self.near = 0.01
        self.aspect = d.camera_w / d.camera_h
        self.lift_obj_height = 0.25
        self.arm_joints_position_gains_during_lift_factor = None
        self.friction_mult = 1
        p.resetDebugVisualizerCamera(cameraDistance=d.camera_dist, cameraYaw=d.camera_yaw_deg,
                                     cameraPitch=d.camera_pitch_deg, cameraTargetPosition=[0, 0, 0])
        if g.hand == 'barrett':
            self.width60, self.height60, self.viewMat60, self.projMat60, self.cameraUp60, self.camForward60, \
            self.horizon60, self.vertical60, _, _, self.dist60, self.camTarget60 = (
                1024, 768, (
                    1.0, 0.0, -0.0, 0.0, -0.0, 0.9998477101325989, -0.0174524188041687, 0.0, 0.0, 0.0174524188041687,
                    0.9998477101325989, 0.0, -0.0, 1.862645149230957e-09, -1.2000000476837158, 1.0), (
                    0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0,
                    0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0174524188041687, -0.9998477101325989),
                (26666.66796875, -0.0, 0.0), (0.0, 19996.953125, 349.0483703613281), 0.0, -89.0, 1.2000000476837158,
                (0.0, 0.0, 0.0))
            self.width, self.height, self.viewMat, self.projMat, self.cameraUp, self.camForward, self.horizon, \
            self.vertical, _, _, self.dist, self.camTarget = (
                1024, 768, (
                    1.0, 0.0, -0.0, 0.0, -0.0, 0.9998477101325989, -0.0174524188041687, 0.0, 0.0, 0.0174524188041687,
                    0.9998477101325989, 0.0, -0.0, 1.862645149230957e-09, -1.2000000476837158, 1.0), (
                    0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0,
                    0.0, -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0174524188041687, -0.9998477101325989),
                (26666.66796875, -0.0, 0.0), (0.0, 19996.953125, 349.0483703613281), 0.0, -89.0, 1.2000000476837158,
                (0.0, 0.0, 0.0))
        else:
            self.width60, self.height60, self.viewMat60, self.projMat60, self.cameraUp60, self.camForward60, \
            self.horizon60, self.vertical60, _, _, self.dist60, self.camTarget60 = (
                1024, 768, (
                    1.0, 0.0, -0.0, 0.0, -0.0, 0.8660253286361694, -0.4999999701976776, 0.0, 0.0, 0.4999999701976776,
                    0.8660253286361694, 0.0, -0.0, -0.0, -0.4999999701976776, 1.0), (
                    0.7412109375, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
                    -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (0.0, 0.4999999701976776, -0.8660253286361694),
                (26982.873046875, -0.0, 0.0), (0.0, 17320.5078125, 10000.0), 0.0, -60.0, 0.5, (0.0, 0.0, 0.0))

            self.width, self.height, self.viewMat, self.projMat, self.cameraUp, self.camForward, self.horizon, \
            self.vertical, _, _, self.dist, self.camTarget = (
                1024, 768, (
                    1.0, 0.0, -0.0, 0.0, -0.0, 0.8660253286361694, -0.4999999701976776, 0.0, 0.0, 0.4999999701976776,
                    0.8660253286361694, 0.0, -0.0, -0.0, -0.4999999701976776, 1.0), (
                    0.7412109375, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
                    -0.02000020071864128, 0.0), (0.0, 0.0, 1.0), (0.0, 0.4999999701976776, -0.8660253286361694),
                (26982.873046875, -0.0, 0.0), (0.0, 17320.5078125, 10000.0), 0.0, -60.0, 0.5, (0.0, 0.0, 0.0))

        self.custom_init()
        self.num_spread_joints = len(self.spread_joint_names)

    def custom_init(self):
        raise NotImplementedError

    def getRayFromTo(self, mouseX, mouseY, depth):
        camPos = [self.camTarget[0] - self.dist * self.camForward[0],
                  self.camTarget[1] - self.dist * self.camForward[1],
                  self.camTarget[2] - self.dist * self.camForward[2]]
        farPlane = 10000
        rayForward = [(self.camTarget[0] - camPos[0]), (self.camTarget[1] - camPos[1]), (self.camTarget[2] - camPos[2])]
        lenFwd = math.sqrt(
            rayForward[0] * rayForward[0] + rayForward[1] * rayForward[1] + rayForward[2] * rayForward[2])
        invLen = farPlane * 1. / lenFwd
        rayForward = [invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]]
        rayFrom = camPos
        oneOverWidth = float(1) / float(self.width)
        oneOverHeight = float(1) / float(self.height)

        dHor = [self.horizon[0] * oneOverWidth, self.horizon[1] * oneOverWidth, self.horizon[2] * oneOverWidth]
        dVer = [self.vertical[0] * oneOverHeight, self.vertical[1] * oneOverHeight, self.vertical[2] * oneOverHeight]
        ortho = [- 0.5 * self.horizon[0] + 0.5 * self.vertical[0] + float(mouseX) * dHor[0] - float(mouseY) * dVer[0],
                 - 0.5 * self.horizon[1] + 0.5 * self.vertical[1] + float(mouseX) * dHor[1] - float(mouseY) * dVer[1],
                 - 0.5 * self.horizon[2] + 0.5 * self.vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]]

        rayTo = [rayFrom[0] + rayForward[0] + ortho[0], rayFrom[1] + rayForward[1] + ortho[1],
                 rayFrom[2] + rayForward[2] + ortho[2]]
        lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] + ortho[2] * ortho[2])
        alpha = math.atan(lenOrtho / farPlane)
        depth /= math.cos(alpha)
        rf = np.array(rayFrom)
        rt = np.array(rayTo)
        vec = rt - rf
        l = np.sqrt(np.dot(vec, vec))
        newTo = (depth / l) * vec + rf
        return np.array(newTo)

    def save_state_disk(self, filename):
        print(filename)
        p.saveWorld(os.path.join(d.pybullet_env_dir, filename + ".world"))
        p.saveBullet(os.path.join(d.pybullet_env_dir, filename + ".bullet"))
        with open(os.path.join(d.pybullet_env_dir, filename + '.json'), 'w') as f:
            json.dump(self.graspable_bodies, f)

    def restore_state(self, filename):
        p.restoreState(fileName=os.path.join(d.pybullet_env_dir, filename + ".bullet"))

    def connect(self):
        if self.test:
            p.connect(p.DIRECT if g.statistics else p.GUI)
        elif 'localhost' in g.servers:
            p.connect(p.DIRECT)
        else:
            raise NotImplementedError

    def loadWorld(self):
        if g.statistics or not self.test:
            self.connect()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.gravity)
        p.resetDebugVisualizerCamera(cameraDistance=d.camera_dist, cameraYaw=d.camera_yaw_deg,
                                     cameraPitch=d.camera_pitch_deg, cameraTargetPosition=[0, 0, 0])

    def clearAllGraspableBodies(self):
        _ = [p.removeBody(body) for body in self.graspable_bodies]
        self.graspable_bodies = {}

    def clearWorld(self, error=None):
        p.resetSimulation()
        if g.statistics or not self.test:
            p.disconnect()
        gc.collect()
        self.robots = {}
        self.graspable_bodies = {}
        self.obstacles = {}

    def vector2GeometryMsgs(self, position, orientation):
        pose = geometry_msgs.msg.Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
        return pose

    def randomize_joints(self, joints):
        for joint in joints:
            p.resetJointState(self.id, joint, targetValue=random.uniform(self.joint_lower_limits[joint],
                                                                         self.joint_upper_limits[joint]),
                              targetVelocity=0)

    def importRobot(self, robot_name, pose=None, use_fixed_base=True, novel=False):
        self.id = self.importObject(robot_name, pose, use_fixed_base)
        self.robots[self.id] = robot_name
        self.joint_name2index = {}
        robot_lateral_friction = self.friction_mult * (
            g.robot_lateral_friction_novel if novel else g.robot_lateral_friction_seen)
        p.changeDynamics(self.id, -1, mass=50, linearDamping=.1, angularDamping=.1,
                         lateralFriction=robot_lateral_friction)

        self.free_joints = []
        self.free_joints_names = []
        self.joint_lower_limits = {}
        self.joint_lower_limits_list = []
        self.joint_upper_limits_list = []
        self.joint_ranges_list = []
        self.joint_upper_limits = {}
        self.all_joints = range(p.getNumJoints(self.id))
        self.all_joints_names = []
        for joint in self.all_joints:
            p.changeDynamics(self.id, joint, mass=1, linearDamping=self.dampingRatio, angularDamping=self.dampingRatio,
                             spinningFriction=10000, rollingFriction=10000, lateralFriction=robot_lateral_friction)
            joint_info = p.getJointInfo(bodyUniqueId=self.id, jointIndex=joint)
            self.all_joints_names.append(joint_info[1])
            self.joint_name2index[joint_info[1]] = joint
            if joint_info[2] != p.JOINT_FIXED:
                self.free_joints.append(joint)
                self.free_joints_names.append(joint_info[1])
                self.joint_lower_limits[joint_info[0]] = joint_info[8]
                self.joint_upper_limits[joint_info[0]] = joint_info[9]
                self.joint_lower_limits_list.append(joint_info[8])
                self.joint_upper_limits_list.append(joint_info[9])
                self.joint_ranges_list.append(joint_info[9] - joint_info[8])
        self.joint_lower_limits_list = self.joint_lower_limits_list
        self.joint_upper_limits_list = self.joint_upper_limits_list
        self.joint_rest_poses_list = (
                (np.array(self.joint_upper_limits_list) + np.array(self.joint_lower_limits_list)) / 2).tolist()

        self.ee_link = self.joint_name2index[self.end_effector_link_name]
        self.arm_joints = [self.joint_name2index[joint] for joint in self.arm_joint_names]
        self.lift_arm_joints = [self.joint_name2index[joint] for joint in self.lift_arm_joint_names]
        self.grip_joints = [self.joint_name2index[joint] for joint in self.grip_joint_names]
        self.finger_joints = [self.joint_name2index[joint] for joint in self.finger_joint_names]
        self.spread_joints = [self.joint_name2index[joint] for joint in self.spread_joint_names]
        self.palm_joints = [self.joint_name2index[joint] for joint in self.palm_joint_names]
        self.hand_joints = [self.ee_link] + self.grip_joints + self.finger_joints + self.spread_joints
        self.open_joint_names = self.grip_joint_names + self.finger_joint_names + self.spread_joint_names
        self.open_joints = self.grip_joints + self.finger_joints + self.spread_joints

        self.grasp_joints = self.grip_joints + self.finger_joints
        self.grasp_joint_names = self.grip_joint_names + self.finger_joint_names
        self.num_grasp_joints = len(self.grasp_joints)
        for joint in self.all_joints:
            p.enableJointForceTorqueSensor(bodyUniqueId=self.id, jointIndex=joint, enableSensor=True)

        self.non_grasp_joints = self.spread_joints + self.arm_joints
        self.non_grasp_joint_names = self.spread_joint_names + self.arm_joint_names
        self.num_non_grasp_joints = len(self.non_grasp_joints)
        self.lift_arm_joints_indices_in_free_joints = [self.free_joints.index(joint) for joint in self.lift_arm_joints]
        self.grasp_joints_position_gains_during_grasp = np.ones_like(self.grasp_joints)
        self.grasp_joints_velocity_gains_during_grasp = np.zeros_like(self.grasp_joints)
        self.non_grasp_joints_position_gains_during_grasp = np.ones_like(self.non_grasp_joints)
        self.non_grasp_joints_velocity_gains_during_grasp = np.zeros_like(self.non_grasp_joints)
        self.non_grasp_target_joint_velocities_during_grasp = np.zeros_like(self.non_grasp_joints)
        self.non_grasp_joint_forces_during_grasp = float('inf') * np.ones_like(self.non_grasp_joints)
        self.grasp_joint_forces_during_grasp = g.hand_mass * g.max_grasp_force_mult * np.ones_like(self.grasp_joints)

        self.grasp_joints_position_gains_during_inc_grasp = np.ones_like(self.grasp_joints) * 0.01

        self.open_joint_forces = g.hand_mass * g.max_grasp_force_mult * np.ones_like(self.open_joints)
        self.open_joints_position_gains = np.ones_like(self.open_joints) * 0.01
        self.arm_target_joint_velocities_during_open = np.zeros_like(self.arm_joints)
        self.arm_joints_position_gains_during_open = np.ones_like(self.arm_joints)
        self.arm_joints_velocity_gains_during_open = np.zeros_like(self.arm_joints)
        self.arm_joint_forces_during_open = np.ones_like(self.arm_joints) * float('inf')

        self.arm_joints_position_gains_during_adj = np.ones_like(self.arm_joints)
        self.arm_joints_velocity_gains_during_adj = np.zeros_like(self.arm_joints)
        self.arm_joint_forces_during_adj = np.ones_like(self.arm_joints) * float('inf')
        self.grasp_target_joint_velocities_during_adj = np.zeros_like(self.grasp_joints)

        self.arm_joints_position_gains_during_lift = np.ones_like(
            self.lift_arm_joints) * self.arm_joints_position_gains_during_lift_factor
        self.arm_joints_velocity_gains_during_lift = np.zeros_like(self.lift_arm_joints)
        self.arm_joints_target_velocities_during_lift = np.ones_like(self.lift_arm_joints) * 0.0002
        self.grasp_target_joint_velocities_during_lift = [0.3] * self.num_grasp_joints
        self.grasp_joint_forces_during_lift = self.grasp_joint_forces_during_grasp
        self.arm_joint_forces_during_lift = float('inf') * np.ones_like(self.lift_arm_joints)

        for joint in self.hand_joints:
            p.changeDynamics(self.id, joint, mass=g.hand_mass, linearDamping=10 * self.dampingRatio,
                             angularDamping=10 * self.dampingRatio, lateralFriction=robot_lateral_friction)

        if g.tactile:
            self.tactile_joint_indices_dict = {}
            for link in self.tactile_link_names:
                for sensor in range(24):
                    if link not in self.tactile_joint_indices_dict:
                        self.tactile_joint_indices_dict[link] = []
                    self.tactile_joint_indices_dict[link].append(
                        self.joint_name2index['bh_%s_sensor%s_joint' % (link, sensor + 1)])

    def rememberRobotPose(self):
        self.original_joint_states = p.getJointStates(self.id, self.all_joints)
        self.original_base_pos, self.original_base_ori = p.getBasePositionAndOrientation(self.id)

    def saveState(self):
        return p.saveState()

    def restoreState(self, state_id):
        return p.restoreState(state_id)

    def getLinkPositions(self, bodyUniqueId, linkIndices):
        return [p.getLinkState(bodyUniqueId, index)[0] for index in linkIndices]

    def closestObjectName(self):
        hand_position = p.getLinkState(self.id, self.ee_link, computeForwardKinematics=1)[0]
        dists = np.linalg.norm(self.get_all_graspable_body_positions() - np.array(hand_position), axis=1)
        obj_id_index = np.argmin(dists)
        obj_id = self.graspable_bodies.keys()[obj_id_index]
        return self.graspable_bodies[obj_id]

    def getCurrentJointValues(self, joint_names):
        return [s[0] for s in p.getJointStates(self.id, [self.joint_name2index[joint] for joint in joint_names])]

    def getTactileContacts(self, joint_indices):
        return np.array(
            [float(len([i for i in p.getContactPoints(bodyA=self.id, linkIndexA=joint_index) if i[1] != i[2]]) > 0) for
             joint_index in joint_indices]) * (self.getTactileForces(joint_indices) > g.sim_tactile_threshold)

    def getTactileForces(self, joint_indices):
        return np.array(
            [sum([contact[9] for contact in p.getContactPoints(bodyA=self.id, linkIndexA=joint_index)]) for joint_index
             in joint_indices])

    def getFingersSafe(self):
        ret = []
        for j1 in self.finger_joints:
            num_closest_points = 0
            for j2 in self.finger_joints:
                if j1 != j2:
                    num_closest_points += len(p.getClosestPoints(self.id, self.id, 0.001, j1, j2))
            num_closest_points += len(p.getClosestPoints(self.id, self.id, 0.001, j1, self.ee_link))
            ret.append(float(num_closest_points == 0))
        return ret

    def incrementalGrasp(self, finger1, finger2, finger3, delta):
        assert self.graspable_bodies, self.graspable_bodies
        self.calm_joints()
        target_grasp_joint_values = np.array(self.getCurrentJointValues(self.grasp_joint_names))

        old_joint_values = np.array(self.getCurrentJointValues(self.grasp_joint_names))
        count = 0
        while True:
            f1_contact, f2_contact, f3_contact = self.getFingersSafe()
            if self.test:
                print(f1_contact, f2_contact, f3_contact, 'f1_contact, f2_contact, f3_contact')
            curr_target_grasp_joint_values = target_grasp_joint_values + [finger1 * delta * f1_contact,
                                                                          finger2 * delta * f2_contact,
                                                                          finger3 * delta * f3_contact,
                                                                          finger1 * delta * self.PROXIMAL_TIP_RATIO *
                                                                          f1_contact,
                                                                          finger2 * delta * self.PROXIMAL_TIP_RATIO *
                                                                          f2_contact,
                                                                          finger3 * delta * self.PROXIMAL_TIP_RATIO *
                                                                          f3_contact]
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.non_grasp_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.curr_non_grasp_joint_positions,
                                        targetVelocities=self.non_grasp_target_joint_velocities_during_grasp,
                                        positionGains=self.non_grasp_joints_position_gains_during_grasp,
                                        velocityGains=self.non_grasp_joints_velocity_gains_during_grasp,
                                        forces=self.non_grasp_joint_forces_during_grasp)

            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.grasp_joints,
                                        controlMode=p.POSITION_CONTROL, targetPositions=curr_target_grasp_joint_values,
                                        positionGains=self.grasp_joints_position_gains_during_inc_grasp,
                                        forces=self.grasp_joint_forces_during_grasp)

            p.stepSimulation()
            self.updateForceHistory(update_queues=False)
            current_joint_values = np.array(self.getCurrentJointValues(self.grasp_joint_names))
            if count > 10 and np.allclose(current_joint_values, old_joint_values, rtol=5e-4, atol=5e-4):
                break
            old_joint_values = current_joint_values
            count += 1

            if count > 300:
                break

        if self.test:
            print(count, 'count')
        self.updateForceHistory()
        return delta

    def updateForceHistory(self, update_queues=True):
        ret = self.getForces()
        if update_queues:
            for name in g.tactile_obv_names:
                if getattr(g, 'tactile_%s' % name):
                    if getattr(self, '%s_queue' % name).full():
                        getattr(self, '%s_queue' % name).get()
                    getattr(self, '%s_queue' % name).put(ret['current_%s' % name])
        if self.full_cartesians_queue.full():
            self.full_cartesians_queue.get()
        self.full_cartesians_queue.put(ret['current_cartesians'])

    def getTactileJointPositions(self):
        return np.array([s[0] for s in self.getTactileJointStates()])

    def getWristReactionForces(self):
        wrist_state = p.getJointState(self.id, self.ee_link)
        return wrist_state[2][:3]

    def getWristReactionTorques(self):
        wrist_state = p.getJointState(self.id, self.ee_link)
        return wrist_state[2][3:]

    def getTactileJointStates(self):
        return p.getJointStates(self.id, self.grasp_joints + self.spread_joints)

    def getForces(self):
        ret = {'current_contacts': np.concatenate(
            [self.getTactileContacts(joint_indices=self.tactile_joint_indices_dict[link]) for link in
             self.tactile_link_names], axis=0)}
        states = self.getTactileJointStates()
        if g.tactile_positions:
            ret['current_positions'] = [s[0] for s in states]
        if g.tactile_velocities:
            ret['velocities'] = [s[1] for s in states]
        if g.tactile_forces:
            wrist_state = p.getJointState(self.id, self.ee_link)
            ret['current_forces'] = wrist_state[2][:3]
            ret['current_torques'] = wrist_state[2][3:]
        if g.tactile_efforts:
            ret['efforts'] = [s[3] for s in states[:3]]
        if g.tactile_cartesians:
            ret['current_cartesians'] = self.getTactileCartesianPositions(ret['current_contacts'])
        return ret

    def getTactileCartesianPositions(self, contacts):
        ret = []
        for link in self.tactile_link_names:
            for i in range(24):
                ret.append(
                    p.getLinkState(self.id, linkIndex=self.tactile_joint_indices_dict[link][i], computeLinkVelocity=0,
                                   computeForwardKinematics=1)[0])
        ee_position = np.array(
            p.getLinkState(self.id, self.ee_link, computeLinkVelocity=0, computeForwardKinematics=1)[0])
        ret = np.array(ret)
        relative_positions = ret - ee_position
        return (relative_positions * contacts[:, np.newaxis]).flatten()

    def plot_tactile(self):
        if g.plot_tactile:
            for link_index in range(len(self.tactile_link_names)):
                plt.subplot(2, 2, link_index + 1)
                link = self.tactile_link_names[link_index]
                plt.title(link)
                setattr(self, 'tactile_%s' % link, np.array(getattr(self, 'tactile_%s' % link)))
                for sensor in range(24):
                    plt.plot(range(getattr(self, 'tactile_%s' % link).shape[0]),
                             getattr(self, 'tactile_%s' % link)[:, sensor],
                             label='%s_sensor%s_joint' % (link, sensor + 1))
                fontP = FontProperties()
                fontP.set_size('small')
                if link_index % 2 == 0:
                    plt.legend(loc='upper right', prop=fontP, bbox_to_anchor=(0, 1))
                else:
                    plt.legend(loc='upper left', prop=fontP, bbox_to_anchor=(1, 1))
            plt.show()

    def hasTableCollision(self):
        self.calm_joints()

        for _ in range(self.auto_grasp_num_iters):
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.non_grasp_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.curr_non_grasp_joint_positions,
                                        targetVelocities=self.non_grasp_target_joint_velocities_during_grasp,
                                        positionGains=self.non_grasp_joints_position_gains_during_grasp,
                                        velocityGains=self.non_grasp_joints_velocity_gains_during_grasp,
                                        forces=self.non_grasp_joint_forces_during_grasp)
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.grasp_joints,
                                        controlMode=p.VELOCITY_CONTROL, targetVelocities=self.target_grasp_velocities,
                                        forces=self.grasp_joint_forces_during_grasp)
            p.stepSimulation()

            if p.getContactPoints(bodyA=self.id, bodyB=self.tray_id):
                return True
        return False

    def avoidTrayCollision(self, lift=None):
        if g.record:
            return
        state_id = p.saveState()
        offset = 0
        while self.hasTableCollision():
            if self.test:
                print('adjusting...')
            p.restoreState(state_id)
            if g.hand == 'barrett':
                offset += 0.01
            else:
                offset += 0.002
            self.adjustBasePosition([0, 0, offset])
            if self.test:
                print('adjusted...')

        p.restoreState(state_id)
        self.adjustBasePosition([0, 0, offset])

    def autoGrasp(self):
        assert self.graspable_bodies, self.graspable_bodies
        self.closest_obj_name = self.closestObjectName()
        self.tactile_link1 = []
        self.tactile_link2 = []
        self.tactile_link3 = []
        self.tactile_palm = []
        self.calm_joints()

        for _ in range(self.auto_grasp_num_iters):
            self.calm_joints()
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.non_grasp_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.curr_non_grasp_joint_positions,
                                        targetVelocities=self.non_grasp_target_joint_velocities_during_grasp,
                                        positionGains=self.non_grasp_joints_position_gains_during_grasp,
                                        velocityGains=self.non_grasp_joints_velocity_gains_during_grasp,
                                        forces=self.non_grasp_joint_forces_during_grasp)
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.grasp_joints,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=self.target_grasp_velocities * 5,
                                        forces=self.grasp_joint_forces_during_grasp * 100)
            p.stepSimulation()
            if g.plot_tactile and self.test:
                for link in self.tactile_link_names:
                    getattr(self, 'tactile_%s' % link).append(
                        self.getTactileForces(joint_indices=self.tactile_joint_indices_dict[link]))

        self.plot_tactile()

        if self.test and not g.statistics and not g.record:
            raw_input('after grasp %s' % self.closest_obj_name)

    def autoOpen(self):
        pass

    def getHandPose(self):
        pos, ori = p.getLinkState(self.id, linkIndex=self.ee_link, computeForwardKinematics=1)[:2]
        return np.array(pos), np.array(ori)

    def getCurrentHandPosition(self):
        pos, ori = p.getLinkState(self.id, linkIndex=self.ee_link, computeForwardKinematics=1)[:2]
        return np.array(pos)

    def validate_joint_values(self, joint_values, joints):
        for i in range(len(joints)):
            if not self.joint_lower_limits[joints[i]] <= joint_values[i] <= self.joint_upper_limits[joints[i]]:
                return False
        return True

    def check_hand_collision(self):
        p.stepSimulation()
        if self.num_hand_contacts():
            self.closest_obj_name = self.closestObjectName()
            raise NotImplementedError

    def setHandPose(self, hand_pose, spread=None, pos_close=5e-3, ori_close=0.2, finger_1=None, finger_2=None,
                    finger_3=None, finger_4=None, post_point_cloud=None, execute=None, stop=None,
                    get_safe_finger_positions=False, robot_state=None):
        state = p.saveState()
        target_hand_pos, target_hand_ori = pose2PosOri(hand_pose)
        target_dir_vec = quaternion_to_directional_vector_ori(hand_pose.orientation)

        new_base_pos = np.copy(self.original_base_pos)
        c = 0

        hand_joints, hand_joint_values = self.computeHandJointValues(finger_1, finger_2, finger_3, finger_4, spread)

        curr_pos_dist, curr_ori_dist = float('inf'), float('inf')
        while curr_ori_dist > ori_close or curr_pos_dist > pos_close or (
                new_base_pos[1] > -0.2 and g.hand == 'barrett'):
            if self.test:
                print(
                curr_pos_dist > pos_close, curr_ori_dist > ori_close, (new_base_pos[1] > -0.2 and g.hand == 'barrett'))
                print(curr_pos_dist, curr_ori_dist, new_base_pos[1])

            c += 1
            while True:
                p.restoreState(state)
                self.randomize_joints(self.arm_joints)
                joint_values = p.calculateInverseKinematics(self.id, self.ee_link, targetPosition=target_hand_pos,
                                                            targetOrientation=target_hand_ori,
                                                            lowerLimits=self.joint_lower_limits_list,
                                                            upperLimits=self.joint_upper_limits_list,
                                                            jointRanges=self.joint_ranges_list,
                                                            restPoses=self.joint_rest_poses_list)
                if self.validate_joint_values(joint_values, self.free_joints):
                    break

            self.resetJointValues(joint_values=joint_values, joints=self.free_joints)
            new_base_pos = np.array(self.original_base_pos) + (target_hand_pos - self.getHandPose()[0])

            self.resetJointStates(joint_states=self.original_joint_states, joints=self.all_joints)
            p.resetBasePositionAndOrientation(self.id, new_base_pos, self.original_base_ori)
            self.resetJointValues(joint_values=joint_values, joints=self.free_joints)

            if hand_joints and hand_joint_values:
                self.resetJointValues(joint_values=hand_joint_values, joints=hand_joints)

            curr_hand_pos, curr_hand_ori = self.getHandPose()
            curr_pos_dist = np.linalg.norm(target_hand_pos - curr_hand_pos)
            curr_dir_vec = np.array(
                Quaternion(curr_hand_ori[3], curr_hand_ori[0], curr_hand_ori[1], curr_hand_ori[2]).rotate([0, 0, 1]))
            curr_ori_dist = np.linalg.norm(target_dir_vec - curr_dir_vec)

            if c > 200:
                break
        if g.hand == 'seed':
            new_base_pos += self.computeMiddlePalmOffset()
            p.resetBasePositionAndOrientation(self.id, new_base_pos, self.original_base_ori)
        self.updateVariables()

    def updateVariables(self):
        self.curr_non_grasp_joint_positions = self.getCurrentJointValues(self.non_grasp_joint_names)
        self.curr_arm_joint_positions = self.getCurrentJointValues(self.arm_joint_names)
        curr_hand_pos, self.curr_hand_ori = self.getHandPose()
        self.target_hand_pos = curr_hand_pos + np.array([0, 0, self.lift_hand_height * 5])
        self.curr_open_joint_positions = self.getCurrentJointValues(self.open_joint_names)

    def setRobotPose(self, robot_pose):
        position, orientation = pose2PosOri(robot_pose)
        p.resetBasePositionAndOrientation(self.id, position, orientation)

    def setGraspableBodyPose(self, body_id, body_pose):
        position, orientation = pose2PosOri(body_pose)
        p.resetBasePositionAndOrientation(body_id, position, orientation)
        p.stepSimulation()

    def toggleAllCollisions(self, enable_collisions):
        pass

    def importObstacle(self, obstacle_name, pose=None):
        self.tray_id = self.importObject(object_name=obstacle_name, pose=pose, use_fixed_base=True)
        p.changeDynamics(self.tray_id, -1, mass=g.hand_mass * 50, lateralFriction=g.table_friction)
        self.obstacles[self.tray_id] = True

    def importObject(self, object_name, pose, use_fixed_base, scale=1):
        if not pose:
            pose = geometry_msgs.msg.Pose()
            pose.orientation.w = 1
        position, orientation = pose2PosOri(pose)
        urdf_path = os.path.join(d.pybullet_obj_dir, os.path.splitext(object_name)[0] + '.urdf')

        try:
            return p.loadURDF(urdf_path, basePosition=position, baseOrientation=orientation,
                              useFixedBase=use_fixed_base, globalScaling=scale)
        except:
            print('error loading', urdf_path)
            raise NotImplementedError

    def importGraspableBody(self, body_name, pose=None, novel=False, specific_scale=None):
        scale = (1 if body_name not in g.obj_scale else g.obj_scale[body_name]) * (
            0.9 if novel else 1) * self.custom_scale
        obj_id = self.importObject(body_name, pose, False, scale=scale)
        p.changeDynamics(obj_id, -1, mass=0.5, linearDamping=g.linearDamping, angularDamping=.4, lateralFriction=0.35,
                         rollingFriction=0.02, spinningFriction=0.02)
        if obj_id in self.graspable_bodies:
            traceback.print_exc()
            raise NotImplementedError
        self.graspable_bodies[obj_id] = body_name
        return obj_id

    def getObjPose(self, obj_id):
        position, orientation = p.getBasePositionAndOrientation(obj_id)
        return self.vector2GeometryMsgs(position, orientation)

    def forceRobotDof(self, dofs):
        for joint in self.free_joints:
            p.resetJointState(self.id, jointIndex=joint, targetValue=0 if joint not in self.spread_joints else dofs[0],
                              targetVelocity=0)

    def simulate(self, seconds):
        iterations = int(240 * seconds)
        for _ in range(iterations):
            p.stepSimulation()
            self.calm_joints()

    def get_all_graspable_body_positions(self):
        return np.array([pose2PosOri(self.getObjPose(obj_id))[0] for obj_id in self.graspable_bodies.keys()])

    def getImageDisk(self, host, port, get_rgb=False):
        w, h = int(d.camera_w), int(d.camera_h)
        images = p.getCameraImage(w, h, viewMatrix=self.viewMat, projectionMatrix=self.projMat,
                                  renderer=p.ER_TINY_RENDERER)
        normalized_depth = images[3][:, :, np.newaxis]
        real_depth = self.far * self.near / (self.far - (self.far - self.near) * normalized_depth)
        assert real_depth.shape == (h, w, 1), (real_depth.shape, [h, w, 1])
        return real_depth, images[2]

    def num_hand_contacts(self):
        all_contacts = []
        for i in [self.ee_link]:
            all_contacts += [contact for contact in p.getContactPoints(bodyA=self.id, linkIndexA=i) if
                             contact[2] != self.id]
        return len(all_contacts)

    def resetJointStates(self, joint_states, joints):
        [p.resetJointState(self.id, joints[i], targetValue=joint_states[i][0], targetVelocity=0) for i in
         range(len(joints))]

    def resetJointValues(self, joint_values, joints):
        [p.resetJointState(self.id, joints[i], targetValue=joint_values[i], targetVelocity=0) for i in
         range(len(joints))]
        p.stepSimulation()
        self.calm_joints()
        p.stepSimulation()

    def dynamicOpen(self, speed=None):
        self.calm_joints()
        current_joint_values = self.getCurrentJointValues(self.open_joint_names)
        count = 0
        if not g.record:
            self.adjustBasePosition([0, 0, 5])
        while count <= 200 and not np.allclose(self.curr_open_joint_positions, current_joint_values, atol=1e-2,
                                               rtol=1e-2):
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.arm_joints, controlMode=p.POSITION_CONTROL,
                                        targetPositions=self.curr_arm_joint_positions,
                                        targetVelocities=self.arm_target_joint_velocities_during_open,
                                        positionGains=self.arm_joints_position_gains_during_open,
                                        velocityGains=self.arm_joints_velocity_gains_during_open,
                                        forces=self.arm_joint_forces_during_open)
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.open_joints,
                                        controlMode=p.POSITION_CONTROL, targetPositions=self.curr_open_joint_positions,
                                        positionGains=self.open_joints_position_gains, forces=self.open_joint_forces)
            p.stepSimulation()
            current_joint_values = [s[0] for s in p.getJointStates(bodyUniqueId=self.id, jointIndices=self.open_joints)]
            count += 1
        if g.record:
            self.adjustBasePosition([0, 0, 5])

    def adjustBasePosition(self, pos_delta):
        base_position, base_orientation = p.getBasePositionAndOrientation(self.id)
        base_position = np.array(base_position) + np.array(pos_delta)
        self.setRobotPose(self.vector2GeometryMsgs(base_position, base_orientation))

    def postTactileCartesian(self, cartesian, name):
        pass

    def adjustHandOrientation(self, ori_rad, safe_lift=True):
        self.calm_joints(force=True)
        if not g.pos_adjustment_only:
            curr_hand_pos, curr_hand_ori = self.getHandPose()
            curr_robot_dirvec = qua_to_dirvec(*xyzw_to_wxyz(*curr_hand_ori))
            rotation_quaternion = Quaternion(axis=curr_robot_dirvec, radians=ori_rad)
            target_hand_ori = wxyz_to_xyzw(*(rotation_quaternion * Quaternion(*xyzw_to_wxyz(*curr_hand_ori))))
            target_robot_dirvec = qua_to_dirvec(*xyzw_to_wxyz(*target_hand_ori))
            assert np.allclose(curr_robot_dirvec, target_robot_dirvec), (curr_robot_dirvec, target_robot_dirvec)

            current_wrist_joint_value = p.getJointState(self.id, self.joint_name2index['joint_6'])[0]
            original_wrist_joint_value = p.getJointState(self.id, self.joint_name2index['joint_6'])[0]
            target_value = float(current_wrist_joint_value + ori_rad)
            target_value -= np.pi / 2
            while True:
                if not np.allclose(original_wrist_joint_value, current_wrist_joint_value, atol=.001, rtol=.001):
                    break
                target_value += np.pi / 2
                while target_value > 2 * math.pi:
                    target_value -= 2 * math.pi
                while target_value < -2 * math.pi:
                    target_value += 2 * math.pi

                target_joint_values = self.getCurrentJointValues(joint_names=self.arm_joint_names)
                target_joint_values[-1] = target_value

                current_joint_values = self.getCurrentJointValues(joint_names=self.arm_joint_names)

                c = 0
                while not np.allclose(target_joint_values, current_joint_values, atol=1e-2, rtol=1e-2):
                    p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.arm_joints,
                                                controlMode=p.POSITION_CONTROL, targetPositions=target_joint_values,
                                                targetVelocities=self.arm_joints_target_velocities_during_lift,
                                                positionGains=self.arm_joints_position_gains_during_adj,
                                                velocityGains=self.arm_joints_velocity_gains_during_lift,
                                                forces=self.arm_joint_forces_during_adj)
                    p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.grasp_joints,
                                                controlMode=p.VELOCITY_CONTROL,
                                                targetVelocities=self.grasp_target_joint_velocities_during_adj,
                                                forces=self.grasp_joint_forces_during_lift)
                    p.stepSimulation()
                    current_joint_values = self.getCurrentJointValues(joint_names=self.arm_joint_names)
                    c += 1
                    if c > 200:
                        break
                original_wrist_joint_value = p.getJointState(self.id, self.joint_name2index['joint_6'])[0]

        self.resetJointValues(joint_values=self.curr_open_joint_positions, joints=self.open_joints)

        self.adjustBasePosition([0, 0, -5])
        self.updateVariables()

    def findInitialContact(self, hand_pose):
        incre = 0.01
        target_hand_position, target_hand_ori = pose2PosOri(hand_pose)
        hand_position, hand_ori = p.getLinkState(self.id, self.ee_link, computeForwardKinematics=1)[:2]
        robot_dirvec = qua_to_dirvec(hand_ori[3], hand_ori[0], hand_ori[1], hand_ori[2])
        base_position, base_orientation = p.getBasePositionAndOrientation()

        joint_states = p.getJointStates(bodyUniqueId=self.id, jointIndices=self.free_joints)
        state = p.saveState()
        c = 0
        while (not self.num_hand_contacts()) and c < 500 and hand_position[2] > target_hand_position[2]:
            c += 1
            self.resetJointStates(joint_states, self.free_joints)
            self.setRobotPose(self.vector2GeometryMsgs(base_position, base_orientation))
            p.stepSimulation()
            base_position += incre * robot_dirvec
            hand_position += incre * robot_dirvec

        if self.num_hand_contacts():
            self.closest_obj_name = self.closestObjectName()
            raise NotImplementedError
        elif hand_position[2] <= target_hand_position[2]:
            base_position -= incre * robot_dirvec
            hand_position -= incre * robot_dirvec

        p.restoreState(state)
        self.setRobotPose(self.vector2GeometryMsgs(base_position, base_orientation))
        self.resetJointStates(joint_states, self.free_joints)

    def calm_joints(self, force=False):
        if g.tactile and not force:
            return
        joint_states = p.getJointStates(self.id, self.all_joints)
        self.resetJointStates(joint_states, self.all_joints)

    def incrementalLift(self):
        self.calm_joints()
        joint_values = p.calculateInverseKinematics(self.id, self.ee_link, self.target_hand_pos, self.curr_hand_ori)

        non_grasp_target_positions_during_lift = [joint_values[i] for i in self.lift_arm_joints_indices_in_free_joints]
        for _ in range(g.lift_ts):
            self.calm_joints()
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.lift_arm_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=non_grasp_target_positions_during_lift,
                                        targetVelocities=self.arm_joints_target_velocities_during_lift,
                                        positionGains=self.arm_joints_position_gains_during_lift,
                                        velocityGains=self.arm_joints_velocity_gains_during_lift,
                                        forces=self.arm_joint_forces_during_lift)
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.grasp_joints,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=self.grasp_target_joint_velocities_during_lift,
                                        forces=self.grasp_joint_forces_during_lift)
            p.stepSimulation()
            if self.getHandPose()[0][2] >= self.lift_hand_height or self.computeIncrementalQuality(delete=False):
                self.curr_non_grasp_joint_positions = self.getCurrentJointValues(self.non_grasp_joint_names)
                return True

        return False

    def lift(self):
        self.calm_joints()
        curr_hand_pos, curr_hand_ori = self.getHandPose()
        target_hand_pos = curr_hand_pos + np.array([0, 0, self.lift_hand_height * 10])
        joint_values = p.calculateInverseKinematics(self.id, self.ee_link, target_hand_pos, curr_hand_ori)

        c = 0
        non_grasp_target_positions_during_lift = [joint_values[i] for i in self.lift_arm_joints_indices_in_free_joints]
        while self.getHandPose()[0][2] < self.lift_hand_height and (
                not self.computeIncrementalQuality(delete=False)) and c < d.l:
            c += 1
            self.calm_joints()
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.lift_arm_joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=non_grasp_target_positions_during_lift,
                                        targetVelocities=self.arm_joints_target_velocities_during_lift,
                                        positionGains=self.arm_joints_position_gains_during_lift,
                                        velocityGains=self.arm_joints_velocity_gains_during_lift,
                                        forces=self.arm_joint_forces_during_lift)
            p.setJointMotorControlArray(bodyIndex=self.id, jointIndices=self.grasp_joints,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=self.grasp_target_joint_velocities_during_lift,
                                        forces=self.grasp_joint_forces_during_lift)
            p.stepSimulation()

    def computeIncrementalQuality(self, delete):
        heights = {obj_id: (self.getObjPose(obj_id).position.z > self.lift_obj_height) for obj_id in
                   self.graspable_bodies}
        successes = sum(heights.values())
        if self.test and delete and g.algo == 'tactile' and not g.record:
            raw_input('success %s' % (successes > 0))
        for graspable_body_id in heights:
            if heights[graspable_body_id] and delete:
                p.removeBody(graspable_body_id)
                del self.graspable_bodies[graspable_body_id]
        if delete:
            print('success', int(successes > 0))
        return int(successes > 0)

    def randomObjectName(self):
        return os.path.basename(random.choice(self.graspable_bodies.values()))

    def computeQuality(self, delete=True):
        assert self.graspable_bodies, self.graspable_bodies
        self.lift()
        return self.computeIncrementalQuality(delete)

    def computeHandJointValues(self, finger_1, finger_2, finger_3, finger_4, spread):
        raise NotImplementedError

    def computeMiddlePalmOffset(self):
        pass
