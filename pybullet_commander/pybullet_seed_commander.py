import pybullet as p

import numpy as np

from config import g
from pybullet_barrett_commander import PybulletBarrettCommander


class PybulletSeedCommander(PybulletBarrettCommander):
    def custom_init(self):
        self.target_grasp_grip_velocity = .25
        self.grip_joint_names = ['Tproximal__thumb_base', 'Iproximal__palm', 'Mproximal__palm', 'Rproximal__palm',
                                 'Pproximal__palm']
        self.finger_joint_names = ['Ttip__Tmiddle', 'Itip__Imiddle', 'Mtip__Mmiddle', 'Rtip__Rmiddle', 'Ptip__Pmiddle',
                                   'Tmiddle__Tproximal', 'Imiddle__Iproximal', 'Mmiddle__Mproximal',
                                   'Rmiddle__Rproximal', 'Pmiddle__Pproximal']
        self.palm_joint_names = ['palm__palm_axis', 'forearm__base', 'palm_axis__forearm']
        self.lift_arm_joint_names = ['shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                                     'wrist_3_joint']
        self.arm_joint_names = ["shoulder_pan_joint", 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                                'wrist_2_joint', 'wrist_3_joint', 'palm__palm_axis', 'forearm__base',
                                'palm_axis__forearm']
        self.spread_joint_names = ['palm__thumb_base']
        self.tactile_link_names = []
        self.end_effector_link_name = 'palm__palm_axis'
        self.grasp_end_effector_link_name = 'Mproximal__palm'
        self.MAX_FINGER_1_JOINT = 1.57079632679
        self.MAX_FINGER_2_JOINT = 1.4835298642
        self.MAX_FINGER_3_JOINT = 1.4835298642
        self.MAX_FINGER_4_JOINT = 1.4835298642
        self.MAX_FINGER_SPREAD = 1.57079632679
        self.MAX_FINGER_SPREAD_OFFSET = self.MAX_FINGER_SPREAD * 0.8
        self.PROXIMAL_TIP_RATIO = 0.3 * 0.9
        self.PROXIMAL_MIDDLE_RATIO = 0.9
        self.target_grasp_velocities = self.target_grasp_grip_velocity * np.array(
            [1] * 5 + [self.PROXIMAL_TIP_RATIO] * 5 + [self.PROXIMAL_MIDDLE_RATIO] * 5)
        self.arm_joints_position_gains_during_lift_factor = 0.001
        self.auto_grasp_num_iters = 250
        self.custom_scale = g.custom_obj_scale
        self.lift_obj_height = 0.2
        self.lift_hand_height = self.lift_obj_height + 0.3
        self.dampingRatio = 1

    def computeMiddlePalmOffset(self):
        self.grasp_end_effector_link = self.joint_name2index[self.grasp_end_effector_link_name]
        middle_finger_pos = p.getLinkState(self.id, self.grasp_end_effector_link, computeForwardKinematics=1)[0]
        palm_pos = p.getLinkState(self.id, self.ee_link, computeForwardKinematics=1)[0]
        return np.array(palm_pos) - np.array(middle_finger_pos)

    def computeHandJointValues(self, finger_1, finger_2, finger_3, finger_4, spread):
        hand_joints = []
        hand_joint_values = []
        if spread is not None:
            hand_joints += self.spread_joints
            hand_joint_values += [spread] * self.num_spread_joints
        if finger_1 is not None and finger_2 is not None and finger_3 is not None and finger_4 is not None:
            hand_joints += self.grip_joints
            hand_joints += self.finger_joints
            hand_joint_values += [finger_1, finger_2, finger_3, finger_4, finger_4]
            hand_joint_values += [finger_1 * self.PROXIMAL_TIP_RATIO, finger_2 * self.PROXIMAL_TIP_RATIO,
                                  finger_3 * self.PROXIMAL_TIP_RATIO, finger_4 * self.PROXIMAL_TIP_RATIO,
                                  finger_4 * self.PROXIMAL_TIP_RATIO, finger_1 * self.PROXIMAL_MIDDLE_RATIO,
                                  finger_2 * self.PROXIMAL_MIDDLE_RATIO, finger_3 * self.PROXIMAL_MIDDLE_RATIO,
                                  finger_4 * self.PROXIMAL_MIDDLE_RATIO, finger_4 * self.PROXIMAL_MIDDLE_RATIO]
        return hand_joints, hand_joint_values

    def move_single_joint(self):
        raise NotImplementedError
