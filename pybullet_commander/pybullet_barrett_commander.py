import numpy as np

from config import g
from pybullet_commander import PybulletCommander


# ('bh_j11_joint', 0.0, 3.1416)
# ('bh_j21_joint', 0.0, 3.1416)
# ('bh_j12_joint', 0.0, 2.44)
# ('bh_j22_joint', 0.0, 2.44)
# ('bh_j32_joint', 0.0, 2.44)
# ('bh_j13_joint', 0.0, 0.84)
# ('bh_j23_joint', 0.0, 0.84)
# ('bh_j33_joint', 0.0, 0.84)

class PybulletBarrettCommander(PybulletCommander):
    MAX_FINGER_SPREAD = np.pi / 2
    MAX_FINGER_SPREAD_OFFSET = 0
    num_tactile_links = 96
    tactile_link_names = ['palm', 'link1', 'link2', 'link3']
    num_cells_per_link = num_tactile_links / len(tactile_link_names)
    MAX_FINGER_1_JOINT = MAX_FINGER_2_JOINT = MAX_FINGER_3_JOINT = 2.44
    MAX_FINGER_1_TIP_JOINT = MAX_FINGER_2_TIP_JOINT = MAX_FINGER_3_TIP_JOINT = 0.84
    PROXIMAL_TIP_RATIO = MAX_FINGER_1_TIP_JOINT / MAX_FINGER_1_JOINT

    def custom_init(self):
        self.target_grasp_grip_velocity = 2.5
        self.grip_joint_names = ['bh_j12_joint', 'bh_j22_joint', 'bh_j32_joint']
        self.finger_joint_names = ['bh_j13_joint', 'bh_j23_joint', 'bh_j33_joint']
        self.spread_joint_names = ['bh_j11_joint', 'bh_j21_joint']
        self.palm_joint_names = []
        self.lift_arm_joint_names = self.arm_joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5',
                                                            'joint_6']
        self.end_effector_link_name = 'bh_base_joint'
        self.PROXIMAL_MIDDLE_RATIO = None
        self.arm_joints_position_gains_during_lift_factor = 0.002
        self.target_grasp_velocities = np.array(
            [self.target_grasp_grip_velocity] * 3 + [self.target_grasp_grip_velocity * self.PROXIMAL_TIP_RATIO] * 3)
        self.auto_grasp_num_iters = 200
        self.custom_scale = g.custom_obj_scale
        self.lift_obj_height = 0.4
        self.lift_hand_height = self.lift_obj_height + 0.4
        self.dampingRatio = 1

    def computeHandJointValues(self, finger_1, finger_2, finger_3, finger_4, spread):
        hand_joints = []
        hand_joint_values = []
        if spread is not None:
            hand_joints += self.spread_joints
            hand_joint_values += [spread] * self.num_spread_joints
        if finger_1 is not None and finger_2 is not None and finger_3 is not None:
            hand_joints += self.grip_joints
            hand_joints += self.finger_joints
            hand_joint_values += [finger_1, finger_2, finger_3]
            hand_joint_values += [finger_1 * self.PROXIMAL_TIP_RATIO, finger_2 * self.PROXIMAL_TIP_RATIO,
                                  finger_3 * self.PROXIMAL_TIP_RATIO]
        return hand_joints, hand_joint_values
