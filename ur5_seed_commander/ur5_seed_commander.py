from staubli_barrett_commander.staubli_barrett_commander import *

try:
    from queue import Queue
except:
    from Queue import Queue
import roslib

roslib.load_manifest('my_seed_hand')

from src.my_seed_hand.motion_controller.action_client.simple_client import Client, map_to_sim_frame


class Ur5SeedCommander(StabuliBarrettCommander):
    WORLD_LINK = "base_link"
    ARM = "ur5"
    GRIPPER = "seed"
    ARM_TIP_LINK = HAND_BASE_LINK = 'Mproximal_virtual'
    ARM_GRIPPER = "ur5_seed"
    PLANNING_TIME_LIMIT = .8
    ARM_JOINT_NAMES = ["shoulder_pan_joint", 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                       'wrist_3_joint', 'palm__palm_axis', 'forearm__base', 'palm_axis__forearm']
    PALM_JOINTS = ['forearm__base', 'palm_axis__forearm', 'palm__palm_axis']
    OPEN_POSITION = [0] * 6
    OPEN_GRIP_POSITION = [0.01] * 3
    CLOSED_POSITION = np.array([2.44, 0.84, 2.44, 0.84, 2.44, 0.84])
    FINGER_JOINTS = ['Ttip__Tmiddle', 'Itip__Imiddle', 'Mtip__Mmiddle', 'Rtip__Rmiddle', 'Ptip__Pmiddle',
                     'Tmiddle__Tproximal', 'Imiddle__Iproximal', 'Mmiddle__Mproximal', 'Rmiddle__Rproximal',
                     'Pmiddle__Pproximal']
    SPREAD_JOINTS = ['palm__thumb_base']
    SPREAD_JOINT = 'palm__thumb_base'
    GRIP_JOINTS = ['Tproximal__thumb_base', 'Iproximal__palm', 'Mproximal__palm', 'Rproximal__palm', 'Pproximal__palm']
    TIP_JOINTS = ['Ttip__Tmiddle', 'Itip__Imiddle', 'Mtip__Mmiddle', 'Rtip__Rmiddle', 'Ptip__Pmiddle',
                  'Tmiddle__Tproximal', 'Imiddle__Iproximal', 'Mmiddle__Mproximal', 'Rmiddle__Rproximal',
                  'Pmiddle__Pproximal']
    HAND_JOINTS = GRIP_JOINTS
    MAX_FINGER_1_JOINT = np.pi
    MAX_FINGER_2_JOINT = np.pi
    MAX_FINGER_3_JOINT = np.pi
    MAX_FINGER_4_JOINT = np.pi
    MAX_FINGER_SPREAD = 5 * np.pi
    MAX_FINGER_SPREAD_OFFSET = MAX_FINGER_SPREAD * 0.8
    PROXIMAL_TIP_RATIO = 0.3 * 0.9
    PROXIMAL_MIDDLE_RATIO = 0.9
    LIFT_HEIGHT = 0.2 if g.eval_single_object else 0.25

    def read_ros_params(self):
        pass

    def getHandBasePose(self):
        return self.getLinkPose(target_frame=self.WORLD_LINK, source_frame=self.HAND_BASE_LINK)

    def process_and_save_pcl_as_mesh(self, cloud_points, mesh_filepath):
        if cloud_points is not None:
            cloud_points = cloud_points[~np.isnan(cloud_points).any(axis=1)]
            ply_data = curvox.cloud_to_mesh_conversions.marching_cubes_completion(cloud_points,
                                                                                  percent_offset=(.5, .5, .5),
                                                                                  smooth=False, patch_size=200)
            ply_data.write(open(mesh_filepath, 'wb'))

        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = 'camera_rgb_optical_frame'
        pose_stamped.pose = Pose(Point(0, 0, -0.002), Quaternion(0, 0, 0, 1))
        wm_client.add_mesh('scene_mesh', mesh_filepath, pose_stamped)

    def setHandPose(self, hand_pose, spread=None, finger_1=None, finger_2=None, finger_3=None, finger_4=None,
                    skip_execution=True, post_point_cloud=True, virtual=False, stop=False, execute=True,
                    get_safe_finger_positions=False, cartesian=False, wait=True, robot_state=None):

        if self.checkNoExecute(hand_pose, skip_execution):
            return

        hand_base_pose = self.getSafeZHandPose(hand_pose)

        if robot_state is None:
            robot_state = self.robot_commander.get_current_state()

        values = dict(zip(robot_state.joint_state.name, robot_state.joint_state.position))
        if spread is not None:
            values[self.SPREAD_JOINT] = map_to_sim_frame(self.SPREAD_JOINT, np.clip(spread, 0, 2 * np.pi))

        if self.needToMoveFingers(finger_1, finger_2, finger_3, finger_4):
            values['Tproximal__thumb_base'] = map_to_sim_frame('Tproximal__thumb_base', finger_1 + 0.7)
            values['Iproximal__palm'] = map_to_sim_frame('Iproximal__palm', finger_2 + 0.4)
            values['Mproximal__palm'] = map_to_sim_frame('Mproximal__palm', finger_3 + 0.4)
            values['Rproximal__palm'] = map_to_sim_frame('Rproximal__palm', finger_4 + 0.4)
            values['Pproximal__palm'] = values['Rproximal__palm']
            values['Tmiddle__Tproximal'] = values['Tproximal__thumb_base']
            values['Imiddle__Iproximal'] = values['Iproximal__palm'] * self.PROXIMAL_MIDDLE_RATIO
            values['Mmiddle__Mproximal'] = values['Mproximal__palm'] * self.PROXIMAL_MIDDLE_RATIO
            values['Rmiddle__Rproximal'] = values['Rproximal__palm'] * self.PROXIMAL_MIDDLE_RATIO
            values['Pmiddle__Pproximal'] = values['Pproximal__palm'] * self.PROXIMAL_MIDDLE_RATIO
            values['Ttip__Tmiddle'] = values['Tproximal__thumb_base']
            values['Itip__Imiddle'] = values['Iproximal__palm'] * self.PROXIMAL_TIP_RATIO
            values['Mtip__Mmiddle'] = values['Mproximal__palm'] * self.PROXIMAL_TIP_RATIO
            values['Rtip__Rmiddle'] = values['Rproximal__palm'] * self.PROXIMAL_TIP_RATIO
            values['Ptip__Pmiddle'] = values['Pproximal__palm'] * self.PROXIMAL_TIP_RATIO

        robot_state.joint_state.position = tuple([values[i] for i in robot_state.joint_state.name])

        self.arm_commander.set_start_state(robot_state)
        if cartesian:
            result, _ = self.arm_commander.compute_cartesian_path(waypoints=[hand_base_pose], eef_step=0.05,
                                                                  jump_threshold=0)
        else:
            result = self.planHandBasePose(hand_base_pose, stop)

        if not execute:
            return hand_pose, 0, spread, finger_1, finger_2, finger_3, finger_4

        if self.needToMoveFingers(finger_1, finger_2, finger_3, finger_4) and not virtual:
            joint_names = result.joint_trajectory.joint_names
            end_position = result.joint_trajectory.points[-1].positions
            self.target_palm_jv = dict(zip(joint_names, end_position))
            real_values = self.getTargetPalmJointValues()
            print(self.target_palm_jv, real_values)
            success = self.client.move_joint_till_match(real_values + [spread, finger_1, finger_2, finger_3, finger_4])
            if not success:
                raise NotImplementedError

        self.execute(result, wait)
        self.cache_rqt()

    def move_object_to_bin_cartesian(self):
        self.slow_grasping = False
        try:
            self.setHandPose(get_pose(*g.real_home), post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                             finger_4=0, spread=0, virtual=True, cartesian=True, wait=g.record_seed_video)

            if g.record_seed_video:
                current_tip_position = self.getArmTipPose().position
                position = [current_tip_position.x, current_tip_position.y]
                while not np.allclose(position, g.real_home[:2], atol=.5e-1, rtol=.5e-1):
                    try:
                        self.setHandPose(get_pose(*g.real_home), post_point_cloud=False, finger_1=0, finger_2=0, finger_3=0,
                                         finger_4=0, spread=0, virtual=True, cartesian=False, wait=True)
                    except:
                        current_tip_position = self.getArmTipPose().position
                        position = [current_tip_position.x, current_tip_position.y]
                self.forceRobotDof()
            else:
                while True:
                    current_tip_position = self.getArmTipPose().position
                    position = [current_tip_position.x, current_tip_position.y]
                    if np.allclose(position, g.real_home[:2], atol=5e-1, rtol=5e-1):
                        self.forceRobotDof()
                        break

        except:
            if not g.record_seed_video:
                self.forceRobotDof()
        time.sleep(2)

    def needToMoveFingers(self, finger_1, finger_2, finger_3, finger_4):
        return finger_1 is not None and finger_2 is not None and finger_3 is not None and finger_4 is not None

    def getTargetPalmJointValues(self):
        raw = [self.target_palm_jv[i] for i in self.PALM_JOINTS]
        return self.client.map_to_real_frames(raw)

    def need_lift_a_bit(self, i):
        return abs(self.client.current_joint_state[4] - max(i - 0.2, 0)) > 0.3 and self.client.current_joint_state[
            4] < 0.6 and self.getArmTipPose().position.z < -0.205

    def autoGrasp(self, slow=False):
        self.scene.removeCollisionObject('scene_mesh')
        self.client.logging = True
        if slow:
            self.client.move_joint(self.getTargetPalmJointValues() + [8, 7, 7, 7, 10], slow=self.slow_grasping)

            self.client.logging = False
            while self.need_lift_a_bit(i=5):
                self.lift_a_bit(delta=0.004)
                self.client.move_joint(self.getTargetPalmJointValues() + [8, 7, 7, 7, 10], slow=self.slow_grasping)
            self.client.logging = True
        else:
            i = -1
            offset = .5
            while i <= 3.0:
                print("Step {}".format(i))
                self.client.move_joint(self.getTargetPalmJointValues() + [8, i - 0.2, i, i, 10 + i])
                self.client.logging = False
                while self.need_lift_a_bit(i):
                    print(self.client.current_joint_state[4], max(i - 0.2, 0))
                    self.lift_a_bit(delta=0.004)
                    self.client.move_joint(
                        self.getTargetPalmJointValues() + [8, i - 0.2, self.client.current_joint_state[5],
                                                           self.client.current_joint_state[6],
                                                           10 + self.client.current_joint_state[7]])
                self.client.logging = True
                i += offset
        self.client.logging = False

    def lift_a_bit(self, delta=0.001):
        self.arm_commander.set_start_state(self.robot_commander.get_current_state())
        pose = self.getArmTipPose()
        pose.position.z += delta
        print('adjusting', pose.position.z, self.baseline_efforts)
        result, _ = self.arm_commander.compute_cartesian_path(waypoints=[pose], eef_step=0.1, jump_threshold=0)
        self.arm_commander.execute(result, True)

    def execute(self, result, wait):
        if wait:
            raw_input('wait')
        self.arm_commander.execute(result, wait)

    def getArmTipPoseFromHandBasePose(self, pose):
        return pose

    def forceRobotDof(self, dofs=None, speed=-.8, avoid_collision=False):
        self.keep_grasping = False
        time.sleep(.2)
        self.client.move_joint([np.pi, np.pi, np.pi, 10, -0.5, -0.5, -0.5, -0.5])

    def cache_rqt(self):
        pass

    def init_tactile(self):
        pass

    def init_tactile_cartesian(self):
        pass

    def init_finger_joint_states(self):
        try:
            self._joint_subscriber = rospy.Subscriber('/joint_states', JointState, self._receive_joints_data)
        except ValueError, e:
            rospy.logerr(e)
        self.client = Client('seed_hand_controller', simulation_input=False, ros_node=False, logging=False)

    def _receive_joints_data(self, msg):
        if self.keep_grasping:
            self.autoGrasp(slow=True)
