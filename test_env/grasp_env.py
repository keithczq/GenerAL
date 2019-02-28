import copy
import json
import os
import uuid

import geometry_msgs
from pyquaternion import Quaternion

from pybullet_commander.pybullet_seed_commander import PybulletSeedCommander
from scripts.transformations import *

try:
    from graspit_commander import GraspitCommander
except:
    pass
from pybullet_commander.pybullet_barrett_commander import PybulletBarrettCommander
from utils import *
from utils import _random


class grasp_env(object):
    def __init__(self, host, port, eval, real, id):
        self.id = id
        self.save_task_id = 0
        self.eval = eval
        self.real = real
        self.h = np.arange(int(d.camera_h))[:, np.newaxis]
        self.w = np.arange(int(d.camera_w))[np.newaxis, :]
        self.obj_index = -1
        self.in_grasp = False
        self.all_pybullet_envs = {i: None for i in os.listdir(d.pybullet_env_dir)}
        self.all_pybullet_envs_prefix = {os.path.splitext(f)[0]: 0 for f in self.all_pybullet_envs}
        for f in self.all_pybullet_envs:
            self.all_pybullet_envs_prefix[os.path.splitext(f)[0]] += 1

        self.multi_seen_envs = list(set([f for f in self.all_pybullet_envs_prefix if
                                         'seen' in f and not f.startswith('1-') and self.files_exist(f)]))
        self.multi_novel_envs = list(set([f for f in self.all_pybullet_envs_prefix if
                                          'novel' in f and not f.startswith('1-') and self.files_exist(f)]))
        self.single_seen_envs = list(set(
            [f for f in self.all_pybullet_envs_prefix if 'seen' in f and f.startswith('1-') and self.files_exist(f)]))
        self.single_novel_envs = list(set(
            [f for f in self.all_pybullet_envs_prefix if 'novel' in f and f.startswith('1-') and self.files_exist(f)]))

        self.envs_dict = {'multi': {'seen': self.multi_seen_envs, 'novel': self.multi_novel_envs},
                          'single': {'seen': self.single_seen_envs, 'novel': self.single_novel_envs}}

        if self.real:
            if g.hand == 'barrett':
                from staubli_barrett_commander.staubli_barrett_commander import StabuliBarrettCommander
                self.commander = StabuliBarrettCommander(self.id)
            else:
                from ur5_seed_commander.ur5_seed_commander import Ur5SeedCommander
                self.commander = Ur5SeedCommander(self.id)
        else:
            if self.is_graspit():
                self.commander = GraspitCommander(host, port)
            else:
                if g.hand == 'barrett':
                    self.commander = PybulletBarrettCommander(eval)
                else:
                    self.commander = PybulletSeedCommander(eval)
        self.host = host
        self.port = port

    def files_exist(self, f):
        return f in self.all_pybullet_envs_prefix and self.all_pybullet_envs_prefix[f] == 3

    def get_state(self, t, get_rgb=False):
        depth, rgb = self.commander.getImageDisk(self.host, self.port, get_rgb)
        normalized_image = normalize_depth(depth, fixed=True)
        resized_image = cv2.resize(normalized_image, d.img_sp)[:, :, np.newaxis]
        ret = {'max_depth': np.max(depth), 'image': resized_image, 'real_depth': depth,
               'transform': {'cx': (depth.shape[1] - 1) / 2.0, 'cy': (depth.shape[0] - 1) / 2.0,
                             'dx': (depth.shape[1] - 1) / 2.0, 'dy': (depth.shape[0] - 1) / 2.0}, 't': t,
               'rgb': (rgb if get_rgb else None), 'scale': 1.0}
        if get_rgb:
            assert 'rgb' in ret
        return ret

    def reset_robot_pose(self):
        if self.real:
            hand_pose = get_pose(*g.real_home)
            if g.hand == 'seed':
                while True:
                    try:
                        self.commander.setHandPose(hand_pose, post_point_cloud=False)
                        break
                    except:
                        traceback.print_exc()
                        pass
            else:
                self.commander.setHandPose(hand_pose, post_point_cloud=False)
        else:
            robot_pose = get_pose(0, -1.8 if g.hand == 'barrett' else 1.8, 0.4, 0, 0, 0, 1)
            self.commander.setRobotPose(robot_pose)
        self.commander.rememberRobotPose()

    def init_world(self, test=False, novel=False, multi=True):
        if self.print_log():
            print('init_world...', test, novel, multi, self.id)
        env = g.novel_objs if (test and novel) else g.seen_objs
        num_envs = len(env)
        self.commander.clearWorld()
        self.commander.toggleAllCollisions(True)
        if self.is_graspit():
            self.commander.loadWorld('plannerTactile')
        else:
            self.commander.loadWorld()
            self.tray_id = self.commander.importObstacle('original/tray/tray')
            self.commander.importRobot(g.robot_urdf, novel=novel)

            if g.load_second_robot:
                pose = geometry_msgs.msg.Pose()
                pose.orientation.w = 1
                pose.position.x = -1
                pose.position.y = -.25
                pose.position.z = 0.5
                self.commander.importObject(g.second_robot_urdf, pose, True)

            if g.use_saved_envs:
                multi_keyword = 'multi' if multi else 'single'
                novel_keyword = 'novel' if novel else 'seen'
                available_envs = self.envs_dict[multi_keyword][novel_keyword] if test else (
                        self.envs_dict['multi']['seen'] + self.envs_dict['single']['seen'])
                sample_env = random.choice(available_envs)
                with open(os.path.join(d.pybullet_env_dir, sample_env + '.json')) as f:
                    objs = json.load(f)
                self.objects = []
                for key in sorted(objs.keys()):
                    index = int(key)
                    pose = geometry_msgs.msg.Pose()
                    pose.position.x = (index + 50) * 2
                    pose.position.y = (index + 50) * 2
                    pose.position.z = (index + 50) * 2
                    pose.orientation.w = 1
                    self.objects.append(self.commander.importGraspableBody(objs[key], pose))
                self.commander.restore_state(sample_env)
                return

        self.reset_robot_pose()

        if self.real:
            self.open()
            return

        if test:
            self.num_bodies = random.randint(g.min_num_bodies, g.max_num_bodies) if multi else 1
        else:
            real_multi = random.randint(0, 1) if multi else 0
            self.num_bodies = random.randint(g.min_num_bodies, g.max_num_bodies) if real_multi else 1
        self.objects = []
        for obj in range(self.num_bodies):
            obj_index = random.randint(0, num_envs - 1)
            pose = geometry_msgs.msg.Pose()
            pose.position.x = (obj + 50) * 2
            pose.position.y = (obj + 50) * 2
            pose.position.z = (obj + 50) * 2
            pose.orientation.w = 1
            if self.is_graspit():
                self.commander.importGraspableBody(env[obj_index], pose)
                self.objects.append(obj)
            else:
                self.objects.append(self.commander.importGraspableBody(env[obj_index], pose))
        self.open()
        if not self.is_graspit():
            self.reset_bodies()
            self.commander.simulate(2)
            if g.statistics and g.save_pybullet_state:
                self.commander.save_state_disk(
                    filename='%s-%s-%s-%s' % (self.num_bodies, 'novel' if novel else 'seen', self.id, uuid.uuid4()))
                self.save_task_id += 1

    def reset_body(self, id=0):
        collision = True
        x = -0.4
        while collision:
            try:
                if self.is_graspit():
                    a = [x, _random(d.pos_delta) * d.rand_p * 5, _random(d.pos_delta) * d.rand_p * 2 + 0.06,
                         _random(d.ori_delta), _random(d.ori_delta), _random(d.ori_delta), _random(d.ori_delta)]
                else:
                    a = [_random(d.pos_delta) * 2, _random(d.pos_delta) * 2 if (d.camera_pitch_deg != -60 or g.random_reset_body) else -0.3, 0.5, _random(d.ori_delta),
                         _random(d.ori_delta), _random(d.ori_delta), _random(d.ori_delta)]
                self.body_pose_vectors.append(normalized_action(a))
                body_pose = get_pose(*self.body_pose_vectors[-1])
                self.commander.setGraspableBodyPose(id, body_pose)
                collision = False
            except:
                traceback.print_exc()
                collision = True
                x += 0.01

    def reset_bodies(self):
        self.body_pose_vectors = []
        for obj in self.objects:
            self.reset_body(obj)

    def reset(self, wait_time=None):
        self.open()
        self.reset_robot_pose()
        if wait_time:
            self.commander.simulate(wait_time)
        if self.is_graspit():
            self.reset_bodies()
        return self.get_state(1, get_rgb=self.real)

    def open(self):
        if self.id == 0:
            self.commander.forceRobotDof([0, 0, 0, 0])

    def get_tactile_info(self):
        return self.commander.getRobot(0).robot.tactile.sensor_forces

    def world_pos_raw(self, a, s):
        transform = s['transform']
        cx, cy = get_local_cx_cy(a['pixel'], w=transform['dx'] * 2, h=transform['dy'] * 2)
        cx, cy = transform_coord(cx=cx, cy=cy, transform=transform)
        cx = max(0, min(cx, d.camera_w - 1))
        cy = max(0, min(cy, d.camera_h - 1))
        low_y_coord = int(math.floor(cy))
        low_x_coord = int(math.floor(cx))
        high_y_coord = int(math.ceil(cy))
        high_x_coord = int(math.ceil(cx))
        low_y_low_x = s['real_depth'][low_y_coord, low_x_coord, 0]
        high_y_low_x = s['real_depth'][high_y_coord, low_x_coord, 0]
        low_y_high_x = s['real_depth'][low_y_coord, high_x_coord, 0]
        high_y_high_x = s['real_depth'][high_y_coord, high_x_coord, 0]
        try:
            depth = bilinear_interpolation(cx, cy, points=[[low_x_coord, low_y_coord, low_y_low_x],
                                                           [high_x_coord, low_y_coord, low_y_high_x],
                                                           [low_x_coord, high_y_coord, high_y_low_x],
                                                           [high_x_coord, high_y_coord, high_y_high_x]])
        except:
            depth = np.mean([low_y_low_x, low_y_high_x, high_y_low_x, high_y_high_x])

        if d.camera_pitch_deg == 0:
            return self.pixel_to_3D_coord(cx, cy, depth)
        else:
            mouseX = cx * self.commander.width / d.camera_w
            mouseY = cy * self.commander.height / d.camera_h
            return self.commander.getRayFromTo(mouseX, mouseY, depth)

    def world_pos(self, a, s):
        pos = self.world_pos_raw(a, s)
        if g.hand == 'seed' and pos[2] < -0.288:
            print(pos, self.id)
            raise NotImplementedError
        if g.hand == 'barrett' and not self.real and pos[2] < 0.1:
            print(pos, self.id)
            raise NotImplementedError

        yaw = np.pi * min(1, max(-1, a['yaw']))
        pitch = np.pi * min(0.99, max(0.01, a['pitch']))
        roll = np.pi * min(1, max(-1, a['roll']))
        original_ori = quaternion_from_euler(roll, pitch, yaw)
        ori = [original_ori[1], original_ori[2], original_ori[3], original_ori[0]]
        dir_vec = np.array(Quaternion(ori[3], ori[0], ori[1], ori[2]).rotate([1, 0, 0]))
        assert dir_vec[2] <= 0 and np.allclose(np.linalg.norm(dir_vec), 1, atol=1e-3), (dir_vec, pitch, np.sin(pitch))
        pos -= g.dist_away * dir_vec
        safe_z = g.safe_real_z if self.real else g.safe_simulator_z
        while pos[2] < safe_z:
            scale = (safe_z - pos[2]) / dir_vec[2]
            pos += scale * dir_vec
        assert pos[2] >= safe_z, (pos, safe_z, dir_vec)
        qua = Quaternion(ori[3], ori[0], ori[1], ori[2]) * Quaternion(g.x_axis_dirvec[3], g.x_axis_dirvec[0],
                                                                      g.x_axis_dirvec[1], g.x_axis_dirvec[2])
        assert np.array(qua.rotate([0, 0, 1]))[2] <= 0
        action = np.concatenate([pos, [qua[1], qua[2], qua[3], qua[0]]])

        assert np.allclose(np.linalg.norm(dir_vec), 1)
        return check_action(action), dir_vec

    def get_cropped_image(self, depth, a, transform, max_depth):
        cx, cy = get_local_cx_cy(a['pixel'], w=transform['dx'] * 2, h=transform['dy'] * 2)
        dx, dy = get_global_local_dx_dy(local_cx=cx, local_cy=cy, local_h=transform['dy'] * 2,
                                        local_w=transform['dx'] * 2, scale=a['scale'])
        cx, cy = transform_coord(cx, cy, transform)
        padded_depth = np.pad(depth, ((max(-int(round(cy - dy)), 0), max(int(round(cy + dy) - (d.camera_h - 1)), 0)),
                                      (max(-int(round(cx - dx)), 0), max(int(round(cx + dx) - (d.camera_w - 1)), 0)),
                                      (0, 0)), 'constant', constant_values=max_depth)

        padded_x = max(-int(round(cx - dx)), 0)
        padded_y = max(-int(round(cy - dy)), 0)
        cropped_depth = padded_depth[int(round(cy - dy)) + padded_y:int(round(cy + dy)) + padded_y,
                        int(round(cx - dx)) + padded_x:int(round(cx + dx)) + padded_x, :]
        resized_depth = cv2.resize(normalize_depth(cropped_depth, fixed=True), d.img_sp)[:, :, np.newaxis]
        return resized_depth, {'cx': cx, 'cy': cy, 'dx': dx, 'dy': dy}

    def pixel_to_3D_coord(self, cx, cy, depth):
        x = (cx - d.cx) / d.fx * depth
        y = -(cy - d.cy) / d.fy * depth
        z = -depth
        if not self.is_graspit():
            z = -z
            y = -y

        return np.matmul(d.camera_to_world_tran, [x, y, z, 1])[:3]

    def print_log(self):
        return self.eval and not self.real and not g.statistics

    def grasp(self, s, a, grasp=True, get_pre_grasp_state=False, execute=True, robot_state=None):
        self.open()
        spread = self.commander.MAX_FINGER_SPREAD_OFFSET + a['spread'] * (
                self.commander.MAX_FINGER_SPREAD - self.commander.MAX_FINGER_SPREAD_OFFSET) if g.learn_spread else (
                    self.commander.MAX_FINGER_SPREAD / 2)
        finger_1 = (a['finger_1'] * self.commander.MAX_FINGER_1_JOINT / g.fd_div) if g.flexible_dof else 0.0
        finger_2 = (a['finger_2'] * self.commander.MAX_FINGER_2_JOINT / g.fd_div) if g.flexible_dof else 0.0
        finger_3 = (a['finger_3'] * self.commander.MAX_FINGER_3_JOINT / g.fd_div) if g.flexible_dof else 0.0
        if g.hand == 'barrett':
            finger_4 = None
        else:
            finger_4 = (a['finger_4'] * self.commander.MAX_FINGER_4_JOINT / g.fd_div) if g.flexible_dof else 0.0
        if self.print_log():
            print(spread, finger_1, finger_2, finger_3, finger_4)
        grasp_act, dir_vec = self.world_pos(a, s)
        grasp_act = check_action(grasp_act)
        pos, ori = pose2PosOri(get_pose(*grasp_act))
        approach_scale = (g.dist_away - (g.real_dist_away + (a['add_z'] if 'add_z' in a else 0)))
        pos += approach_scale * dir_vec
        safe_z = g.safe_real_z if self.real else g.safe_simulator_z
        while pos[2] < safe_z:
            scale = (safe_z - pos[2]) / dir_vec[2]
            pos += scale * dir_vec
        assert pos[2] >= safe_z, (pos, safe_z, dir_vec)

        if self.real and self.id == 0:
            print(a['index'], self.id, dir_vec)

        if g.check_collision:
            state = self.commander.saveState()

        if not g.real:
            random_angle = random.random() * 2 * np.pi
            pos += (2.5 * g.calibration_noise * np.array([np.sin(random_angle), np.cos(random_angle), 0]))
        else:
            if g.hand == 'barrett':
                pos += np.array([0, g.calibration_noise, -0.05])
            else:
                pos += np.array([0, 0, 0])

        ret = self.commander.setHandPose(vector2GeometryMsgs(pos, ori), spread=spread, finger_1=finger_1,
                                         finger_2=finger_2, finger_3=finger_3, finger_4=finger_4,
                                         post_point_cloud=False, execute=execute, get_safe_finger_positions=True,
                                         robot_state=robot_state)
        if g.check_collision:
            try:
                self.commander.check_hand_collision()
            except:
                self.commander.restoreState(state)
                raise NotImplementedError

        if not execute:
            return ret

        if not self.real and self.eval and not g.record:
            raw_input('after final pose: %s' % dir_vec)
        self.commander.avoidTrayCollision()
        if get_pre_grasp_state:
            state_id = self.commander.saveState()
        if grasp:
            self.commander.autoGrasp()
        return state_id if get_pre_grasp_state else dir_vec

    def is_graspit(self):
        return g.simulator == 'graspit'

    def step(self, a, info=None, next_s=False):
        s = info['s']
        if (a['terminate'] and not g.zoom_image) or info['t'] >= g.grasp_horizon:
            if info['t'] < g.grasp_horizon and g.zoom_image:
                raise NotImplementedError

            dir_vec = self.grasp(s, a)
            if self.eval:
                print('dir_vec', dir_vec)

            if g.zoom_image and self.real:
                cv2.imwrite(os.path.join(d.dirname, 'rgb_test_final.png'),
                            self.get_state(info['t'], get_rgb=True)['rgb'])
            quality = self.commander.computeQuality()
            collision = False

            if self.is_graspit():
                r = (quality.volume + quality.epsilon + 1) * g.quality_multiplier
            else:
                r = quality

            next_state = self.get_state(info['t'], get_rgb=True) if next_s else None
            if self.is_graspit():
                return next_state, r, True, {'volume': quality.volume, 'epsilon': quality.epsilon, 'dir_vec': dir_vec,
                                             'closest_obj_name': self.commander.closest_obj_name if not d.real else
                                             None}
            else:
                return next_state, r, True, {'success': quality, 'dir_vec': dir_vec, 'collision': collision,
                                             'closest_obj_name': self.commander.closest_obj_name if hasattr(
                                                 self.commander, 'closest_obj_name') else None,
                                             'no_force': int(info['t'] < g.grasp_horizon)}
        else:
            try:
                image, transform = self.get_cropped_image(s['real_depth'], a, s['transform'], s['max_depth'])
            except:
                copy_a = copy.copy(a)
                copy_a['terminate'] = True
                return self.step(copy_a, info)

            return {'image': image, 'transform': transform, 'real_depth': s['real_depth'], 'max_depth': s['max_depth'],
                    't': info['t'], 'rgb': s['rgb'] if next_s else None,
                    'scale': transform['dx'] * 2.0 / d.camera_w}, 0, False, {
                       'no_force': int(info['t'] < g.grasp_horizon)}

    def grasp_step(self, a, info=None, next_s=False, get_baseline=False):
        s = info['s']
        if (a['terminate'] and not g.zoom_image) or info['t'] >= g.grasp_horizon:
            if get_baseline and not self.real and g.eval_baseline:
                state_id = self.grasp(s, a, grasp=True, get_pre_grasp_state=True)
                r = self.commander.computeQuality(delete=False)
                if self.eval:
                    raw_input(str({'baseline': r}))
                self.commander.restoreState(state_id)
            else:
                dir_vec = self.grasp(s, a)
                if self.eval:
                    print('dir_vec', dir_vec)
                if g.zoom_image and self.real:
                    cv2.imwrite(os.path.join(d.dirname, 'rgb_test_final.png'),
                                self.get_state(info['t'], get_rgb=True)['rgb'])
                quality = self.commander.computeQuality()
                collision = False

                if self.is_graspit():
                    r = (quality.volume + quality.epsilon + 1) * g.quality_multiplier
                else:
                    r = quality

                next_state = self.get_state(info['t'], get_rgb=True) if next_s else None
                if self.is_graspit():
                    return next_state, r, True, {'volume': quality.volume, 'epsilon': quality.epsilon,
                                                 'dir_vec': dir_vec,
                                                 'closest_obj_name': self.commander.closest_obj_name if not d.real
                                                 else None}
                else:
                    return next_state, r, True, {'success': quality, 'dir_vec': dir_vec, 'collision': collision,
                                                 'closest_obj_name': self.commander.closest_obj_name if hasattr(
                                                     self.commander, 'closest_obj_name') else None,
                                                 'no_force': int(info['t'] < g.grasp_horizon)}
            return None, 0, True, {}
        else:
            try:
                image, transform = self.get_cropped_image(s['real_depth'], a, s['transform'], s['max_depth'])
            except:
                traceback.print_exc()
                copy_grasp_a = copy.copy(a)
                copy_grasp_a['terminate'] = True
                return self.grasp_step(copy_grasp_a, info)

            return {'image': image, 'transform': transform, 'real_depth': s['real_depth'], 'max_depth': s['max_depth'],
                    't': info['t'], 'rgb': s['rgb'] if next_s else None,
                    'scale': transform['dx'] * 2.0 / d.camera_w}, 0, False, {
                       'no_force': int(info['t'] < g.grasp_horizon)}

    def grasp_plan(self, a, info=None, next_s=False):
        s = info['s']
        if (a['terminate'] and not g.zoom_image) or info['t'] >= g.grasp_horizon:
            return self.grasp(s, a, grasp=False, execute=False, robot_state=info['robot_state']), 0, True, {}
        else:
            try:
                image, transform = self.get_cropped_image(s['real_depth'], a, s['transform'], s['max_depth'])
            except:
                traceback.print_exc()
                copy_grasp_a = copy.copy(a)
                copy_grasp_a['terminate'] = True
                return self.grasp_plan(copy_grasp_a, info)

            return {'image': image, 'transform': transform, 'real_depth': s['real_depth'], 'max_depth': s['max_depth'],
                    't': info['t'], 'rgb': s['rgb'] if next_s else None,
                    'scale': transform['dx'] * 2.0 / d.camera_w}, 0, False, {
                       'no_force': int(info['t'] < g.grasp_horizon)}
