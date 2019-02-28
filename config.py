import collections
import itertools
import math
import multiprocessing
import os
import socket

import numpy as np
from pyquaternion import Quaternion

from staubli_barrett_ws.src.staubli_barrett_meta_package.barrett_hand_ros.rqt_bhand.src.rqt_bhand.tact_maps import \
    THRESHOLD_VALUE


def value(dict, k):
    for key in dict:
        if k == key or (not isinstance(key, str) and isinstance(key, collections.Iterable) and k in key):
            return dict[key]
    raise NotImplementedError(k)


def key(dict, v):
    for k in dict:
        if v == dict[k] or (not isinstance(k, str) and isinstance(k, collections.Iterable) and v in dict[k]):
            return k
    raise NotImplementedError(v)


def normalized_qua(quaternion):
    return quaternion / np.linalg.norm(quaternion)


def dire_vec_to_qua(direction_vector):
    v1 = [0, 0, 1]
    v2 = direction_vector / np.linalg.norm(direction_vector)
    assert np.allclose(np.linalg.norm(v2), 1), (direction_vector, v2)
    x, y, z = np.cross(v1, v2)
    w = math.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2)) + np.dot(v1, v2)
    return normalized_qua([x, y, z, w])


class g(object):
    ip = socket.gethostname()
    real = False
    short = {True: 'T', False: 'F', 'pybullet': 'pb', 'None': 'N', None: 'N', 'sigmoid': 'sig', 'relu': 're',
             'barrett': 'b', 'seed': 's', 'tactile': 't', 'position': 'p'}
    abbr = {('t', 'position'): 'tactile', 'p': 'position', 's': 'seed', ('2', 'pb', 'b'): 'barrett'}
    path = None
    algo = 'position'
    hand = 'seed'

    # NOTE: MAT ablation
    planning_baseline = False
    finger_closing_only = False or planning_baseline
    regrasp_only = False
    pos_adjustment_only = False
    ori_adjustment_only = False

    # NOTE: SAC
    tau = 0.01
    alpha = 5e-4
    direct_v_network_concat = True
    weighted_mse = False
    calibration_noise = 0

    update_target_v_interval = 1
    monte_carlo_q_target = True
    off_policy = False
    on_policy = not off_policy
    replay_buffer_num_batch_mult = 10
    regrasp_r_penalty = 0.05

    max_train_tactile_delta = 0.4
    min_train_tactile_delta = 0.1
    eval_tactile_delta = .2
    dynamic_timestep = True
    delta_tolerance = 0.01
    repost_point_cloud_interval = value({'barrett': 10, 'seed': 100}, hand)

    demo_position = False
    tactile_delta = True
    tactile_contacts = True
    tactile_positions = True
    tactile_cartesians = True
    tactile_velocities = False
    tactile_forces = tactile_torques = False
    tactile_efforts = False
    history = 10
    random_reset_body = False
    check_history = 5
    translation_planning = True
    tactile_obv_names = ['contacts', 'positions', 'cartesians', 'velocities', 'forces', 'torques', 'efforts']

    lift_ts = 500
    regrasp = True
    tactile_regrasp = True and regrasp
    visual_regrasp = not tactile_regrasp
    adjust_distance = 0.05
    additional_pos_adj = 0
    num_adjust_logits = 20
    delta_history = (history - 1) * tactile_delta
    sim_tactile_threshold = .8
    real_tactile_threshold = THRESHOLD_VALUE
    custom_obj_scale = value({'barrett': 1, 'seed': .6}, hand)
    home_joint_position = [1.2058508962471073, -1.0744060459765796, -0.369084810203193, -3.1399431521321377,
                           1.7449314195713546, -2.339314460070107]
    tactile_act_shift = 0.1
    tactile_logit_multiplier = 1
    plot_tactile = False
    plot_torques = False
    step_by_step = False
    eval_baseline = False
    tactile = (algo == 'tactile')
    old = False
    check_collision = False
    use_saved_envs = False
    branch_conv = 3
    video = (hand == 'barrett')
    statistics = (ip in ('mainland', 'long', 'instance-2', 'instance-1'))

    save_pybullet_state = False
    if hand == 'barrett':
        real_dist_away = 0.04 if not real else 0.14
    else:
        real_dist_away = 0.03 if real else 0.4
    simulator = 'pybullet'

    if algo == 'position':
        num_actors_per_server = multiprocessing.cpu_count() if 'instance' in ip else int(value(
            {'long': 8, 'skye': 4, ('curacao', 'darcy', 'tong', 'Bohans-MBP'): 6, 'mainland': 7, 'fyn': 3, 'syros': 8,
             ('delfino', 'comoros', 'Bohans-MacBook-Pro.local'): 10 + real}, ip))
    else:
        num_actors_per_server = multiprocessing.cpu_count() if 'instance' not in ip else 36
    servers = ['localhost']
    num_actors = num_actors_per_server * len(servers)

    bs = value({'position': 500, 'tactile': 500}, algo)
    if off_policy and tactile:
        bs = 2000
    tactile_pi_mini_bs = 350
    tactile_v_mini_bs = 200
    angle_div = 1
    scale_obv = True
    fd_div = 320
    robot_lateral_friction_seen = 1
    robot_lateral_friction_novel = robot_lateral_friction_seen
    x_axis_dirvec = dire_vec_to_qua([1, 0, 0])
    # NOTE: pixel attentive ablation
    learn_attention = True
    learn_spread = True
    orientation = 8
    top_down = False
    grasp_horizon = float('inf')
    tactile_horizon = 250
    if hand == 'seed':
        max_grasp_force_mult = value({'position': 1000, 'tactile': 500}, algo)
    else:
        max_grasp_force_mult = value({'position': 50, 'tactile': 500}, algo)
    flexible_dof = True
    summary_horizon = 10
    logit_multiplier = 1
    top_k = None
    check_tactile_baseline = 1
    best_terminate = False
    hand_mass = 1
    table_friction = 0.3
    # default = 0.05
    square = True
    linearDamping = 20
    square_type = 'min'
    crop_method = 1
    # log_std_offset = 3.5
    log_std_offset = 4
    horizon_obv = False
    use_batchnorm = False
    min_num_bodies = 95
    max_num_bodies = 100
    use_grasp_bl = False
    sac = True
    reinit_pybullet_depth_threshold_novel = value({'barrett': 0.08, 'seed': 0.2}, hand)
    reinit_pybullet_depth_threshold_seen = value({'barrett': 0.08, 'seed': 0.2}, hand)
    use_bl = value({'position': use_grasp_bl, 'tactile': sac}, algo)
    quality_multiplier = 1
    finger_distance_mult = 0
    fpn_pyramid_chnls = 4
    min_depth = True
    dist_min_depth = False
    approach_to_contact = False
    safe_train_r = .1
    use_fpn = True
    fpn_all_pyramids = True
    optim_epochs = 10
    if hand == 'barrett':
        robot_urdf = os.path.expanduser('./pybullet_commander/staubli_barrett_hand_ws/src/staubli_barrett_ws'
                                        '/staubli_barretthand_description/urdf/staubli_bhand.urdf')
    else:
        robot_urdf = os.path.expanduser('./pybullet_commander/seed/hand_model/urdf/ur5_seed_pybullet.urdf')
    load_second_robot = False
    second_robot_urdf = os.path.expanduser('./pybullet_commander/staubli_barrett_hand_ws/src/staubli_barrett_ws'
                                           '/staubli_barretthand_description/urdf/staubli_bhand.urdf')
    num_eval_episodes = 500 if statistics else num_actors
    num_eval_episodes_per_actor = num_eval_episodes // num_actors
    hosts = list(itertools.chain(*[[server] * num_actors_per_server for server in servers]))
    ports = list(itertools.chain(*[range(11311, 11311 + num_actors_per_server) for server in servers]))

    gamma = value({'tactile': 0.999, 'position': 0.99}, algo)
    lam = 0.95
    clip_param = 0.2
    dist_away = 0.3
    safe_simulator_z = 0.24 if hand == 'barrett' else 0.13
    safe_real_z = -0.21 if hand == 'barrett' else -0.215

    # NOTE: VF
    qv_hid_fc_size = 128
    v_lr = 1e-4
    qv_n_layers = 3

    # NOTE: VF
    q_lr = v_lr

    ros_cartesian = True

    # NOTE: grasp
    grasp_cnn = 'segmentation'
    grasp_lr = 1e-4
    grasp_mini_bs = (int(value({('instance-1', 'instance-2', 'instance-3'): 50, ('skye', 'tong', 'darcy'): 100,
                                ('delfino', 'mainland', 'Bohans-MBP', 'long', 'comoros', 'curacao'): 96, 'syros': 96,
                                'fyn': 64, 'Bohans-MacBook-Pro.local': 10}, ip)) - int(hand == 'seed') * 16)

    # NOTE: Tactile grasp adjuster
    tactile_pi_n_layers = 2
    tactile_pi_hid_fc_size = qv_hid_fc_size
    tactile_embedding_size = 128
    tactile_pi_lr = 1e-4

    mini_bs = tactile_pi_mini_bs if tactile else grasp_mini_bs
    replay_buffer_queue_size = bs // mini_bs * replay_buffer_num_batch_mult

    # Overall
    num_iters = 2000
    record = False and not real
    record_seed_video = False

    teleoperation = False
    bound = 0.2
    test_bound = False
    force_grasp = True
    zoom_image = False
    horizontal = False

    gradclip = 200
    heatmaps = ['position', 'zoom', 'scale', 'roll', 'pitch', 'yaw', 'spread', 'finger_1', 'finger_2', 'finger_3',
                'finger_4']
    if path and not zoom_image and not real:
        assert 'hor-%s' % grasp_horizon in path
        assert ('checkpoints' in path) or ('fri=%s-' % robot_lateral_friction_seen in path)
        assert ('checkpoints' in path) or ('fr=%s-' % max_grasp_force_mult in path)
        assert ('checkpoints' in path) or ('TT-%s' % sim_tactile_threshold in path) or algo == 'position'
        print('grasp_horizon sim_tactile_threshold lift_ts max_grasp_force_mult robot_lateral_friction_seen '
              'regrasp_prob_offset checked!')
    curriculum_learning = True
    # automatic
    total_ts = num_iters * bs
    pi_replay_buffer_batches = bs // grasp_mini_bs
    gpu = True
    tactile_dim = history * (
            tactile_contacts * 96 + tactile_positions * 8 + tactile_velocities * 8 + tactile_forces * 18 +
            tactile_efforts * 3 + tactile_cartesians * 96 * 3)
    user_dict = 'bohan'
    server_user = 'bohan'
    client_user = 'bohan'
    pybullet_obj_dir = os.path.expanduser('../graspit/meshes')
    raw_obj_scale_original = {'009_gelatin_box': 2, '019_pitcher_base': 0.8, '063-b_marbles': 2, '016_pear': 2,
                              '015_peach': 2, '012_strawberry': 2, '073-b_lego_duplo': 2, '073-c_lego_duplo': 2,
                              '073-d_lego_duplo': 2, '073-e_lego_duplo': 2, '070-b_colored_wood_blocks': 2,
                              '073-f_lego_duplo': 2, '073-a_lego_duplo': 2, '038_padlock': 2, '058_golf_ball': 2,
                              '050_medium_clamp': 2, '072-d_toy_airplane': 2, '072-e_toy_airplane': 2,
                              '037_scissors': 2, '065-a_cups': 2, '014_lemon': 2, '062_dice': 3, '030_fork': 2,
                              '072-c_toy_airplane': 2, '055_baseball': 2, '018_plum': 2, '057_racquetball': 1.5,
                              '040_large_marker': 2.2, '031_spoon': 2, '042_adjustable_wrench': 2, '065-e_cups': 2,
                              '032_knife': 2}
    raw_obj_scale_KIT = {'Curry_800_tex': 1.2}
    obj_scale = {}
    for i in raw_obj_scale_original:
        obj_scale[os.path.join('original', i)] = raw_obj_scale_original[i]
    for i in raw_obj_scale_KIT:
        obj_scale[os.path.join('KIT', i)] = raw_obj_scale_KIT[i]

    original_objs = [os.path.join('original', i) for i in os.listdir('original')]
    KIT_objs = list(set(
        [os.path.join('KIT', os.path.splitext(i)[0]) for i in os.listdir(os.path.join(pybullet_obj_dir, 'KIT')) if
         i.endswith('.obj')]))

    novel_objs = list(set(
        [os.path.join('BigBIRD', os.path.splitext(i)[0]) for i in os.listdir(os.path.join(pybullet_obj_dir, 'BigBIRD'))
         if i.endswith('.obj')]))
    seen_objs = original_objs
    stabuli_home_x = -0.3
    stabuli_home_y = -0.73
    real_home = value(
        {'barrett': [stabuli_home_x, stabuli_home_y, 0.0510038, -0.2004434, 0.97943445, 0.01879157, 0.01332207],
         'seed': [-0.4, -0.75, 0.09, 0.283871499548, -0.918802512869, 0.274158316607, -0.00749209756086]}, hand)
    eval_single_object = False


class d(object):
    ip = socket.gethostname()
    real = False
    depth = True
    num_chnls = 1
    rand_p = True
    depth_scale = 1.0
    depth_bias = 0.0
    pos_delta = .1
    ori_delta = 20
    config = g()
    e = 1e-10
    l = 2000
    tmp_dir = os.path.expanduser('~/.graspit/images/tmp.jpg')

    server_graspit_img_dir = os.path.dirname(tmp_dir).replace(g.client_user, g.server_user)
    tmp_depth_dir = '%s_depth.npy' % os.path.splitext(tmp_dir)[0]
    dirname = os.path.dirname(tmp_dir)
    pybullet_env_dir = '../graspit/bullet_states'
    pybullet_obj_dir = os.path.expanduser('../graspit/meshes')
    img_s = 224
    map_s = img_s
    img_sp = (img_s, img_s)
    debug = False
    camera_h = 480.0 if real else 684.0
    camera_w = 640.0 if real else 1000.0
    cx = camera_w / 2.0 if not real else 319.5
    cy = camera_h / 2.0 if not real else 239.5

    if config.simulator == 'graspit':
        fx = 825.661
        fy = 825.661
        camera_to_world_qua = Quaternion(0.510746, 0.494626, 0.482173, 0.511852)
        camera_to_world_tra = np.array([512.727, 11.2119, 92.4714]) / 1000.0
    elif real:
        fx = 525.0
        fy = 525.0
        # NOTE: finger
        # NOTE: arm
        if g.hand == 'barrett':
            camera_to_world_tra = np.array([-0.732, 0.105, 0.763])
            camera_to_world_qua = Quaternion([0, 0.690, 0.724, -0.000])
        else:
            camera_to_world_tra = np.array([0.418, -0.322, 0.774])
            camera_to_world_qua = Quaternion([0.010, 0.925, 0.379, -0.003])
        camera_dist = 1.2
        camera_pitch_deg = 0
    else:
        if g.hand == 'barrett':
            camera_yaw_deg, camera_pitch_deg, roll_deg = 0, -89, 0
            camera_dist = 1.2
        else:
            camera_yaw_deg, camera_pitch_deg, roll_deg = 0, -89, 0
            camera_dist = .5
        camera_to_world_tra = np.array([0, 0, camera_dist])
        camera_to_world_qua = Quaternion([0, -1, 0, 0])
        pybullet_height = 768
        fov = 60
        fx = pybullet_height / (2 * math.tan(fov * math.pi / 360))
        fy = pybullet_height / (2 * math.tan(fov * math.pi / 360))

    camera_to_world_rot = camera_to_world_qua.rotation_matrix
    camera_to_world_tran = np.array(
        [[camera_to_world_rot[0][0], camera_to_world_rot[0][1], camera_to_world_rot[0][2], camera_to_world_tra[0]],
         [camera_to_world_rot[1][0], camera_to_world_rot[1][1], camera_to_world_rot[1][2], camera_to_world_tra[1]],
         [camera_to_world_rot[2][0], camera_to_world_rot[2][1], camera_to_world_rot[2][2], camera_to_world_tra[2]],
         [0, 0, 0, 1]])


def n():
    if g.algo == 'position':
        return '%s-%s-ts=%s-%s-lr=%s-hor-%s' \
               '-ori=%s-%s-%s-d=%s%s%s-bl=%s-%s-gc=%s-obj=%s-%s-%s-%s-cm=%s' \
               '-cc=%s-at=%s-fpn=%s-%s-%s-sprd=%s-g=%s-ad=%s-bt=%s-off=%s-ssz=%s' \
               '-ffd=%s-fd=%s-fri=%s-br=%s-env=%s-ap=%s-cl=%s-fr=%s-tf=%s-so=%s-lm=%s-p=%s-%s-%s' % (
                   g.short[g.algo], g.short[g.hand], g.bs, g.grasp_mini_bs, g.grasp_lr, g.grasp_horizon, g.orientation,
                   g.real_dist_away, g.dist_away, g.short[g.min_depth], g.short[d.depth], g.short[g.dist_min_depth],
                   g.short[g.use_bl], d.img_s, g.gradclip, g.max_num_bodies, g.short[g.horizon_obv], g.safe_train_r,
                   g.optim_epochs, g.crop_method, g.short[g.check_collision], g.short[g.learn_attention],
                   g.short[g.use_fpn], g.fpn_pyramid_chnls, g.short[g.fpn_all_pyramids], g.short[g.learn_spread],
                   g.gamma, g.angle_div, g.short[g.best_terminate], g.log_std_offset, g.safe_simulator_z,
                   g.short[g.flexible_dof], g.fd_div, g.robot_lateral_friction_seen, g.branch_conv,
                   g.short[g.use_saved_envs], g.short[g.approach_to_contact], g.short[g.curriculum_learning],
                   g.max_grasp_force_mult, g.table_friction, g.short[g.scale_obv], g.logit_multiplier,
                   d.camera_pitch_deg if hasattr(d, 'camera_pitch_deg') else '', g.custom_obj_scale, d.camera_dist)
    elif g.algo == 'tactile':
        return '%s-%s-%s-%s-%s-%s-%s-%s-%s%s%s%s%s%s%s-bs=%s-lr=%s-%s-hor-%s-%s' \
               '-ori=%s-%s-%s-gc=%s-%s-%s-%s-%s' \
               '-at=%s-g=%s-ad=%s-bt=%s-%s' \
               '-fd=%s-fri=%s-br=%s-for=%s-tf=%s-lm=%s-p=%s-tts=%s-%s-s=%s-rg=%s%s-%s-%s-%s-%s-ta=%s-%s-op' \
               '=%s%s%s%s-%s-%s-%s-%s-%s-%s-%s' \
               '' % (g.short[g.algo], g.short[g.hand], g.tactile_pi_n_layers, g.qv_n_layers, g.tactile_pi_hid_fc_size,
                     g.tactile_embedding_size, g.tactile_dim, g.history, g.short[g.tactile_delta],
                     g.short[g.tactile_contacts], g.short[g.tactile_positions], g.short[g.tactile_cartesians],
                     g.short[g.tactile_velocities], g.short[g.tactile_forces], g.short[g.tactile_efforts],
                     g.tactile_pi_mini_bs, g.v_lr, g.tactile_pi_lr, g.grasp_horizon, g.tactile_horizon, g.orientation,
                     g.real_dist_away, g.dist_away, g.gradclip, g.max_num_bodies, g.short[g.horizon_obv],
                     g.safe_train_r, g.optim_epochs, g.short[g.learn_attention], g.gamma, g.angle_div,
                     g.short[g.best_terminate], g.safe_simulator_z, g.fd_div, g.robot_lateral_friction_seen,
                     g.branch_conv, g.max_grasp_force_mult, g.table_friction, g.tactile_logit_multiplier,
                     d.camera_pitch_deg if hasattr(d, 'camera_pitch_deg') else '', g.max_train_tactile_delta,
                     g.min_train_tactile_delta, g.tactile_act_shift, g.short[g.regrasp], g.short[g.tactile_regrasp],
                     g.sim_tactile_threshold, g.num_adjust_logits, g.adjust_distance, g.lift_ts, g.tau, g.alpha,
                     g.short[g.off_policy], g.short[g.monte_carlo_q_target], g.short[g.weighted_mse],
                     g.short[g.translation_planning], g.update_target_v_interval, g.replay_buffer_num_batch_mult,
                     g.regrasp_r_penalty, g.additional_pos_adj, g.delta_tolerance, g.check_history, g.calibration_noise)
    else:
        raise NotImplementedError
