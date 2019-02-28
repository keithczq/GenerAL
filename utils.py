import math
import random
import traceback

try:
    from queue import Queue
except:
    from Queue import Queue

import cv2
import geometry_msgs.msg
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

from config import d, g


def _random(scale):
    return random.random() * scale * 2 - scale


def dynamicTactileDelta(accuracy):
    return float(g.min_train_tactile_delta + (g.max_train_tactile_delta - g.min_train_tactile_delta) * (
            1 - accuracy)) if g.dynamic_timestep else g.max_train_tactile_delta


def vector2GeometryMsgs(position, orientation):
    pose = geometry_msgs.msg.Pose()
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = orientation
    return pose


def _position_random(scale):
    return random.random() * scale


def get_point_clouds_from_depth(depth, w, h):
    point_d = depth[:, :, 0]
    x = ((w - d.cx) / d.fx * point_d).flatten()
    y = ((h - d.cy) / d.fy * point_d).flatten()
    z = point_d.flatten()
    ones = np.ones_like(z)
    camera_pos = np.stack((x, y, z, ones), axis=0)
    point_clouds = np.transpose(np.matmul(d.camera_to_world_tran, camera_pos)[:3, :])

    return point_clouds


def quaternion_to_directional_vector_ori(orientation):
    q0, q1, q2, q3 = orientation.w, orientation.x, orientation.y, orientation.z

    rotation_matrix = np.array([[1 - 2 * q2 ** 2 - 2 * q3 ** 2, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
                                [2 * q1 * q2 - 2 * q0 * q3, 1 - 2 * q1 ** 2 - 2 * q3 ** 2, 2 * q2 * q3 + 2 * q0 * q1],
                                [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, 1 - 2 * q1 ** 2 - 2 * q2 ** 2]])
    original = np.array([0, 0, 1])
    ret = np.matmul(original, rotation_matrix)
    ret /= np.linalg.norm(ret)
    return ret


def pose2PosOri(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z]), np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])


def upsample_cnn(input):
    with tf.variable_scope('FCN'):
        net = input
        print(net.shape)
        net = slim.conv2d(net, 32, kernel_size=[7, 7], scope='conv1', stride=2)
        print(net.shape)
        net = slim.conv2d(net, 16, kernel_size=[5, 5], scope='conv2', stride=2)
        print(net.shape)
        net = slim.conv2d(net, 8, kernel_size=[3, 3], scope='conv3', stride=2)
        print(net.shape)

        net = slim.conv2d_transpose(net, 8, kernel_size=[3, 3], scope='deconv3', stride=2)
        print(net.shape)
        net = slim.conv2d_transpose(net, 16, kernel_size=[5, 5], scope='deconv2', stride=2)
        print(net.shape)
        net = slim.conv2d_transpose(net, 32, kernel_size=[7, 7], scope='deconv1', stride=2)
        print(net.shape)

        net = slim.conv2d(net, 3, kernel_size=[2, 2], scope='ori')
        print(net.shape)
    return net


def generate_seg(dones, rews, baseline):
    seg = {}
    seg['new'] = [1] + dones[:-1]
    seg['rew'] = rews
    seg['vpred'] = baseline
    seg['nextvpred'] = seg['vpred'][-1] * (1 - seg['new'][-1])
    return seg


def add_vtarg_and_adv(seg):
    """
    Compute target value using TD(lambda) estimator and advantage with GAE(
    lambda)
    """
    new = np.append(seg["new"], 0)
    # NOTE: last element is only used for last vtarg, but we
    # already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = seg["rew"].shape[0]
    seg["adv"] = np.empty(T, 'float32')
    seg['cum_r'] = np.empty(T, 'float32')
    rew = seg["rew"]
    cum_r = 0
    for t in reversed(range(T)):
        if new[t + 1]:
            cum_r = 0
        seg['cum_r'][t] = cum_r * g.gamma + rew[t]
        cum_r = seg['cum_r'][t]
        seg["adv"][t] = cum_r - vpred[t]


def normalize_heatmap(image, index=0):
    if index in [0]:
        min = np.min(image)
        max = np.max(image)
    else:
        min = -np.pi
        max = np.pi
        if index in [3, 4, 5]:
            image *= np.pi
        elif index == 6:
            image *= np.pi / 2
        elif index in [7, 8, 9]:
            image *= .61
    # image = np.minimum(image, np.pi)
    # image = np.maximum(image, -np.pi)
    ret = (image - min) * 255.0 / (max - min + d.e)
    if index == 1:
        ret = 255.0 - ret
    blur = 3
    kernel = 5
    ret = np.repeat(ret.astype(np.uint8), repeats=3, axis=2)
    ret = cv2.GaussianBlur(ret, (kernel, kernel), blur, blur)
    ret = cv2.equalizeHist(ret[:, :, 0])
    ret = cv2.GaussianBlur(ret, (kernel, kernel), blur, blur)
    ret = cv2.applyColorMap(ret, cv2.COLORMAP_JET)

    return ret


def normalize_depth(image, fixed=True):
    try:
        if fixed:
            image = np.minimum(image, d.camera_dist)
        min = d.camera_dist / 2 if fixed else np.min(image)
        max = (d.camera_dist + (
            0.1 if (hasattr(d, 'camera_pitch_deg') and d.camera_pitch_deg <= -80) else 0.2)) if fixed else np.max(image)
        ret = (image - min) * 255.0 / (max - min + d.e)
        return ret
    except:
        traceback.print_exc()
        import ipdb
        ipdb.set_trace()
        print(image.shape)
        raise NotImplementedError


def explained_variance(ypred, y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    # ipdb.set_trace()
    # print(ypred.shape, y.shape)
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def bilinear_interpolation(x, y, points):
    points = sorted(points)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    assert ((x2 - x1) * (y2 - y1) + 0.0) != 0, 'I am here'
    return (q11 * (x2 - x) * (y2 - y) + q21 * (x - x1) * (y2 - y) + q12 * (x2 - x) * (y - y1) + q22 * (x - x1) * (
            y - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()


def check_action(action):
    assert len(action) == 7
    orientation = action[3:]
    assert np.allclose(np.linalg.norm(orientation), 1), orientation
    return action


def announce_usage_start(gpu_ram_GB, num_training_epochs, num_training_examples, batch_size, hours):
    from slackclient import SlackClient
    import socket, getpass
    slack_token = ''
    sc = SlackClient(slack_token)
    message = "On `%s`, `%s` has started a training job. They are using `%s GB` of GPU RAM. They have" \
              " `%s` training epochs, `%s` training examples, `%s` batch size, and will take approximately `%s` " \
              "hours." % (
                  socket.gethostname(), getpass.getuser(), gpu_ram_GB, num_training_epochs, num_training_examples,
                  batch_size, hours)
    sc.api_call("chat.postMessage", channel="machine_status", text=message)


def announce_usage_end():
    from slackclient import SlackClient
    import socket, getpass
    slack_token = ''
    sc = SlackClient(slack_token)
    message = "Job on `%s` by `%s` has been completed." % (socket.gethostname(), getpass.getuser())
    sc.api_call("chat.postMessage", channel="machine_status", text=message)


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


def vector_l2_loss(pred, gt, name):
    return tf.reduce_mean(tf.reduce_sum(tf.square(gt - pred), axis=1), name=name)


def get_robot_state(commander):
    robot = commander.getRobot(0).robot
    print(robot)
    return [robot.position.x, robot.position.y, robot.position.z, robot.orientation.x, robot.orientation.y,
            robot.orientation.z, robot.orientation.w]


def normalized_qua(quaternion):
    return quaternion / np.linalg.norm(quaternion)


def img_parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=d.num_chnls)
    return tf.image.resize_images(image_decoded, [d.img_s, d.img_s])


def normalized_action(action):
    pos = action[:3]
    ori = action[3:]
    ori /= np.linalg.norm(ori)
    return check_action(np.concatenate([pos, ori], axis=0))


def dire_vec_to_qua(direction_vector):
    v1 = [0, 0, 1]
    v2 = direction_vector / np.linalg.norm(direction_vector)
    assert np.allclose(np.linalg.norm(v2), 1), (direction_vector, v2)
    x, y, z = np.cross(v1, v2)
    w = math.sqrt(np.linalg.norm(v1) * np.linalg.norm(v2)) + np.dot(v1, v2)
    return normalized_qua([x, y, z, w])


def quaternion_multiply(quaternion0, quaternion1):
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0, -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0, -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, ], dtype=np.float64)


def wxyz_to_xyzw(w, x, y, z):
    return np.array([x, y, z, w])


def xyzw_to_wxyz(x, y, z, w):
    return np.array([w, x, y, z])


def qua_to_dirvec(w, x, y, z):
    rotation_matrix = np.array([[1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y],
                                [2 * x * y - 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z + 2 * w * x],
                                [2 * x * z + 2 * w * y, 2 * y * z - 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]])
    original = np.array([0, 0, 1])
    ret = np.matmul(original, rotation_matrix)
    ret = ret / np.linalg.norm(ret)
    return ret


def gen_example(commander):
    robot_pose = get_pose(0, 0, 0, 0, 0, 0, 1)
    commander.setRobotPose(robot_pose, 0)

    a = normalized_action(
        [0, 0, 0.2, _random(d.ori_delta), _random(d.ori_delta), _random(d.ori_delta), _random(d.ori_delta)])

    body_pose = get_pose(*check_action(a))
    commander.setBodyPose(0, body_pose)
    # evaluate
    commander.approachToContact()
    commander.approachToContact(-random.random() * 100)
    commander.autoGrasp()

    _state = []
    sensor_poses = commander.getRobot(0).robot.tactile.sensor_poses
    for i in range(len(commander.getRobot(0).robot.tactile.sensor_poses)):
        sensor_pose = sensor_poses[i].pose
        _state += [sensor_pose.position.x, sensor_pose.position.y, sensor_pose.position.z, sensor_pose.orientation.x,
                   sensor_pose.orientation.y, sensor_pose.orientation.z, sensor_pose.orientation.w]
    quality = commander.computeQuality()
    _value = [quality.volume, quality.epsilon]
    _image = commander.getImage(shape=d.img_sp)
    commander.autoOpen()
    return _state, _value, _image


def np_to_tfrecords(X, Y, file_path_prefix, verbose=True):
    """
    Converts a Numpy array (or two Numpy arrays) into a tfrecord file.
    For supervised learning, feed training inputs to X and training labels to Y.
    For unsupervised learning, only feed training inputs to X, and feed None to Y.
    The length of the first dimensions of X and Y should be the number of samples.

    Parameters
    ----------
    X : numpy.ndarray of rank 2
        Numpy array for training inputs. Its dtype should be float32, float64, or int64.
        If X has a higher rank, it should be rshape before fed to this function.
    Y : numpy.ndarray of rank 2 or None
        Numpy array for training labels. Its dtype should be float32, float64, or int64.
        None if there is no label array.
    file_path_prefix : str
        The path and name of the resulting tfrecord file to be generated, without '.tfrecords'
    verbose : bool
        If true, progress is reported.

    Raises
    ------
    ValueError
        If input type is not float (64 or 32) or int.

    """

    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2  # If X has a higher rank,
    # it should be rshape before fed to this function.
    assert isinstance(Y, np.ndarray) or Y is None

    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_x = _dtype_feature(X)
    if Y is not None:
        assert X.shape[0] == Y.shape[0]
        assert len(Y.shape) == 2
        dtype_feature_y = _dtype_feature(Y)

        # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))

    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in range(X.shape[0]):
        x = X[idx]
        if Y is not None:
            y = Y[idx]

        d_feature = {}
        d_feature['X'] = dtype_feature_x(x)
        if Y is not None:
            d_feature['Y'] = dtype_feature_y(y)

        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    if verbose:
        print("Writing {} done!".format(result_tf_file))


def get_dataset_size(tf_records_filenames):
    c = 0
    for fn in tf_records_filenames:
        for _ in tf.python_io.tf_record_iterator(fn):
            c += 1
    return c


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)

    return _initializer


def dnn(input, output_size, scope, n_layers, size, trainable, hid_init, final_init, final_activation_fn_str,
        reuse=False, final_bias_init=None, hid_act_fn_str='relu'):
    out = input

    final_act_fn = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid, 'None': None}[
        final_activation_fn_str]
    hid_act_fn = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh, 'sigmoid': tf.nn.sigmoid, 'None': None}[hid_act_fn_str]

    with tf.variable_scope(scope):
        for i in range(n_layers):
            out = layers.fully_connected(out, size, normalizer_fn=None, activation_fn=hid_act_fn, scope='hid_%s' % i,
                                         trainable=trainable, weights_initializer=hid_init, reuse=reuse)

        if final_bias_init is None:
            out = layers.fully_connected(out, output_size, normalizer_fn=None, activation_fn=final_act_fn,
                                         scope='final', trainable=trainable, weights_initializer=final_init,
                                         reuse=reuse)
        else:
            out = layers.fully_connected(out, output_size, normalizer_fn=None, activation_fn=final_act_fn,
                                         trainable=trainable, weights_initializer=final_init, scope='final',
                                         biases_initializer=final_bias_init, reuse=reuse)
    return out


def parse_fn(example):
    dict = {'X': tf.FixedLenFeature(672, tf.float32), 'Y': tf.FixedLenFeature(2, tf.float32)}
    parsed = tf.parse_single_example(example, dict)
    return parsed['X'], parsed['Y']


def transform_coord(cx, cy, transform):
    return transform['cx'] - transform['dx'] + cx, transform['cy'] - transform['dy'] + cy


def perc_to_index_tranform(range, value):
    return max(min(value * range, range - 1), 0)


def get_global_local_dx_dy(local_cx, local_cy, local_h, local_w, scale):
    if g.crop_method == 0:
        dx = min(local_cx, local_w - local_cx)
        dy = min(local_cy, local_h - local_cy)
        l = min(dx, dy)
        dx, dy = l, l
    elif g.crop_method == 1:
        dx = scale * local_w / 2
        dy = scale * local_h / 2
    else:
        raise NotImplementedError
    return dx, dy


def get_local_cx_cy(pixel, w, h):
    py = float(int(pixel) / d.img_s) / d.img_s
    px = float(int(pixel) % d.img_s) / d.img_s
    cy = perc_to_index_tranform(h, py)
    cx = perc_to_index_tranform(w, px)
    return cx, cy


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector
