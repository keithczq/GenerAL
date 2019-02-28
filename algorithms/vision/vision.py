import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import d, g


def build_feature_pyramid(feature_maps):
    num_channels = g.fpn_pyramid_chnls
    feature_pyramids = {}
    with tf.variable_scope('build_feature_pyramid'):
        with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(0.0001)):
            feature_pyramids['P%s' % len(feature_maps)] = slim.conv2d(feature_maps['block_layer%s' % len(feature_maps)],
                                                                      num_outputs=num_channels, kernel_size=[1, 1],
                                                                      stride=1,
                                                                      scope='build_P%s' % (len(feature_maps) + 1))

            feature_pyramids['P%s' % (len(feature_maps) + 1)] = slim.max_pool2d(
                feature_pyramids['P%s' % len(feature_maps)], kernel_size=[2, 2], stride=2,
                scope='build_P%s' % (len(feature_maps) + 2))
            # P6 is down sample of P5

            for layer in range(len(feature_maps) + 1, 1, -1):
                p, c = feature_pyramids['P' + str(layer)], feature_maps['block_layer' + str(layer - 1)]
                up_sample_shape = tf.shape(c)
                up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                             name='build_P%d/up_sample_nearest_neighbor' % (layer - 1))

                c = slim.conv2d(c, num_outputs=num_channels, kernel_size=[1, 1], stride=1,
                                scope='build_P%d/reduce_dimension' % (layer - 1))
                p = up_sample + c
                p = slim.conv2d(p, num_channels, kernel_size=[3, 3], stride=1, padding='SAME',
                                scope='build_P%d/avoid_aliasing' % (layer - 1))
                feature_pyramids['P' + str(layer - 1)] = p

    return feature_pyramids


def segmentation(input, scale_obv_2d):
    with tf.variable_scope('segmentation'):
        net = input
        feature_maps = {}
        net = slim.conv2d(net, 32, [9, 9], stride=[2, 2], activation_fn=tf.nn.relu, scope='backbone' + str(0))
        feature_maps['block_layer1'] = net
        net = slim.conv2d(net, 16, [7, 7], stride=[2, 2], activation_fn=tf.nn.relu, scope='backbone' + str(1))
        feature_maps['block_layer2'] = net
        net = slim.conv2d(net, 8, [5, 5], stride=[2, 2], activation_fn=tf.nn.relu, scope='backbone' + str(2))
        feature_maps['block_layer3'] = net
        net = slim.conv2d(net, 4, [3, 3], stride=[2, 2], activation_fn=tf.nn.relu, scope='backbone' + str(3))
        feature_maps['block_layer4'] = net
        print('net.shape', net.shape)
        if g.use_fpn:
            feature_pyramids = build_feature_pyramid(feature_maps)
            obvs = []
            for i in range(4 if g.fpn_all_pyramids else 1):
                feature_pyramid = feature_pyramids['P%s' % (i + 1)]
                num_convs = math.log(d.map_s / int(feature_pyramid.shape[1]), 2)
                print('num_convs', num_convs)
                num_convs = int(num_convs)
                num_channels = 8 / num_convs
                kernel = 11 - 2 * num_convs
                for j in range(num_convs):
                    feature_pyramid = slim.conv2d_transpose(feature_pyramid, num_channels, [kernel, kernel],
                                                            stride=[2, 2], activation_fn=tf.nn.relu,
                                                            scope='conv_pyramid_%s_conv_%s' % (i, j))
                    kernel += 2
                    num_channels *= 2
                obvs.append(feature_pyramid)
            obv = tf.concat(obvs, axis=3)

        else:
            net = slim.conv2d_transpose(net, 4, [3, 3], stride=[2, 2], activation_fn=tf.nn.relu,
                                        scope='backbone' + str(4))
            net = slim.conv2d_transpose(net, 8, [5, 5], stride=[2, 2], activation_fn=tf.nn.relu,
                                        scope='backbone' + str(5))
            net = slim.conv2d_transpose(net, 16, [7, 7], stride=[2, 2], activation_fn=tf.nn.relu,
                                        scope='backbone' + str(6))
            obv = slim.conv2d_transpose(net, 32, [9, 9], stride=[2, 2], activation_fn=tf.nn.relu,
                                        scope='backbone' + str(7))

        assert obv.shape[1:3] == [d.map_s, d.map_s], (obv.shape, d.map_s)
        position = slim.flatten(slim.conv2d(obv, 1, [3, 3], stride=[1, 1], activation_fn=tf.nn.relu,
                                            scope='conv_position_0')) * g.logit_multiplier

        out = []
        out.append(position)
        out.append(multi_conv_layer_branch(obv, 'terminate', activation=tf.nn.sigmoid))

        if g.crop_method:
            out.append(multi_conv_layer_branch(obv, 'scale', activation=tf.nn.sigmoid,
                                               scale_map=None if g.old else scale_obv_2d))
        if g.orientation == 8:
            out.append(multi_conv_layer_branch(obv, 'roll', activation=tf.nn.tanh, scale_map=scale_obv_2d))
            out.append(multi_conv_layer_branch(obv, 'pitch', activation=tf.nn.sigmoid, scale_map=scale_obv_2d))
            out.append(multi_conv_layer_branch(obv, 'yaw', activation=tf.nn.tanh, scale_map=scale_obv_2d))
        elif g.orientation == 4:
            out.append(multi_conv_layer_branch(obv, 'roll', activation=tf.nn.tanh, scale_map=scale_obv_2d))
        else:
            raise NotImplementedError

        spread = multi_conv_layer_branch(obv, branch='spread', activation=tf.nn.sigmoid,
                                         scale_map=None if g.old else scale_obv_2d)
        out += [spread]

        finger_1 = multi_conv_layer_branch(obv, branch='finger_1', activation=tf.nn.sigmoid,
                                           scale_map=None if g.old else scale_obv_2d)
        finger_2 = multi_conv_layer_branch(obv, branch='finger_2', activation=tf.nn.sigmoid,
                                           scale_map=None if g.old else scale_obv_2d)
        finger_3 = multi_conv_layer_branch(obv, branch='finger_3', activation=tf.nn.sigmoid,
                                           scale_map=None if g.old else scale_obv_2d)
        out += [finger_1, finger_2, finger_3]
        if g.hand == 'seed':
            finger_4 = multi_conv_layer_branch(obv, branch='finger_4', activation=tf.nn.sigmoid,
                                               scale_map=None if g.old else scale_obv_2d)
            out += [finger_4]

        return tf.stack(out, axis=2), slim.flatten(obv)


def multi_conv_layer_branch(obv, branch, activation, scale_map=None):
    net = obv
    for layer in range(g.branch_conv):
        net = slim.conv2d(net, 1, [3, 3], stride=[1, 1],
                          activation_fn=tf.nn.relu if 0 < layer < g.branch_conv - 1 else activation,
                          scope='conv_%s_%s' % (branch, layer))
        if g.scale_obv and scale_map is not None and (layer == 0):
            net = tf.concat((net, scale_map), axis=3)
    return slim.flatten(net)
