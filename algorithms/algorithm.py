import json
import logging
import os
import shutil
import subprocess
import time
import traceback

import numpy as np
import ray
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import d, g, n


class algorithm(object):
    def delete_prev_directory(self):
        if not os.path.exists('results'):
            os.mkdir('results')
        if g.path is not None and self.dir_name == os.path.basename(g.path):
            return
        if os.path.exists(os.path.join('results', self.dir_name)):
            print("DELETING...", self.dir_name)
            shutil.rmtree(os.path.join('results', self.dir_name), ignore_errors=True)
            self.dir_name += '-'
            if os.path.exists(os.path.join('results', self.dir_name)):
                print("DELETING...", self.dir_name)
                shutil.rmtree(os.path.join('results', self.dir_name), ignore_errors=True)
        else:
            if os.path.exists(os.path.join('results', self.dir_name + '-')):
                print("DELETING...", self.dir_name + '-')
            shutil.rmtree(os.path.join('results', self.dir_name + '-'), ignore_errors=True)

        os.mkdir(os.path.join('results', self.dir_name))
        shutil.copyfile('config.py', os.path.join('results', self.dir_name, 'config.py'))

    def is_grasp_master(self):
        return not self.id

    def is_tactile_master(self):
        return not self.id

    def is_master(self):
        return self.is_grasp_master() or self.is_tactile_master()

    def __init__(self, id=0, eval=False, dirname=None, real=False):
        self.pi_replay_buffer = None
        self.id = id
        self.tensorboard_keys = []
        self.dir_name = n() if self.is_master() else dirname
        print('self.dir_name', self.dir_name)
        self.fd = {}
        self.t = 0
        self.env = None
        self.actors = None
        self.effective_timesteps_so_far = 0
        self.real_timesteps_so_far = 0
        self.eval = eval
        self.real = real
        self.grasp_scale_obv_2d = None
        self.max_train_success_perc = 0

        if self.is_master() and not self.eval:
            self.delete_prev_directory()
            self.custom_init_op()
            self.summary_phs = {}
            for tb_key in self.tensorboard_keys:
                key = tb_key.split('/')[-1]
                if tb_key.endswith('_tr'):
                    tf.summary.scalar(tb_key[:-3], getattr(self, key))
                else:
                    self.summary_phs[tb_key] = tf.placeholder(tf.float64, shape=(), name=tb_key)
                    tf.summary.scalar(tb_key, self.summary_phs[tb_key])

            self.summary_op = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(logdir=os.path.join('results', self.dir_name),
                                                     graph=tf.get_default_graph())
        else:
            self.custom_init_op(host=g.hosts[self.id - 1], port=g.ports[self.id - 1])

        self.sess = tf.Session()
        self.init_op = tf.global_variables_initializer()
        if len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=g.grasp_cnn)) > 0:
            if self.eval:
                self.grasp_saver = tf.train.Saver(
                    var_list=[var for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=g.grasp_cnn) if
                              'finger_4' not in var.name and 'log_std' not in var.name])
            else:
                self.grasp_saver = tf.train.Saver(
                    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=g.grasp_cnn))
        self.saver = tf.train.Saver()
        self.sess.graph.finalize()

        if self.is_master():
            self.eval_op() if self.eval else self.master()

    def test(self):
        raise NotImplementedError

    def eval_op(self):
        with tf.Graph().as_default():
            self.sess.run(self.init_op)
            self.restore_model_eval_op()
            if g.statistics and not self.real:
                ray.init(num_cpus=g.num_actors, logging_level=logging.ERROR)
                module = __import__("algorithms.%s" % g.algo, fromlist=[g.algo])
                algo_class = ray.remote(getattr(module, g.algo))
                self.actors = [algo_class.remote(id + 1, False, self.dir_name) for id in range(g.num_actors)]
                self.sync_workers()
                results = ray.get([actor.test.remote() for actor in self.actors])
                results = np.array(results)
                print(results.shape)
                print(['test_multi_seen_ret, test_multi_novel_ret', 'test_single_seen_ret', 'test_single_novel_ret'])
                print(np.mean(results, axis=0))
                print(np.std(results, axis=0))
                with open('results.json', 'w') as f:
                    json.dump({'mean': np.mean(results, axis=0).tolist(), 'std': np.std(results, axis=0).tolist()}, f)
            else:
                self.demo()

    def save_best_model_op(self, accuracy=None):
        self.saver.save(self.sess, os.path.join('results', self.dir_name,
                                                'model-best.ckpt' if accuracy is None else 'model-%s.ckpt' % accuracy))
        print('best model saved!')

    def custom_init_op(self, host=None, port=None):
        raise NotImplementedError

    def demo(self):
        raise NotImplementedError

    def master(self):
        raise NotImplementedError

    def sync_workers(self):
        self.save_tmp_model_op()
        ray.get([actor.restore_tmp_model_op.remote() for actor in self.actors])
        if g.curriculum_learning:
            self.friction = (1 - self.max_train_success_perc) * g.robot_lateral_friction_seen
        else:
            self.friction = g.robot_lateral_friction_seen
        ray.get([actor.sync_max_success.remote(self.max_train_success_perc) for actor in self.actors])

    def sync_max_success(self, success):
        self.max_train_success_perc = success
        if g.curriculum_learning:
            self.env.commander.friction_mult = (1 - self.max_train_success_perc)
        else:
            self.env.commander.friction_mult = 1

    def save_tmp_model_op(self):
        self.saver.save(self.sess, os.path.join('results', self.dir_name, 'model-tmp.ckpt'))
        print('tmp model saved')

    def restore_tmp_model_op(self):
        self.saver.restore(self.sess, os.path.join('results', self.dir_name, 'model-tmp.ckpt'))
        print('tmp model restored')

    def summary(self):
        if not os.path.exists(os.path.join('results', self.dir_name)):
            raise NotImplementedError

        for key in self.summary_phs:
            self.fd[self.summary_phs[key]] = getattr(self, key.split('/')[-1])
        summary = self.sess.run(self.summary_op, feed_dict=self.fd)
        self.t += 1
        self.file_writer.add_summary(summary=summary, global_step=self.t)

    def save_model_op(self):
        self.saver.save(self.sess, os.path.join('results', self.dir_name, 'model.ckpt'))

    def restore_best_model_op(self, accuracy=None):
        self.saver.restore(self.sess, os.path.join('results', self.dir_name,
                                                   'model-best.ckpt' if accuracy is None else 'model-%s.ckpt' %
                                                                                              accuracy))
        print('best model restored')

    def restore_model_op(self):
        self.saver.restore(self.sess, os.path.join('results', self.dir_name, 'model.ckpt'))
        print('model restored')

    def restore_model_eval_op(self):
        try:
            print('%s restoring' % self.id)
            restore_path = g.path
            if restore_path is None:
                return
            saver = self.grasp_saver if 'position' in restore_path else self.saver
            saver.restore(self.sess, os.path.join(restore_path, 'model-best.ckpt'))
            print('%s model restored %s' % (self.id, 'eval' if self.eval else 'train'), restore_path)
        except:
            traceback.print_exc()

    def restart_ray(self):
        ray.shutdown()
        time.sleep(1)
        module = __import__("algorithms.%s" % g.algo, fromlist=[g.algo])
        ray.init(num_cpus=g.num_actors, logging_level=logging.ERROR, ignore_reinit_error=True)
        algo_class = ray.remote(getattr(module, g.algo))
        self.actors = [algo_class.remote(id + 1, False, self.dir_name) for id in range(g.num_actors)]

    def refresh_simulator_raw(self):
        initialized = False
        if self.is_graspit():
            while not initialized:
                for server in g.servers:
                    command = 'ssh -t bohan@%s \"pkill -f roslaunch\"' % server
                    try:
                        print(subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT))
                    except:
                        traceback.print_exc()
                        pass
                time.sleep(15)
                ret = ray.get([actor.restart_simulator.remote() for actor in self.actors])
                if sum(ret) == len(self.actors):
                    initialized = True
        elif 'localhost' not in g.servers:
            for server in g.servers:
                command = 'ssh -t bohan@%s \"pkill -f bullet3\"' % server
                print(subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT))

    def refresh_simulator(self):
        if self.t % 10 == 0:
            self.refresh_simulator_raw()

    def is_graspit(self):
        return g.simulator == 'graspit'

    def restart_simulator(self):
        try:
            self.env.commander.clearWorld(error=True)
            return True
        except:
            traceback.print_exc()
            return False

    def initialize_iterators(self, bs):
        raise NotImplementedError

    def do_train(self, train_ops, loss_name, buffer, epochs, bs):
        try:
            buffer_list = list(buffer.queue)
            for i in range(epochs):
                if bs is not None:
                    self.initialize_iterators(bs * len(buffer_list))
                for j in range(len(buffer_list)):
                    print(i, j, loss_name, self.sess.run(train_ops, feed_dict=buffer_list[j]))
        except:
            traceback.print_exc()
            raise NotImplementedError

    def build_q_train_op(self, q, next_obv_v_target, r_ph, scope, q_lr_ph, done_ph, qtarg_ph):
        ys = tf.stop_gradient(
            qtarg_ph if g.monte_carlo_q_target else (r_ph + (1 - done_ph) * g.gamma * next_obv_v_target))

        mse = (ys - q) ** 2
        positive = tf.cast(ys > 0.5, tf.float32)
        num_positive = tf.reduce_sum(positive) + d.e
        negative = 1 - positive
        num_negative = tf.reduce_sum(negative) + d.e
        self.positive_mse_tr = tf.reduce_sum(positive * mse) / num_positive
        self.negative_mse_tr = tf.reduce_sum(negative * mse) / num_negative
        if g.weighted_mse:
            loss = 0.5 * (self.positive_mse_tr + self.negative_mse_tr)
        else:
            loss = 0.5 * tf.reduce_mean(mse)
        return self.build_train_op(loss, q_lr_ph, scope)

    def build_v_train_op(self, v_tr, v_lr_ph, q1, q2, curr_pi, scope, vtarg_ph):
        q = tf.minimum(q1, q2) if g.off_policy else vtarg_ph
        loss = 0.5 * tf.reduce_mean((v_tr - tf.stop_gradient(q - g.alpha * curr_pi['log_prob'])) ** 2)
        return self.build_train_op(loss, v_lr_ph, scope)

    def build_pi_train_op(self, scope, curr_pi, old_pi, lr_ph, v, q, lr_mult_ph, adv_ph):
        log_ratio_tr = curr_pi['log_prob'] - old_pi['log_prob']
        ratio_tr = tf.exp(log_ratio_tr)
        clip_param_tr = g.clip_param * lr_mult_ph
        self.raw_adv_tr = q if g.off_policy else adv_ph
        self.adv_mean_tr, self.adv_var_tr = tf.nn.moments(self.raw_adv_tr, axes=[0])
        self.adv_std_tr = self.adv_var_tr ** 0.5
        self.normalized_adv_tr = (self.raw_adv_tr - self.adv_mean_tr) / (self.adv_std_tr + d.e)
        adv = tf.stop_gradient(self.normalized_adv_tr - g.alpha * curr_pi['log_prob'])
        surr1 = ratio_tr * adv
        surr2 = tf.clip_by_value(ratio_tr, 1.0 - clip_param_tr, 1.0 + clip_param_tr) * adv
        loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        return self.build_train_op(loss, lr_ph, scope)

    def build_train_op(self, loss_tr, lr_ph, scope):
        with tf.variable_scope(scope):
            opt = tf.train.AdamOptimizer(learning_rate=lr_ph, epsilon=1e-5)
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            print('TRAINABLE_VARIABLES', scope, vars)
            self.trainable_vars = vars
            train_op = slim.learning.create_train_op(loss_tr, opt, variables_to_train=vars,
                                                     clip_gradient_norm=g.gradclip, check_numerics=True,
                                                     summarize_gradients=True)
        return train_op, loss_tr

    def build_update_old_pi_op(self, curr_scope, old_scope):
        curr_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s' % curr_scope)
        old_pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s' % old_scope)
        assert len(curr_pi_vars) == len(old_pi_vars), (curr_pi_vars, old_pi_vars)
        for i in range(len(curr_pi_vars)):
            old_var_name = old_pi_vars[i].name
            curr_var_name = curr_pi_vars[i].name
            print('update', curr_var_name, old_var_name)
            assert old_var_name[old_var_name.find('/'):] == curr_var_name[curr_var_name.find('/'):], "%s, %s" % (
                old_pi_vars[i].name, curr_pi_vars[i].name)
        assign_ops = [tf.assign(old_pi_vars[i], curr_pi_vars[i]) for i in range(len(curr_pi_vars))]
        return tf.group(*assign_ops)

    def build_update_v_target_op(self, v_scope, v_target_scope):
        v_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s' % v_scope)
        v_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='%s' % v_target_scope)
        try:
            assert len(v_vars) == len(v_target_vars), (v_vars, v_target_vars)
            for i in range(len(v_vars)):
                target_v_var_name = v_target_vars[i].name
                v_var_name = v_vars[i].name
                print('update', v_var_name, target_v_var_name)
                assert target_v_var_name[target_v_var_name.find('/'):] == v_var_name[
                                                                          v_var_name.find('/'):], "%s, %s" % (
                    v_target_vars[i].name, v_vars[i].name)
            assign_ops = [tf.assign(ref=v_target_vars[i], value=(1 - g.tau) * v_target_vars[i] + g.tau * v_vars[i]) for
                          i in range(len(v_vars))]
            return tf.group(*assign_ops)
        except:
            traceback.print_exc()
            import ipdb
            ipdb.set_trace()

    def build_feature_pyramid(self):
        feature_pyramids = {}
        with tf.variable_scope('build_feature_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(0.0001)):
                feature_pyramids['P5'] = slim.conv2d(self.feature_maps['block_layer4'], num_outputs=256,
                                                     kernel_size=[1, 1], stride=1, scope='build_P5')

                feature_pyramids['P6'] = slim.max_pool2d(feature_pyramids['P5'], kernel_size=[2, 2], stride=2,
                                                         scope='build_P6')
                # P6 is down sample of P5

                for layer in range(4, 1, -1):
                    p, c = feature_pyramids['P' + str(layer + 1)], self.feature_maps['block_layer' + str(layer - 1)]
                    up_sample_shape = tf.shape(c)
                    up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                                 name='build_P%d/up_sample_nearest_neighbor' % layer)

                    c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                    scope='build_P%d/reduce_dimension' % layer)
                    p = up_sample + c
                    p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1, padding='SAME',
                                    scope='build_P%d/avoid_aliasing' % layer)
                    feature_pyramids['P' + str(layer)] = p

        final_feature_vector = []

        for key in feature_pyramids:
            feature = feature_pyramids[key]
            final_feature_vector.append(slim.flatten(
                slim.conv2d(inputs=feature, num_outputs=64, kernel_size=[3, 3], stride=1, scope='conv_%s' % key,
                            reuse=False)))

        return tf.concat(final_feature_vector, axis=1)

    def calculate_success_rate_per_obj(self):
        successes_per_obj = {}
        trials_per_obj = {}
        for info in self.infos:
            if 'closest_obj_name' in info:
                closest_obj_id = info['closest_obj_name']
                if closest_obj_id not in successes_per_obj:
                    successes_per_obj[closest_obj_id] = 0
                successes_per_obj[closest_obj_id] += float(info['success'])
                if closest_obj_id not in trials_per_obj:
                    trials_per_obj[closest_obj_id] = 0
                trials_per_obj[closest_obj_id] += 1
        success_rate_per_obj = {closest_obj_name: successes_per_obj[closest_obj_name] / trials_per_obj[closest_obj_name]
                                for closest_obj_name in successes_per_obj}
        for obj in success_rate_per_obj:
            setattr(self, obj.split('/')[-1], success_rate_per_obj[obj])
        self.success_rate_per_obj = str(success_rate_per_obj)
        self.success_rate_per_obj_values = dict(success_rate_per_obj).values()
