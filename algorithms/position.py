from __future__ import print_function

import gc
import glob
import itertools
import logging
import os
import time

import psutil
import ray
import tensorflow_probability as tfp
from pyquaternion import Quaternion

from algorithms.algorithm import algorithm
from algorithms.vision.vision import segmentation
from scripts.transformations import quaternion_from_matrix
from test_env.grasp_env import grasp_env
from utils import *


class position(algorithm):
    def custom_init_op(self, host=None, port=None):
        self.build_grasp_placeholders_op()

        self.grasp_curr_obv_tr, self.grasp_bl_obv_tr = self.grasp_build_perception_net(g.grasp_cnn, self.grasp_img)

        self.grasp_curr_pi = self.grasp_build_pi_network_op(scope='%s_curr_grasp_pi' % g.grasp_cnn,
                                                            input=self.grasp_curr_obv_tr, act_ph=self.grasp_act_ph,
                                                            trainable=self.is_grasp_master())
        if self.is_grasp_master() and not self.eval:
            self.grasp_old_obv_ph = tf.placeholder(tf.float32, self.grasp_curr_obv_tr.shape)
            self.old_grasp_pi = self.grasp_build_pi_network_op(scope='old_pi', input=self.grasp_old_obv_ph,
                                                               act_ph=self.grasp_act_ph, trainable=False)

            self.update_old_pi_op = self.build_update_old_pi_op(curr_scope='%s_curr_grasp_pi/pi' % g.grasp_cnn,
                                                                old_scope='old_pi/pi')

            self.pi_train_op, self.pi_loss_tr = self.build_pi_train_op(scope=g.grasp_cnn, curr_pi=self.grasp_curr_pi,
                                                                       old_pi=self.old_grasp_pi,
                                                                       adv_ph=self.grasp_adv_ph, lr_ph=self.grasp_lr_ph,
                                                                       lr_mult_ph=self.grasp_lr_mult_ph, q=None, v=None)
            self.build_image_ops()
            self.build_text_ops()
            self.build_grasp_histogram_ops()

            self.tensorboard_keys += ['experience/num_train_traj', 'experience/num_ts',
                                      'experience/effective_timesteps_so_far', 'experience/real_timesteps_so_far',
                                      'seconds', 'variance/improved_grasp_variance', 'loss/pi_loss_tr',
                                      'quality/max_train_success_perc', 'quality/train_success_perc',
                                      'quality/train_collision_perc', 'quality/max_test_multi_novel_success_perc',
                                      'quality/max_test_multi_seen_success_perc',
                                      'quality/max_test_single_novel_success_perc',
                                      'quality/max_test_single_seen_success_perc',
                                      'quality/test_multi_seen_success_perc', 'quality/test_multi_novel_success_perc',
                                      'quality/test_single_seen_success_perc', 'quality/test_single_novel_success_perc',
                                      'rewards/best_avg_train_r', 'rewards/avg_train_r', 'lr/grasp_v_lr',
                                      'lr/grasp_pi_lr', 'lr/lr_mult', 'system/memory', 'curriculum/friction']
            for obj in g.seen_objs:
                self.tensorboard_keys.append('per_obj_accuracy/%s' % obj)
            if g.use_grasp_bl:
                self.tensorboard_keys += ['loss/v_loss_tr']
        else:
            self.env = grasp_env(host, port, eval=self.eval, real=self.real, id=self.id)

    def grasp_build_pi_network_op(self, scope, input, act_ph, trainable):
        pi = {}

        with tf.variable_scope(scope + '/pi'):
            pi['pixel_mean'] = input[:, :, 0]
            pixel_dist = tfp.distributions.Categorical(logits=pi['pixel_mean'])

            if self.is_grasp_master() and not self.eval:
                sampled_index = act_ph[:, 0]
            else:
                sampled_index = pixel_dist.sample()
                if self.real:
                    pi['pixel_best_action'] = tf.expand_dims(tf.cast(self.grasp_index_ph, tf.float32), axis=1)
                else:
                    pi['pixel_best_action'] = tf.expand_dims(tf.cast(tf.argmax(pi['pixel_mean'], axis=1), tf.float32),
                                                             axis=1)

            sample_index_one_hot_vector = tf.one_hot(tf.cast(sampled_index, tf.int32), depth=pi['pixel_mean'].shape[1],
                                                     axis=1, dtype=tf.float32)
            termination_mean = tf.reduce_sum(input[:, :, 1] * sample_index_one_hot_vector, axis=1)
            termination_dist = tfp.distributions.Bernoulli(probs=termination_mean)
            scale_mean = tf.reduce_sum(input[:, :, 2] * sample_index_one_hot_vector, axis=1)
            pi['scale_log_std_raw'] = tf.get_variable('scale_log_std', shape=[1], trainable=trainable,
                                                      initializer=normc_initializer(0.1))
            pi['scale_log_std'] = pi['scale_log_std_raw'] - g.log_std_offset
            pi['scale_std'] = tf.exp(pi['scale_log_std'])
            scale_dist = tfp.distributions.MultivariateNormalDiag(loc=scale_mean, scale_diag=pi['scale_std'])

            if self.is_grasp_master() and not self.eval:
                pi['pixel_log_prob'] = pixel_dist.log_prob(sampled_index) * self.grasp_done_ph
                pi['pixel_prob'] = tf.exp(pi['pixel_log_prob'])
                pi['terminate_action'] = act_ph[:, 1]
                pi['terminate_log_prob'] = termination_dist.log_prob(pi['terminate_action']) * self.grasp_no_force_ph

                pi['scale_action'] = act_ph[:, 2]
                pi['scale_log_prob'] = scale_dist.log_prob(pi['scale_action']) * (1 - self.grasp_done_ph)
            else:
                pi['pixel_sampled_action'] = tf.cast(tf.expand_dims(sampled_index, axis=1), tf.float32)
                pi['terminate_sampled_action'] = tf.cast(tf.expand_dims(termination_dist.sample(), axis=1), tf.float32)
                pi['terminate_best_action'] = tf.cast(tf.expand_dims(termination_mean > 0.5, axis=1), tf.float32)
                pi['scale_sampled_action'] = tf.cast(tf.expand_dims(scale_dist.sample(), axis=1), tf.float32)

            orientation_mean = tf.reduce_sum(
                input[:, :, 2 + g.crop_method:] * tf.expand_dims(sample_index_one_hot_vector, axis=2), axis=1)
            pi['ori_log_std_raw'] = tf.get_variable('ori_log_std', shape=[orientation_mean.get_shape().as_list()[1]],
                                                    trainable=trainable, initializer=normc_initializer(0.1))

            pi['ori_log_std'] = pi['ori_log_std_raw'] - ([2] + [g.log_std_offset] * 7)

            pi['ori_std'] = tf.exp(pi['ori_log_std'])
            self.log_std_init_op = tf.variables_initializer(var_list=[pi['ori_log_std_raw']])

            orientation_dist = tfp.distributions.MultivariateNormalDiag(loc=orientation_mean, scale_diag=pi['ori_std'])

            if self.is_grasp_master() and not self.eval:
                pi['orientation_log_prob'] = orientation_dist.log_prob(
                    act_ph[:, 2 + g.crop_method:]) * self.grasp_done_ph
                pi['log_prob'] = pi['pixel_log_prob'] + pi['terminate_log_prob'] + (
                    pi['scale_log_prob'] if g.crop_method else 0) + pi['orientation_log_prob']
            else:
                pi['ori_sampled_action'] = orientation_dist.sample()
                pi['ori_best_action'] = orientation_mean
                pi['sampled_action'] = tf.concat((
                    pi['pixel_sampled_action'], pi['terminate_sampled_action'], pi['scale_sampled_action'],
                    pi['ori_sampled_action']), axis=1)
                pi['best_action'] = tf.concat((pi['pixel_best_action'],
                                               pi['terminate_sampled_action'] if not g.best_terminate else pi[
                                                   'terminate_best_action'], pi['scale_sampled_action'],
                                               pi['ori_best_action']), axis=1)
            return pi

    def master(self):
        if not self.is_graspit() and 'localhost' not in g.servers:
            self.refresh_simulator()
        for obj in g.seen_objs:
            setattr(self, obj.split('/')[-1], 0)
        ray.init(num_cpus=g.num_actors, logging_level=logging.ERROR)
        module = __import__("algorithms.%s" % d.config.algo, fromlist=[d.config.algo])
        algo_class = ray.remote(getattr(module, d.config.algo))
        self.actors = [algo_class.remote(id + 1, False, self.dir_name) for id in range(g.num_actors)]

        self.avg_train_r = -float('inf')
        self.threshold_train_r = 0.1
        self.best_avg_train_r = -float('inf')
        self.max_train_success_perc = 0
        self.max_outperformance = 0
        self.max_test_multi_seen_success_perc = 0
        self.max_test_multi_novel_success_perc = 0
        self.max_test_single_seen_success_perc = 0
        self.max_test_single_novel_success_perc = 0
        just_restored = True
        load_depth_dirs = [os.path.join(os.path.expanduser('~/Desktop/images'),
                                        server) if server != 'localhost' else d.server_graspit_img_dir for server in
                           g.servers]
        with tf.Graph().as_default():
            self.sess.run(self.init_op)
            self.restore_model_eval_op()
            self.save_best_model_op()
            start = time.time()

            while True:
                [[os.remove(f) for f in glob.glob(os.path.join(load_depth_dir, '*'))] for load_depth_dir in
                 load_depth_dirs]
                [os.remove(f) for f in glob.glob(os.path.join(d.dirname, '*'))]
                self.sess.run(self.update_old_pi_op)
                try:
                    self.sync_workers()
                    if self.is_graspit():
                        self.refresh_simulator()

                    ret = list(zip(*ray.get([actor.train_and_test.remote() for actor in self.actors])))
                    experience = list(itertools.chain(*ret[0]))
                    self.num_train_traj = sum(ret[1])
                    traj_r = sum(ret[2])
                    self.final_trajs = ret[3]
                    self.test_multi_seen_success_perc = np.mean(ret[4])
                    self.test_multi_novel_success_perc = np.mean(ret[5])
                    self.test_single_seen_success_perc = np.mean(ret[6])
                    self.test_single_novel_success_perc = np.mean(ret[7])
                    del ret
                except:
                    traceback.print_exc()
                    self.restore_best_model_op()
                    if self.is_graspit():
                        self.refresh_simulator_raw()
                    else:
                        raise ValueError(traceback.format_exc())
                    continue

                self.num_ts = len(experience)
                self.real_timesteps_so_far += self.num_ts

                new_avg_train_r = traj_r / (self.num_train_traj + d.e)
                print('self.avg_train_r', new_avg_train_r)
                if self.best_avg_train_r - g.safe_train_r > new_avg_train_r and not just_restored:
                    self.restore_best_model_op()
                    just_restored = True
                    continue

                self.avg_train_r = new_avg_train_r

                if self.avg_train_r > self.threshold_train_r:
                    self.save_best_model_op(accuracy=int(self.threshold_train_r * 100))
                    self.threshold_train_r += 0.1

                if self.test_multi_seen_success_perc > self.max_test_multi_seen_success_perc:
                    self.max_test_multi_seen_success_perc = self.test_multi_seen_success_perc

                if self.test_multi_novel_success_perc > self.max_test_multi_novel_success_perc:
                    self.max_test_multi_novel_success_perc = self.test_multi_novel_success_perc

                if self.test_single_seen_success_perc > self.max_test_single_seen_success_perc:
                    self.max_test_single_seen_success_perc = self.test_single_seen_success_perc

                if self.test_single_novel_success_perc > self.max_test_single_novel_success_perc:
                    self.max_test_single_novel_success_perc = self.test_single_novel_success_perc

                self.update_learning_rates_op()

                self.sess.run(self.update_old_pi_op)

                try:
                    self.prepare_learning(experience)
                    self.initialize_iterators(self.num_ts)
                    save = bool(self.avg_train_r > self.best_avg_train_r)
                    if save:
                        self.best_avg_train_r = self.avg_train_r
                    self.seconds = time.time() - start
                    start = time.time()
                    self.summary()
                    if save:
                        self.save_best_model_op()
                    self.learn()
                except:
                    traceback.print_exc()
                    self.restore_best_model_op()
                    if self.is_graspit():
                        self.refresh_simulator_raw()
                    if g.algo == 'position':
                        self.sess.run(self.log_std_init_op)
                    continue

                gc.collect()
                just_restored = False

    def initialize_iterators(self, bs):
        self.sess.run(self.grasp_iterator.initializer, feed_dict={self.grasp_img_path_ph: self.img_paths[:bs]})

    def get_grasp_action_raw(self, s, best, index):
        try:
            self.sess.run(self.grasp_iterator.initializer, feed_dict={self.grasp_img_path_ph: [s['image_path']]})
            action, grasp_obv = self.sess.run(
                [self.grasp_curr_pi['best_action' if best else 'sampled_action'], self.grasp_curr_obv_tr],
                feed_dict={self.grasp_scale_obv_ph: [s['scale']], self.grasp_index_ph: [index] if self.real else [0]})
            return action, grasp_obv[0]
        except:
            if self.real:
                traceback.print_exc()
                import ipdb
                ipdb.set_trace()
            else:
                raise NotImplementedError

    def get_grasp_action(self, s, best=False, index=0):
        return self.get_grasp_action_raw(s, best, index)

    def get_qua_from_dirvec_and_angle(self, dir_vec, sin=None, cos=None):
        if sin is not None and cos is not None:
            qua = dire_vec_to_qua(dir_vec)
            first_qua = Quaternion(qua[3], qua[0], qua[1], qua[2])
            radian = np.arctan(sin / (cos + d.e))
            rotate_qua = Quaternion(axis=dir_vec, radians=radian)

            final_qua = quaternion_from_matrix(
                np.matmul(rotate_qua.transformation_matrix, first_qua.transformation_matrix))

            if d.debug:
                rotated_dir_vec = first_qua.rotate([0, 0, 1])
                rotated_dir_vec = rotated_dir_vec / np.linalg.norm(rotated_dir_vec)
                assert np.allclose(rotated_dir_vec - dir_vec, 0, atol=1e-3), '%s %s' % (rotated_dir_vec, dir_vec)
                assert np.allclose(dir_vec - rotate_qua.rotate(dir_vec), 0)
                assert np.allclose(np.linalg.norm(final_qua), 1)
                final_qua_2 = Quaternion(final_qua[3], final_qua[0], final_qua[1], final_qua[2])
                final_vec = final_qua_2.rotate([0, 0, 1])
                final_vec /= np.linalg.norm(final_vec)
                assert np.allclose(dir_vec - final_vec, 0, atol=1e-5), '%s %s' % (dir_vec, final_vec)
            return final_qua
        else:
            return dire_vec_to_qua(dir_vec)

    def need_reinit_pybullet(self, s, novel):
        threshold = g.reinit_pybullet_depth_threshold_novel if novel else g.reinit_pybullet_depth_threshold_seen
        if d.camera_pitch_deg != -89 and not g.random_reset_body:
            threshold += 0.1
        max_depth = s['real_depth'].min()
        return (not self.real) and (d.camera_dist - threshold < max_depth or (not self.env.commander.graspable_bodies))

    def test_objects(self, novel, multi):
        if multi or not novel:
            return 0
        successes = []
        print('testing %s %s %s scenes...' % (
            g.num_eval_episodes_per_actor, 'novel' if novel else 'seen', 'multi' if multi else 'single'))
        self.env.init_world(test=True, novel=novel, multi=multi)
        for episode in range(g.num_eval_episodes_per_actor):
            while True:
                try:
                    if episode % 5 == 0 and g.statistics:
                        print('Progress: %s / %s' % (episode, self.id))
                    self.env.init_world(test=True, novel=novel, multi=multi)
                    s = self.env.reset()
                    while self.need_reinit_pybullet(s, novel):
                        self.env.init_world(test=True, novel=novel, multi=multi)
                        s = self.env.reset()
                    done = False
                    t = 1
                    while not done:
                        s['image_path'] = os.path.join(d.dirname, 'grasp_test_%s_id_%s.png' % (t, self.id))
                        cv2.imwrite(s['image_path'], s['image'])
                        while True:
                            try:
                                action, grasp_obv = self.get_grasp_action(s, best=True)
                                s, r, done, info = self.env.step(self.get_grasp_action_dict(action[0]),
                                                                 info={'s': s, 't': t})
                                break
                            except:
                                pass

                        t += 1
                    successes.append(info['success'])
                    break
                except:
                    pass
        return float(np.mean(successes)) if len(successes) > 0 else 1.0

    def test(self):
        test_single_novel_ret = self.test_objects(novel=True, multi=False)
        test_single_seen_ret = self.test_objects(novel=False, multi=False)
        test_multi_seen_ret = self.test_objects(novel=False, multi=True)
        test_multi_novel_ret = self.test_objects(novel=True, multi=True)
        if g.algo == 'position' or (not self.eval):
            return tuple([test_single_seen_ret, test_single_novel_ret, test_multi_seen_ret, test_multi_novel_ret])
        else:
            return np.array([test_single_seen_ret, test_single_novel_ret, test_multi_seen_ret, test_multi_novel_ret])

    def train_and_test(self):
        train_ret = self.train()
        test_ret = self.test()
        return train_ret + test_ret

    def train(self):
        experience, traj, traj_r = [], [], []
        cum_r, num_traj, num_succ, curr_t_steps = [0] * 4

        self.env.obj_index = -1
        self.env.init_world(test=False, multi=True, novel=False)
        s = self.env.reset()

        t = 1
        while curr_t_steps < g.bs // g.num_actors:
            if curr_t_steps % 5 == 0:
                print('Progress: %s / %s' % (curr_t_steps, g.bs // g.num_actors))

            s['image_path'] = os.path.join(d.dirname, 'grasp_ts_%s_t_%s_id_%s.png' % (curr_t_steps, t, self.id))
            cv2.imwrite(s['image_path'], s['image'])

            while True:
                try:
                    action, obv = self.get_grasp_action(s)
                    next_s, r, done, info = self.env.step(self.get_grasp_action_dict(action[0]), info={'s': s, 't': t})
                    break
                except:
                    pass

            s['obv'] = obv
            traj.append([s, action[0], r, next_s, done, info])
            cum_r += r

            if done:
                traj = [self.save_memory(exp) for exp in traj]
                final_traj = traj
                experience += traj
                traj = []
                self.env.init_world(test=False, multi=True, novel=False)
                s = self.env.reset()
                while self.need_reinit_pybullet(s, novel=False):
                    self.env.init_world(test=False, multi=True, novel=False)
                    s = self.env.reset()
                num_traj += 1
                traj_r.append(cum_r)
                cum_r = 0
                t = 1
            else:
                t += 1
                s = next_s

            curr_t_steps += 1
        return experience, num_traj, sum(traj_r), final_traj

    def delete_keys(self, dictionary, keys):
        if dictionary is None:
            return dictionary
        for key in keys:
            if key in dictionary:
                del dictionary[key]
        return dictionary

    def get_grasp_action_dict(self, a):
        if g.orientation == 4 or g.top_down:
            dictionary = {'roll': a[2 + g.crop_method], 'pitch': .5, 'yaw': .25}
        elif g.orientation == 8:
            # print(a[3 + g.crop_method], 0.5, np.pi - 0.5)
            if g.hand == 'barrett':
                dictionary = {'roll': a[2 + g.crop_method], 'pitch': np.clip(a[3 + g.crop_method], 0.01, .005),
                              'yaw': a[4 + g.crop_method]}
            else:
                dictionary = {'roll': a[2 + g.crop_method], 'pitch': a[3 + g.crop_method], 'yaw': a[4 + g.crop_method]}
        else:
            raise NotImplementedError

        action_index = -1
        if g.hand == 'seed':
            dictionary['finger_4'] = np.clip(a[action_index], a_min=0, a_max=1)
            action_index -= 1
        dictionary['finger_3'] = np.clip(a[action_index], a_min=0, a_max=1)
        action_index -= 1
        dictionary['finger_2'] = np.clip(a[action_index], a_min=0, a_max=1)
        action_index -= 1
        dictionary['finger_1'] = np.clip(a[action_index], a_min=0, a_max=1)
        action_index -= 1

        dictionary['spread'] = np.clip(a[action_index], a_min=0.05, a_max=0.1)
        action_index -= 1

        dictionary['pixel'] = a[0]
        dictionary['terminate'] = a[1] if g.learn_attention else True
        dictionary['scale'] = max(0, min(1, a[2])) if g.crop_method else None
        return dictionary

    def save_memory(self, traj):
        traj[0] = self.delete_keys(traj[0], ('rgb', 'real_depth', 'image', 'point_cloud', 'cropped_depth'))
        return traj

    def compute_data(self, experience):
        obvs, acts, rews, _, dones, self.infos = zip(*experience)
        self.num_iter = int(self.num_ts // g.grasp_mini_bs)
        self.img_paths = np.array([obv['image_path'] for obv in obvs])
        self.obvs = np.array([obv['obv'] for obv in obvs])
        self.scale_obvs = np.array([obv['scale'] for obv in obvs])
        acts = np.array(acts)
        rews = np.array(rews)
        dones = list(dones)

        if g.use_grasp_bl:
            self.initialize_iterators(self.num_ts)
            baseline = []
            for i in range(int(math.ceil(float(self.num_ts) / g.grasp_mini_bs))):
                bl = self.sess.run(self.grasp_bl_tr)
                if type(bl) is np.ndarray and bl.shape[0] > 0:
                    baseline.append(bl)
            baseline = np.concatenate(baseline)
        else:
            baseline = np.zeros(self.num_ts)
        assert baseline.shape[0] == len(dones) == rews.shape[0], (len(dones), rews.shape, baseline.shape)
        seg = generate_seg(dones, rews, baseline)
        add_vtarg_and_adv(seg)
        advs, vtarg = seg["adv"], seg["cum_r"]
        advs = (advs - advs.mean()) / (advs.std() + d.e)
        self.improved_grasp_variance = explained_variance(np.squeeze(baseline), np.squeeze(vtarg))

        # shuffle
        self.order = np.random.permutation(self.num_ts).tolist()
        self.off_acts = acts[self.order]
        self.advs = advs[self.order]
        self.obvs = self.obvs[self.order]
        self.scale_obvs = self.scale_obvs[self.order]
        self.img_paths = self.img_paths[self.order]
        self.vtarg = vtarg[self.order]
        self.dones = np.array(dones)[self.order]
        self.infos = [self.infos[i] for i in self.order]

        self.no_forces = np.array([info['no_force'] for info in self.infos])
        self.dir_vec = [info['dir_vec'] for info in self.infos if 'dir_vec' in info]
        self.train_collision_perc = np.mean([info['collision'] for info in self.infos if 'collision' in info])
        self.train_success_perc = np.mean([info['success'] for info in self.infos if 'success' in info])
        self.max_train_success_perc = max(self.max_train_success_perc, self.train_success_perc)

        self.calculate_success_rate_per_obj()

    def prepare_learning(self, experience):
        self.compute_data(experience)

        self.pi_replay_buffer = Queue(maxsize=g.replay_buffer_queue_size)
        self.pi_replay_buffer = self.prepare_pi_replay_buffer(bs=g.grasp_mini_bs, buffer=self.pi_replay_buffer)
        self.pi_replay_buffer_size = len(list(self.pi_replay_buffer.queue))

        self.num_ts = g.grasp_mini_bs / 2

        process = psutil.Process(os.getpid())
        self.memory = process.memory_percent()

        self.fd = {self.grasp_scale_obv_ph: self.scale_obvs[:self.num_ts],
                   self.grasp_act_ph: self.off_acts[:self.num_ts], self.grasp_adv_ph: self.advs[:self.num_ts],
                   self.grasp_old_obv_ph: self.obvs[:self.num_ts], self.grasp_lr_ph: self.grasp_pi_lr,
                   self.grasp_done_ph: self.dones[:self.num_ts],
                   self.grasp_v_target_ph: self.vtarg[:self.num_ts, np.newaxis], self.grasp_v_lr_ph: self.grasp_v_lr,
                   self.grasp_lr_mult_ph: self.lr_mult, self.seen_objs_ph: g.seen_objs,
                   self.novel_objs_ph: g.novel_objs, self.grasp_dir_vec_ph: self.dir_vec[:self.num_ts],
                   self.success_rate_per_obj_ph: self.success_rate_per_obj,
                   self.success_rate_per_obj_values_ph: self.success_rate_per_obj_values,
                   self.restore_path_ph: str(g.path), self.grasp_no_force_ph: self.no_forces[:self.num_ts]}

        traj = self.final_trajs[0]
        self.fd[self.traj_depth_ph] = []
        self.fd[self.traj_position_ph] = []
        self.fd[self.traj_termination_ph] = []
        for i in range(len(traj)):
            img = np.repeat(cv2.imread(traj[i][0]['image_path'], cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis], repeats=3,
                            axis=2)
            a = self.get_grasp_action_dict(traj[i][1])
            cx, cy = get_local_cx_cy(a['pixel'], w=d.img_s, h=d.img_s)
            dx, dy = get_global_local_dx_dy(local_cx=cx, local_cy=cy, local_h=d.img_s, local_w=d.img_s,
                                            scale=a['scale'])

            cv2.circle(img, (int(round(cx)), int(round(cy))), 3, (255, 0, 0), -1)
            cv2.putText(img, '%s=(%s, %s) r=%s z=%s' % (
                a['pixel'], int(round(cy)), int(round(cx)), traj[i][2], a['terminate']), (d.map_s / 8, d.map_s / 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.rectangle(img, (max(0, int(round(cx - dx))), max(int(round(cy - dy)), 0)),
                          (int(round(cx + dx)), int(round(cy + dy))), (255, 0, 0), 3)
            heatmap = normalize_depth(np.resize(traj[i][0]['obv'][:, 0], (d.map_s, d.map_s, 1)), fixed=False)
            termination_map = normalize_depth(np.resize(traj[i][0]['obv'][:, 1], (d.map_s, d.map_s, 1)), fixed=False)
            self.fd[self.traj_depth_ph].append(img)
            self.fd[self.traj_position_ph].append(heatmap)
            self.fd[self.traj_termination_ph].append(termination_map)

    def prepare_pi_replay_buffer(self, bs, buffer):
        for i in range(self.num_iter):
            real_ord = range(i * g.grasp_mini_bs, min((i + 1) * g.grasp_mini_bs, self.num_ts))

            batch_fd = {self.grasp_scale_obv_ph: self.scale_obvs[real_ord], self.grasp_act_ph: self.off_acts[real_ord],
                        self.grasp_adv_ph: self.advs[real_ord],
                        self.grasp_v_target_ph: self.vtarg[real_ord, np.newaxis],
                        self.grasp_old_obv_ph: self.obvs[real_ord], self.grasp_done_ph: self.dones[real_ord],
                        self.grasp_lr_ph: self.grasp_pi_lr, self.grasp_lr_mult_ph: self.lr_mult,
                        self.grasp_v_lr_ph: self.grasp_v_lr, self.grasp_no_force_ph: self.no_forces[real_ord]}
            if buffer.full():
                buffer.get()
            buffer.put(batch_fd)
        return buffer

    def learn(self):
        train_ops = [self.pi_loss_tr, self.pi_train_op, self.grasp_curr_pi['ori_log_std']]
        if g.use_grasp_bl:
            train_ops += [self.v_loss_tr, self.train_v_op]

        self.do_train(train_ops, 'pi', self.pi_replay_buffer, g.optim_epochs, g.grasp_mini_bs)

        self.pi_replay_buffer = None

    def update_learning_rates_op(self):
        self.effective_timesteps_so_far += self.num_ts
        self.lr_mult = max(1.0 - float(self.effective_timesteps_so_far) / g.total_ts, 0.01)
        self.grasp_v_lr = g.v_lr * self.lr_mult
        self.grasp_pi_lr = g.grasp_lr * self.lr_mult

    def calc_kl_divergence(self, d1, d2):
        return tf.reduce_sum(
            d2['log_std'] - d1['log_std'] + (tf.square(d1['std']) + tf.square(d1['pixel_mean'] - d2['pixel_mean'])) / (
                    2.0 * tf.square(d2['std'])) - 0.5, axis=-1)

    def grasp_build_perception_net(self, scope, img):
        if scope == 'segmentation':
            with tf.variable_scope(scope):
                logits = segmentation(input=img, scale_obv_2d=self.grasp_scale_obv_2d)
                return logits
        else:
            raise NotImplementedError

    def build_grasp_placeholders_op(self):
        with tf.variable_scope('inputs'):
            self.is_grasp_training_ph = tf.placeholder(tf.bool, name='is_grasp_training_ph')
            self.grasp_img_path_ph = tf.placeholder(tf.string, shape=[None], name='grasp_img_path_ph')
            self.grasp_iterator = tf.data.Dataset.from_tensor_slices(self.grasp_img_path_ph).map(
                img_parse_function).batch(g.grasp_mini_bs).make_initializable_iterator()
            self.grasp_img = self.grasp_iterator.get_next()
            self.grasp_img = 255.0 - self.grasp_img
            self.grasp_scale_obv_ph = tf.placeholder(tf.float32, shape=[None])
            if g.scale_obv:
                self.grasp_scale_obv_expanded = tf.expand_dims(
                    tf.expand_dims(tf.expand_dims(self.grasp_scale_obv_ph, axis=1), axis=2), axis=3)
                self.grasp_scale_obv_2d = tf.tile(self.grasp_scale_obv_expanded, multiples=(1, d.img_s, d.img_s, 1))

            self.grasp_index_ph = tf.placeholder(tf.int64, shape=[None], name='grasp_index_ph')
            self.grasp_done_ph = tf.placeholder(tf.float32, shape=[None], name='grasp_done_ph')
            self.grasp_no_force_ph = tf.placeholder(tf.float32, shape=[None], name='grasp_no_force_ph')
            self.success_rate_per_obj_values_ph = tf.placeholder(tf.float32, shape=[None],
                                                                 name='success_rate_per_obj_values_ph')
            self.grasp_dir_vec_ph = tf.placeholder(tf.float32, shape=[None, 3], name='grasp_dir_vec_ph')
            ori_act_dim = {8: 3, 4: 1}[g.orientation]
            self.grasp_act_ph = tf.placeholder(tf.float32, [None, 3 + g.crop_method + ori_act_dim + (
                3 if g.hand == 'barrett' else 4)], 'grasp_act_ph')

            if self.is_grasp_master():
                self.grasp_adv_ph = tf.placeholder(tf.float32, [None], name='grasp_adv_ph')
                self.grasp_lr_ph = tf.placeholder(tf.float32, name='grasp_lr_ph')

                self.grasp_lr_mult_ph = tf.placeholder(tf.float32, name='grasp_lr_mult_ph')
                self.grasp_v_target_ph = tf.placeholder(tf.float32, (None, 1), name='grasp_v_target_ph')
                self.grasp_v_lr_ph = tf.placeholder(tf.float32, name='grasp_v_lr_ph')

    def build_image_ops(self):
        self.traj_depth_ph = tf.placeholder(tf.uint8, (None, d.img_s, d.img_s, 3), 'traj_depth_ph')
        self.traj_position_ph = tf.placeholder(tf.uint8, (None, d.map_s, d.map_s, 1), 'traj_position_ph')
        self.traj_termination_ph = tf.placeholder(tf.uint8, (None, d.map_s, d.map_s, 1), 'traj_termination_ph')
        tf.summary.image('traj_depth_ph', self.traj_depth_ph, max_outputs=g.summary_horizon)
        tf.summary.image('traj_position_ph', self.traj_position_ph, max_outputs=g.summary_horizon)
        tf.summary.image('traj_termination_ph', self.traj_termination_ph, max_outputs=g.summary_horizon)

    def build_grasp_histogram_ops(self):
        tf.summary.histogram('dir_vec_x', self.grasp_dir_vec_ph[:, 0])
        tf.summary.histogram('dir_vec_y', self.grasp_dir_vec_ph[:, 1])
        tf.summary.histogram('dir_vec_z', self.grasp_dir_vec_ph[:, 2])
        tf.summary.histogram('pixel_prob', self.grasp_curr_pi['pixel_prob'])
        tf.summary.histogram('terminate_action', self.grasp_curr_pi['terminate_action'])
        tf.summary.histogram('grasp_done_ph', self.grasp_done_ph)
        tf.summary.histogram('success_rate_per_obj', self.success_rate_per_obj_values_ph)
        if g.orientation >= 5:
            tf.summary.histogram('ori_std', self.grasp_curr_pi['ori_std'])
            tf.summary.histogram('ori_log_std', self.grasp_curr_pi['ori_log_std'])

    def build_text_ops(self):
        self.restore_path_ph = tf.placeholder(tf.string, name='restore_path_ph')
        self.seen_objs_ph = tf.placeholder(tf.string, name='seen_objs_ph')
        self.novel_objs_ph = tf.placeholder(tf.string, name='novel_objs_ph')
        self.success_rate_per_obj_ph = tf.placeholder(tf.string, name='success_rate_per_obj_ph')
        tf.summary.text('path', self.restore_path_ph)
        tf.summary.text('seen_objs', self.seen_objs_ph)
        tf.summary.text('novel_objs', self.novel_objs_ph)
        tf.summary.text('success_rate_per_obj', self.success_rate_per_obj_ph)

    def demo(self):
        print('testing novel graspable_bodies...')
        if self.real:
            ray.init(num_cpus=g.num_actors, logging_level=logging.ERROR)
            module = __import__("algorithms.%s" % d.config.algo, fromlist=[d.config.algo])
            algo_class = ray.remote(getattr(module, d.config.algo))
            print('num_actors', g.num_actors)
            self.actors = [algo_class.remote(id + 1, True, self.dir_name, real=True) for id in range(g.num_actors)]
            ray.get([actor.restore_model_eval_op.remote() for actor in self.actors])
            print('Models restored!')
        while True:
            try:
                [os.remove(os.path.join(d.server_graspit_img_dir, i)) for i in os.listdir(d.server_graspit_img_dir)]
                if self.real:
                    self.generate_grasp_multithreaded(test=True, multi=(not g.eval_single_object), novel=False)
                else:
                    self.generate_grasp(test=True, multi=(not g.eval_single_object), novel=False)
            except:
                traceback.print_exc()
                pass

    def generate_grasp_multithreaded(self, test, multi, novel):
        self.env.init_world(test=test, multi=multi, novel=novel)
        s = self.env.reset()
        while self.need_reinit_pybullet(s, novel):
            self.env.init_world(test=test, multi=multi, novel=novel)
            s = self.env.reset()
        plans = []
        trial = 0

        while not plans:
            slave_post_point_clouds = (trial % g.repost_point_cloud_interval == 0)
            if slave_post_point_clouds:
                self.env.commander.post_point_cloud()
            robot_state = self.env.commander.getRobotState()
            raw_plans = ray.get([actor.generate_grasp_plan.remote(s, slave_post_point_clouds, robot_state,
                                                                  trial % g.repost_point_cloud_interval) for actor in
                                 self.actors])
            plans = [plan for plan in raw_plans if plan is not None]
            trial += 1
            if g.hand == 'barrett':
                zs = [self.env.world_pos(plan['a'], plan['s'])[1][2] for plan in plans]
            else:
                zs = [self.env.world_pos_raw(plan['a'], plan['s'])[2] for plan in plans]
            print('z', zs)
            if not zs or (g.hand == 'barrett' and np.max(zs) < -0.3) or (g.hand == 'seed' and np.max(zs) < -0.288):
                plans = []
                continue
        good_plan = plans[np.argmax(zs)]
        while True:
            try:
                s, r, done, grasp_info = self.env.grasp_step(a=good_plan['a'], info=good_plan)
                break
            except:
                traceback.print_exc()
                pass
        assert done
        return s

    def generate_grasp(self, test, multi, novel, reinit_env=True, get_baseline=True, state_id=None):
        index = 0
        while True:
            if reinit_env:
                self.env.init_world(test=test, multi=multi, novel=novel)
                s = self.env.reset()
                while self.need_reinit_pybullet(s, novel):
                    self.env.init_world(test=test, multi=multi, novel=novel)
                    s = self.env.reset()
            else:
                self.env.commander.restoreState(state_id)
                s = self.env.get_state(1, get_rgb=self.real)
            t = 1
            done = False
            curr_t_steps = 0
            try:
                while not done:
                    s['image_path'] = os.path.join(d.dirname, 'grasp_ts_%s_t_%s_id_%s.png' % (curr_t_steps, t, self.id))
                    cv2.imwrite(s['image_path'], s['image'])
                    action, obv = self.get_grasp_action(s, best=True)
                    a = self.get_grasp_action_dict(action[0])

                    if index is not None:
                        a['index'] = index
                    s, r, done, grasp_info = self.env.grasp_step(a, info={'s': s, 't': t},
                                                                 get_baseline=get_baseline and (
                                                                         self.t % g.check_tactile_baseline == 0))
                    curr_t_steps += 1
                return s
            except:
                traceback.print_exc()
                index += 1

    def generate_grasp_plan(self, s, post_point_cloud, robot_state, z_increment):
        if post_point_cloud:
            self.env.commander.post_point_cloud()
        t = 1
        done = False
        curr_t_steps = 0
        index = 0
        while True:
            try:
                while not done:
                    s['image_path'] = os.path.join(d.dirname, 'grasp_ts_%s_t_%s_id_%s.png' % (curr_t_steps, t, self.id))
                    cv2.imwrite(s['image_path'], s['image'])
                    action, obv = self.get_grasp_action(s, best=False)
                    a = self.get_grasp_action_dict(action[0])
                    if index is not None:
                        a['index'] = index
                    a['add_z'] = z_increment * 0.001
                    info = {'s': s, 't': t, 'robot_state': robot_state}
                    s, r, done, grasp_info = self.env.grasp_plan(a, info=info)
                    curr_t_steps += 1
                info['a'] = a
                return info
            except:
                return None
