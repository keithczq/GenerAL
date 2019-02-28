#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2010, Antons Rebguns, Cody Jorgensen, Cara Slutter
#               2010-2011, Antons Rebguns
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of University of Arizona nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import copy


def map_to_sim_frame(joint_name, value):
    joint_angle = copy.copy(value)
    if joint_name == "forearm__base":
        joint_angle /= 2 * np.pi
        joint_angle *= np.pi
        joint_angle -= np.pi / 2
    elif joint_name == "palm_axis__forearm":
        joint_angle /= 2 * np.pi
        joint_angle *= 2 * 0.785398163397
        joint_angle = 2 * 0.785398163397 - joint_angle
        joint_angle -= 0.785398163397
    elif joint_name == "palm__palm_axis":
        joint_angle /= 2 * np.pi
        joint_angle *= 2 * 0.785398163397
        joint_angle -= 0.785398163397
    elif joint_name == "Rproximal__palm":
        joint_angle /= 2 * np.pi
        joint_angle *= 1.4835298642
        joint_angle -= 0
    elif joint_name == "Mproximal__palm":
        joint_angle /= 2 * np.pi
        joint_angle *= 1.4835298642
        joint_angle -= 0
    elif joint_name == "palm__thumb_base":
        joint_angle /= 2 * np.pi
        joint_angle *= np.pi / 2
        joint_angle -= 0
    elif joint_name == "Tproximal__thumb_base":
        joint_angle /= 2 * np.pi
        joint_angle *= np.pi / 2
        joint_angle -= 0
    elif joint_name == "Iproximal__palm":
        joint_angle /= 2 * np.pi
        joint_angle *= 1.4835298642
        joint_angle -= 0
    return joint_angle


__author__ = 'Antons Rebguns, Cody Jorgensen, Cara Slutter'
__copyright__ = 'Copyright (c) 2010-2011 Antons Rebguns, Cody Jorgensen, Cara Slutter'

__license__ = 'BSD'
__maintainer__ = 'Antons Rebguns'
__email__ = 'anton@email.arizona.edu'

import sys
from threading import Lock, Thread

import numpy as np
import rospy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from dynamixel_controllers.srv import RestartController, RestartControllerResponse, StartController, \
    StartControllerResponse, StopController, StopControllerResponse
from dynamixel_driver.dynamixel_serial_proxy import SerialProxy
from sensor_msgs.msg import JointState


class ControllerManager:
    def __init__(self):
        rospy.init_node('dynamixel_controller_manager', anonymous=True)
        rospy.on_shutdown(self.on_shutdown)
        self.current_joint_state = np.zeros(8)
        self.joint_names = ['wrist_rotation', 'wrist_adduction', 'wrist_flexion', 'thumb_adduction', 'thumb_flexion',
                            'index_flexion', 'middle_flexion', 'ring_and_pinky_flexion']
        self.raw_joint_names = ['forearm__base', 'palm_axis__forearm', 'palm__palm_axis', 'palm__thumb_base',
                                'Tproximal__thumb_base', 'Iproximal__palm', 'Mproximal__palm', 'Rproximal__palm']
        self.joint_names_to_raw = dict(zip(self.joint_names, self.raw_joint_names))
        self.waiting_meta_controllers = []
        self.controllers = {}
        self.serial_proxies = {}
        self.diagnostics_rate = rospy.get_param('~diagnostics_rate', 1)

        self.start_controller_lock = Lock()
        self.stop_controller_lock = Lock()

        manager_namespace = rospy.get_param('~namespace')
        serial_ports = rospy.get_param('~serial_ports')

        for port_namespace, port_config in serial_ports.items():
            port_name = port_config['port_name']
            baud_rate = port_config['baud_rate']
            readback_echo = port_config['readback_echo'] if 'readback_echo' in port_config else False
            min_motor_id = port_config['min_motor_id'] if 'min_motor_id' in port_config else 0
            max_motor_id = port_config['max_motor_id'] if 'max_motor_id' in port_config else 253
            update_rate = port_config['update_rate'] if 'update_rate' in port_config else 5
            error_level_temp = 75
            warn_level_temp = 70

            if 'diagnostics' in port_config:
                if 'error_level_temp' in port_config['diagnostics']:
                    error_level_temp = port_config['diagnostics']['error_level_temp']
                if 'warn_level_temp' in port_config['diagnostics']:
                    warn_level_temp = port_config['diagnostics']['warn_level_temp']

            serial_proxy = SerialProxy(port_name, port_namespace, baud_rate, min_motor_id, max_motor_id, update_rate,
                                       self.diagnostics_rate, error_level_temp, warn_level_temp, readback_echo)
            serial_proxy.connect()

            # will create a set of services for each serial port under common manager namesapce
            # e.g. /dynamixel_manager/robot_arm_port/start_controller
            #      /dynamixel_manager/robot_head_port/start_controller
            # where 'dynamixel_manager' is manager's namespace
            #       'robot_arm_port' and 'robot_head_port' are human readable names for serial ports
            rospy.Service('%s/%s/start_controller' % (manager_namespace, port_namespace), StartController,
                          self.start_controller)
            rospy.Service('%s/%s/stop_controller' % (manager_namespace, port_namespace), StopController,
                          self.stop_controller)
            rospy.Service('%s/%s/restart_controller' % (manager_namespace, port_namespace), RestartController,
                          self.restart_controller)

            self.serial_proxies[port_namespace] = serial_proxy

        # services for 'meta' controllers, e.g. joint trajectory controller
        # these controllers don't have their own serial port, instead they rely
        # on regular controllers for serial connection. The advantage of meta
        # controller is that it can pack commands for multiple motors on multiple
        # serial ports.
        # NOTE: all serial ports that meta controller needs should be managed by
        # the same controler manager.
        rospy.Service('%s/meta/start_controller' % manager_namespace, StartController, self.start_controller)
        rospy.Service('%s/meta/stop_controller' % manager_namespace, StopController, self.stop_controller)
        rospy.Service('%s/meta/restart_controller' % manager_namespace, RestartController, self.restart_controller)

        self.diagnostics_pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=1)
        self.joint_states_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
        if self.diagnostics_rate > 0:
            Thread(target=self.diagnostics_processor).start()

    def on_shutdown(self):
        for serial_proxy in self.serial_proxies.values():
            serial_proxy.disconnect()

    def diagnostics_processor(self):
        diag_msg = DiagnosticArray()

        rate = rospy.Rate(self.diagnostics_rate)
        while not rospy.is_shutdown():
            diag_msg.status = []
            diag_msg.header.stamp = rospy.Time.now()

            for controller in self.controllers.values():
                try:
                    joint_state = controller.joint_state
                    temps = joint_state.motor_temps
                    max_temp = max(temps)

                    status = DiagnosticStatus()
                    status.name = 'Joint Controller (%s)' % controller.joint_name
                    status.hardware_id = 'Robotis Dynamixel %s on port %s' % (
                        str(joint_state.motor_ids), controller.port_namespace)
                    status.values.append(KeyValue('Goal', str(joint_state.goal_pos)))
                    status.values.append(KeyValue('Position', str(joint_state.current_pos)))
                    status.values.append(KeyValue('Error', str(joint_state.error)))
                    status.values.append(KeyValue('Velocity', str(joint_state.velocity)))
                    status.values.append(KeyValue('Load', str(joint_state.load)))
                    status.values.append(KeyValue('Moving', str(joint_state.is_moving)))
                    status.values.append(KeyValue('Temperature', str(max_temp)))
                    status.level = DiagnosticStatus.OK
                    status.message = 'OK'

                    target_index = self.joint_names.index(controller.joint_name)
                    self.current_joint_state[target_index] = joint_state.current_pos
                    diag_msg.status.append(status)  # print(status.name, joint_state.current_pos)
                except:
                    pass

            self.diagnostics_pub.publish(diag_msg)
            message = JointState()
            message.name = self.raw_joint_names
            message.header.stamp = rospy.Time.now()
            message.position = self.map_to_simulated_frames(self.current_joint_state)
            self.joint_states_pub.publish(message)
            rate.sleep()

    def check_deps(self):
        controllers_still_waiting = []

        for i, (controller_name, deps, kls) in enumerate(self.waiting_meta_controllers):
            if not set(deps).issubset(self.controllers.keys()):
                controllers_still_waiting.append(self.waiting_meta_controllers[i])
                rospy.logwarn('[%s] not all dependencies started, still waiting for %s...' % (
                    controller_name, str(list(set(deps).difference(self.controllers.keys())))))
            else:
                dependencies = [self.controllers[dep_name] for dep_name in deps]
                controller = kls(controller_name, dependencies)

                if controller.initialize():
                    controller.start()
                    self.controllers[controller_name] = controller

        self.waiting_meta_controllers = controllers_still_waiting[:]

    def start_controller(self, req):
        port_name = req.port_name
        package_path = req.package_path
        module_name = req.module_name
        class_name = req.class_name
        controller_name = req.controller_name

        self.start_controller_lock.acquire()

        if controller_name in self.controllers:
            self.start_controller_lock.release()
            return StartControllerResponse(False, 'Controller [%s] already started. If you want to restart it, '
                                                  'call restart.' % controller_name)

        try:
            if module_name not in sys.modules:
                # import if module not previously imported
                package_module = __import__(package_path, globals(), locals(), [module_name], -1)
            else:
                # reload module if previously imported
                package_module = reload(sys.modules[package_path])
            controller_module = getattr(package_module, module_name)
        except ImportError, ie:
            self.start_controller_lock.release()
            return StartControllerResponse(False, 'Cannot find controller module. Unable to start controller %s\n%s' % (
                module_name, str(ie)))
        except SyntaxError, se:
            self.start_controller_lock.release()
            return StartControllerResponse(False,
                                           'Syntax error in controller module. Unable to start controller %s\n%s' % (
                                               module_name, str(se)))
        except Exception, e:
            self.start_controller_lock.release()
            return StartControllerResponse(False, 'Unknown error has occured. Unable to start controller %s\n%s' % (
                module_name, str(e)))

        kls = getattr(controller_module, class_name)

        if port_name == 'meta':
            self.waiting_meta_controllers.append((controller_name, req.dependencies, kls))
            self.check_deps()
            self.start_controller_lock.release()
            return StartControllerResponse(True, '')

        if port_name != 'meta' and (port_name not in self.serial_proxies):
            self.start_controller_lock.release()
            return StartControllerResponse(False,
                                           'Specified port [%s] not found, available ports are %s. Unable to start '
                                           'controller %s' % (
                                               port_name, str(self.serial_proxies.keys()), controller_name))

        controller = kls(self.serial_proxies[port_name].dxl_io, controller_name, port_name)

        if controller.initialize():
            controller.start()
            self.controllers[controller_name] = controller

            self.check_deps()
            self.start_controller_lock.release()

            return StartControllerResponse(True, 'Controller %s successfully started.' % controller_name)
        else:
            self.start_controller_lock.release()
            return StartControllerResponse(False,
                                           'Initialization failed. Unable to start controller %s' % controller_name)

    def stop_controller(self, req):
        controller_name = req.controller_name
        self.stop_controller_lock.acquire()

        if controller_name in self.controllers:
            self.controllers[controller_name].stop()
            del self.controllers[controller_name]
            self.stop_controller_lock.release()
            return StopControllerResponse(True, 'controller %s successfully stopped.' % controller_name)
        else:
            self.stop_controller_lock.release()
            return StopControllerResponse(False, 'controller %s was not running.' % controller_name)

    def restart_controller(self, req):
        response1 = self.stop_controller(StopController(req.controller_name))
        response2 = self.start_controller(req)
        return RestartControllerResponse(response1.success and response2.success,
                                         '%s\n%s' % (response1.reason, response2.reason))

    def map_to_simulated_frames(self, angles):
        joint_angles = []
        for i in range(len(angles)):
            joint_angle = map_to_sim_frame(self.raw_joint_names[i], angles[i])
            joint_angles.append(joint_angle)
        return joint_angles


if __name__ == '__main__':
    try:
        manager = ControllerManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
