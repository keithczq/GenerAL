#! /usr/bin/env python
import collections
import os
import threading
import time

import curvox.cloud_transformations
import cv2
import numpy as np
import pcl
import rospy
import sensor_msgs.msg
import tf
import tf_conversions.posemath
from cv_bridge import CvBridge

_rgbd_data_collector = None

RGBDAggregate = collections.namedtuple('RGBDAggregate',
                                       ['rgb_image', 'depth_image', 'point_cloud_array', 'point_cloud_pcl',
                                        'camera2world_tf_mat'])


class RGBDDataCollector:
    def __init__(self, rgb_image_topic, depth_image_topic, cloud_topic, camera_frame=None, world_frame=None):
        if not rospy.core.is_initialized():
            rospy.init_node('get_rgbd_server', anonymous=False)

        self._read_lock = threading.Lock()
        self._br = CvBridge()

        self._rgb_sub = rospy.Subscriber(rgb_image_topic, sensor_msgs.msg.Image, self.rgb_callback, queue_size=1)
        self._depth_sub = rospy.Subscriber(depth_image_topic, sensor_msgs.msg.Image, self.depth_callback, queue_size=1)
        self._pcl_sub = rospy.Subscriber(cloud_topic, sensor_msgs.msg.PointCloud2, self.pcl_callback, queue_size=1)

        self._tf_listener = tf.TransformListener()
        self._camera_frame = camera_frame
        self._world_frame = world_frame

        self._rgb_image = None
        self._depth_image = None
        self._pcl_data = None
        self._pcl_array = None
        self._camera2world_tf_mat = None

        if self.capturing_tf_frames:
            self._rate = rospy.Rate(10)  # 10hz
            self._still_running = True
            rospy.on_shutdown(self._shutdown)
            self._tf_thread = threading.Thread(target=self.tf_callback)
            self._tf_thread.daemon = True
            self._tf_thread.start()

    def _shutdown(self):
        if self.capturing_tf_frames:
            self._still_running = False
            self._tf_thread.join()

    def rgb_callback(self, data):
        if self._read_lock.locked():
            return
        self._rgb_image = self._br.imgmsg_to_cv2(data)

    def depth_callback(self, data):
        if self._read_lock.locked():
            return
        self._depth_image = self._br.imgmsg_to_cv2(data)

    def pcl_callback(self, data):
        if self._read_lock.locked():
            return
        self._pcl_array = curvox.cloud_transformations.cloud_msg_to_np(data)
        self._pcl_data = pcl.PointCloud()
        self._pcl_data.from_array(self._pcl_array)

    def tf_callback(self):
        while not rospy.is_shutdown() and self._still_running:
            if not self._read_lock.locked():
                now = rospy.Time(0)
                self._tf_listener.waitForTransform(self._camera_frame, self._world_frame, now,
                                                   timeout=rospy.Duration(5))
                tf_msg = self._tf_listener.lookupTransform(self._camera_frame, self._world_frame, now)
                camera2world_tf_msg = tf_conversions.posemath.fromTf(tf_msg)
                self._camera2world_tf_mat = tf_conversions.posemath.toMatrix(camera2world_tf_msg)
            self._rate.sleep()

    def get_rgbd_data(self):
        with self._read_lock:

            if not self.ready():
                print('not ready')
                return None

            response = RGBDAggregate(rgb_image=self._rgb_image, depth_image=self._depth_image,
                point_cloud_array=self._pcl_array, point_cloud_pcl=self._pcl_data,
                camera2world_tf_mat=self._camera2world_tf_mat)
            return response

    def ready(self):
        if self.capturing_tf_frames:
            return self._rgb_image is not None and self._depth_image is not None and self._pcl_array is not None and \
                   self._pcl_data is not None and self._camera2world_tf_mat is not None
        else:
            return self._rgb_image is not None and self._depth_image is not None and self._pcl_array is not None and \
                   self._pcl_data is not None

    def __del__(self):
        self._shutdown()

    capturing_tf_frames = property(lambda self: self._camera_frame is not None and self._world_frame is not None)


def init_service(rgb_image_topic, depth_image_topic, cloud_topic, camera_frame=None, world_frame=None):
    global _rgbd_data_collector
    _rgbd_data_collector = RGBDDataCollector(rgb_image_topic=rgb_image_topic, depth_image_topic=depth_image_topic,
        camera_frame=camera_frame, world_frame=world_frame, cloud_topic=cloud_topic)


def get_rgbd():
    global _rgbd_data_collector
    if not _rgbd_data_collector:
        raise ValueError("RGBDDataCollector has not been initialized. Did you call 'init_service'?")
    return _rgbd_data_collector.get_rgbd_data()


def save_rgbd_example(rgbd_example, save_location, index):
    # type: (RGBDAggregate, str, any) -> ()

    if rgbd_example is None:
        return

    if not os.path.isdir(save_location):
        os.makedirs(save_location)

    pcl.save(rgbd_example.point_cloud_pcl, os.path.join(save_location, '{}_pcl.pcd'.format(index)), binary=True)
    cv2.imwrite(os.path.join(save_location, '{}_rgb.png'.format(index)), rgbd_example.rgb_image)
    cv2.imwrite(os.path.join(save_location, '{}_depth.png'.format(index)), rgbd_example.depth_image)
    if rgbd_example.camera2world_tf_mat is not None:
        np.savetxt(os.path.join(save_location, '{}_camera2world_tf.npy'.format(index)),
                   rgbd_example.camera2world_tf_mat)


def ready():
    global _rgbd_data_collector
    return _rgbd_data_collector.ready()


def main():
    rgb_image_topic = "/camera/rgb/image_color"
    depth_image_topic = "/camera/depth/image_raw"

    cloud_topic = '/camera/depth/points'

    camera_frame = 'camera_link'
    world_frame = 'ar_marker_0'

    save_location = './examples'

    init_service(rgb_image_topic, depth_image_topic, cloud_topic, camera_frame, world_frame)
    # init_service(rgb_image_topic, depth_image_topic, cloud_topic)  # Omit camera_frame and world_frame if you don't
    #  want tracking

    rate = rospy.Rate(10)
    while not ready():
        rate.sleep()

    for i in range(100):
        if rospy.is_shutdown():
            exit()

        start_time = time.time()
        d = get_rgbd()
        print("Example {}".format(i))
        print("Capture Time taken: {}s".format(time.time() - start_time))
        start_time = time.time()
        if d:
            save_rgbd_example(d, save_location, i)
        print("Save Time taken: {}s".format(time.time() - start_time))


if __name__ == '__main__':
    main()
