#! /usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import threading

from kinect_rgdb_collector.srv import GetRGBD, GetRGBDResponse


class RGBDDataCollector:
    def __init__(self, rgb_image_topic, depth_image_topic):
        self.lock = threading.Lock()

        self.rgb_sub = rospy.Subscriber(rgb_image_topic, Image, self.rgb_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(depth_image_topic, Image, self.depth_callback, queue_size=1)

        self.rgb_image = None
        self.depth_image = None

        rospy.Service('get_rgbd_service', GetRGBD, self.get_rgbd_data)

    def rgb_callback(self, data):
        with self.lock:
            self.rgb_image = data

    def depth_callback(self, data):
        with self.lock:
            self.depth_image = data

    def get_rgbd_data(self, req):
        response = GetRGBDResponse()
        response.rgb_image = self.rgb_image
        response.depth_image = self.depth_image
        return response


if __name__ == '__main__':
    rospy.init_node('get_rgbd_server', anonymous=False)

    rgb_image_topic = "/camera/rgb/image_raw"
    depth_image_topic = "/camera/depth/image_raw"

    data_collector = RGBDDataCollector(rgb_image_topic, depth_image_topic)

    rospy.spin()
