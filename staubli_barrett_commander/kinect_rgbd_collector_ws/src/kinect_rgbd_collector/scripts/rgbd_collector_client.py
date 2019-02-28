#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt

from cv_bridge import CvBridge
from kinect_rgdb_collector.srv import GetRGBD, GetRGBDRequest


def get_rgbd():
	get_rgbd_service_name = 'get_rgbd_service'
	rospy.wait_for_service(get_rgbd_service_name)
	get_rgbd_service_session = rospy.ServiceProxy(get_rgbd_service_name, GetRGBD)

	response = get_rgbd_service_session(GetRGBDRequest())

	bridge = CvBridge()
	rgb_image = bridge.imgmsg_to_cv2(response.rgb_image, desired_encoding="rgb8")
	depth_image = bridge.imgmsg_to_cv2(response.depth_image, desired_encoding="passthrough")
	return rgb_image, depth_image


def main():
	rgb_image, depth_image = get_rgbd()
	f = plt.figure()
	f.add_subplot(1, 2, 1)
	plt.imshow(rgb_image)
	f.add_subplot(1, 2, 2)
	plt.imshow(depth_image)
	plt.show()

	import IPython
	IPython.embed()


if __name__ == "__main__":
	main()
