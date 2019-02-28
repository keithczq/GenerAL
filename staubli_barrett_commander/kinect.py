import rospy

import kinect_rgbd_standalone

kinect_rgbd_standalone.init_service("/camera/rgb/image_raw", "/camera/depth_registered/image_raw", 'camera_link',
                                    'camera_link')

rate = rospy.Rate(10)
while not kinect_rgbd_standalone.ready():
    print('not ready')
    rate.sleep()

if rospy.is_shutdown():
    exit()

d = kinect_rgbd_standalone.get_rgbd()
print(d.depth_image.shape)
