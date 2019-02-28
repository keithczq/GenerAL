# kinect_rgbd_collector

## Create ROS workspace
```
mkdir kinect_rgbd_collector_ws/src -p
cd kinect_rgbd_collector_ws/src
git clone git@github.com:CRLab/kinect_rgbd_collector.git
cd ..
catkin_make
```

## Launch the kinect device
```
source devel/setup.bash
roslaunch kinect_rgbd_collector vision.launch 
```

## Start the server script
```
source devel/setup.bash
cd src/kinect_rgdb_collector/scripts
python rgbd_collector_server.py
```

## Run sample client script
```
source devel/setup.bash
cd src/kinect_rgdb_collector/scripts
python rgbd_collector_client.py
```

## Example:
```python
import kinect_rgbd_standalone
import time
import rospy
kinect_rgbd_standalone.init_service("/camera/rgb/image_raw", "/camera/depth/image_raw", '/camera/depth/points', 'camera_link', 'camera_link')

rate = rospy.Rate(10)
while not kinect_rgbd_standalone.ready():
	rate.sleep()

for i in range(100):
	if rospy.is_shutdown():
		exit()

	start_time = time.time()
	d = kinect_rgbd_standalone.get_rgbd()
	print("Example {}".format(i))
	print("Capture Time taken: {}s".format(time.time() - start_time))
	start_time = time.time()
	if d:
		kinect_rgbd_standalone.save_rgbd_example(d, './examples', i)
	print("Save Time taken: {}s".format(time.time() - start_time))
```
