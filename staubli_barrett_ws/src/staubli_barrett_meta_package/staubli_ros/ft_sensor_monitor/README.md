# ft_sensor_monitor
Code for reading from the force torque sensor

## Setup
```bash
sudo chmod a+rw /dev/ttyUSB0
```
This to to set up correct permissions for device. There is probably a way to do this using udev but unsure. 

## Running
Make sure you plug in the force torque sensor into an available USB port and it is switched on. 
```bash
$ cd <catkin_ws>
$ source devel/setup.bash
# After a ROS master has been started
$ rosrun ft_sensor_monitor force_torque_server
```

The force_torque_watcher, on the other hand, watches the force_torque_readings to see if the sensor has been hit by anything. This is a good way to check if the hand has collided with anything. 
```bash
$ cd <catkin_ws>
$ source devel/setup.bash
# After a ROS master has been started
$ rosrun ft_sensor_monitor force_torque_watcher
```

## Subscribing
The force_torque_server will publish a message of type `geometry_msgs/Wrench`, populating the torque and force fields. You can create a simple subscriber in python like so:
```python
import rospy
import geometry_msgs.msg

rospy.init_node("tmp")
def listener(torque_msg):
    print(torque_msg)

sub = rospy.Subscriber("force_torque_readings", geometry_msgs.msg.Wrench, listener)
```
