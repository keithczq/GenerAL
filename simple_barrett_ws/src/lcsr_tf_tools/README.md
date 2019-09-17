LCSR TF Tools
=============

This package contains a bunch of utilities that have been created in the aim to
fill some usability gaps in TF.

## Tools

### Interactive Transform Publisher

This program creates a TF frame based off of an interactive marker which can be manipulated in RViz.

For example, to create an interactive tf frame, you can run this script, and then add an "Interactive Marker" display to an instance of RViz.

```
rosrun lcsr_tf_tools interactive_transform_publisher 0 0 0  0 0 0 /world /cmd_frame 10
```


### Twist Frame Integrator

This script integrates `geomatry_msgs/TwistStamped` messages and broadcasts
a TF frame of a given name.

Paramters:

* `~frame_id`
* `~child_frame_id`
* `~linear_multiplier`
* `~angular_multiplier`
* `~broadcast_rate`
* `~body_fixed`

### Multi-Static Publisher

This node listens on a topic for static transforms to publish, and then
re-publishes them as batch messages when it can do so. It will publish
at the highest common period of the requested frames.

The publisher itself has no arguments, but instead it is controlled by running
two other scripts:
- set_multi_static.py
- del_multi_static.py

This is a stop-gap until TF2's static transforms are better supported.

Usage:

```
set_multi_static.py [-h] [--node-name node_name] [-xyz x y z]
                           [-aa x y z t | -ypr yaw pitch roll | -quat qx qy qz qw]
                           frame_id child_frame_id period

Set frames being broadcast by a multi_static_transform_publisher

positional arguments:
  frame_id              The frame_id of the frame in which new new frame is
                        defined.
  child_frame_id        The frame_id of the new frame.
  period                Publish period in ms.

optional arguments:
  -h, --help            show this help message and exit
  --node-name node_name
                        The name of the multi publisher that should publish
                        this transform. Default is 'multi_tf_pub'
  -xyz x y z            Position in x, y, z)
  -aa x y z t           Orientation in axis/angle x, y, z, theta
  -ypr yaw pitch roll   Orientation in yaw, pitch, roll
  -quat qx qy qz qw     Orientation in quaternion
```


Usage:

```
del_multi_static.py [-h] [--node-name node_name]
                           frame_id [frame_id ...]

Delete frames being broadcast by a multi_static_transform_publisher

positional arguments:
  frame_id              The frame_ids of each frame to stop broadcasting.

optional arguments:
  -h, --help            show this help message and exit
  --node-name node_name
                        The name of the multi publisher that should publish
                        this transform. Default is 'multi_tf_pub'
