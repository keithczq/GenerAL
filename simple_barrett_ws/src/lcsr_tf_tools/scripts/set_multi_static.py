#!/usr/bin/env python
import sys
import argparse

import roslib; roslib.load_manifest('lcsr_tf_tools')
import rospy
import tf

import PyKDL

import std_msgs.msg as std_msgs
import geometry_msgs.msg as geometry_msgs
import lcsr_tf_tools.msg as lcsr_tf_tools
import tf.msg

from multi_static_transform_publisher import MultiPublisher

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Set frames being broadcast by a multi_static_transform_publisher')
    parser.add_argument('--node-name', 
            metavar=('node_name'), 
            default=['multi_tf_pub'],
            type=str, nargs=1,
            help='The name of the multi publisher that should publish this transform. Default is \'multi_tf_pub\'')

    parser.add_argument('--xyz', 
            metavar=('x','y','z'),
            default=[0.0,0.0,0.0],
            type=float, nargs=3,
            help='Position in x, y, z)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quat', 
            metavar=('qx','qy','qz','qw'), 
            default=[0.0,0.0,0.0,1.0],
            type=float, nargs=4,
            help='Orientation in quaternion')
    group.add_argument('--aa', 
            metavar=('x','y','z','t'), 
            type=float, nargs=4,
            help='Orientation in axis/angle x, y, z, theta')
    group.add_argument('--ypr', 
            metavar=('yaw','pitch','roll'), 
            type=float, nargs=3,
            help='Orientation in yaw, pitch, roll')

    parser.add_argument('parent_frame_id', 
            metavar=('frame_id'), 
            type=str, nargs=1,
            help='The frame_id of the frame in which new new frame is defined.')
    parser.add_argument('child_frame_id', 
            metavar=('child_frame_id'), 
            type=str, nargs=1,
            help='The frame_id of the new frame.')
    parser.add_argument('period', 
            metavar=('period'), 
            type=float, nargs=1,
            help='Publish period in ms.')

    myargv = rospy.myargv(argv=sys.argv)
    args = parser.parse_args(args=myargv[1:])

    rospy.init_node('multi_static')

    # Check if other rotation format is used
    if args.ypr:
        # Transform YPR to quat
        rotation = PyKDL.Rotation.RPY(args.ypr[2],args.ypr[1],args.ypr[0])
        args.quat = rotation.GetQuaternion()
    elif args.aa:
        # Transform axis angle to quat
        axis = PyKDL.Vector(args.aa[0],args.aa[1],args.aa[2])
        axis.Normalize()
        rotation = PyKDL.Rotation.Rot(axis, args.aa[3])
        args.quat = rotation.GetQuaternion()

    # Construct transform message
    tform = geometry_msgs.TransformStamped()
    tform.header.frame_id = args.parent_frame_id[0]
    tform.header.stamp = rospy.Time(0)

    tform.child_frame_id = args.child_frame_id[0]

    tform.transform.translation.x = args.xyz[0]
    tform.transform.translation.y = args.xyz[1]
    tform.transform.translation.z = args.xyz[2]

    tform.transform.rotation.x = args.quat[0]
    tform.transform.rotation.y = args.quat[1]
    tform.transform.rotation.z = args.quat[2]
    tform.transform.rotation.w = args.quat[3]

    static_tform = lcsr_tf_tools.StaticTransform()
    static_tform.header.stamp = rospy.Time.now()
    static_tform.transform = tform
    static_tform.publish_period = rospy.Duration(1E-3*args.period[0])

    # Publish the transform
    rospy.loginfo("Registering static transform %s --> %s\n%s" %(tform.header.frame_id, tform.child_frame_id, str(tform)))
    set_pub = None

    # Get a tf_listener to make sure the frame gets published
    listener = tf.TransformListener(True, rospy.Duration(10.0))

    # Wait for the frame to be published
    while listener and not rospy.is_shutdown():
        static_tform.header.stamp = rospy.Time.now()
        set_pub = rospy.Publisher(args.node_name[0]+'/set_frame', lcsr_tf_tools.StaticTransform, latch=True)
        set_pub.publish(static_tform)
        try:
            now = rospy.Time(0)
            listener.waitForTransform(tform.header.frame_id, tform.child_frame_id, now, rospy.Duration(1.0))
            pose = listener.lookupTransform(tform.header.frame_id, tform.child_frame_id, now)
            rospy.loginfo("Registered static transform %s --> %s\n%s: %s" %(tform.header.frame_id, tform.child_frame_id, str(tform), str(pose)))
            listener = None
        except tf.Exception as ex:
            rospy.logwarn("Multi static transform %s --> %s not being published yet: %s" % (tform.header.frame_id, tform.child_frame_id, str(ex)))

    # Wait for sigint
    rospy.spin()

if __name__ == '__main__':
    main()
