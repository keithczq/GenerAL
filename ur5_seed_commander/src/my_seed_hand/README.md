1. roslaunch my_seed_hand controller_manager.launch serial_port:=/dev/ttyUSB0
2. roslaunch my_seed_hand start_meta_controller.launch

3. operate seed hand from computer
    1. python my_seed_hand/motion_controller/action_client/simple_client.py

4. teleoperate seed hand (use two different command prompts)
    1. python my_teleop/my_methods/linear_map/linear_map_publisher.py -a (for armband) -d (for dataglove)
    2. python my_seed_hand/motion_controller/action_client/trajectory_client.py