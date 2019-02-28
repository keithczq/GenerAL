# simple_barrett_ws

## Building the workspace
```
git clone git@github.com:CRLab/simple_barrett_ws.git
cd simple_barrett_ws
gitman install
catkin_make
```

## Test Arm Trajectory Planning
Launch Robot with fake controllers and moveit's movegroup:
```
source devel/setup.bash
roslaunch staubli_barretthand_moveit_config demo.launch
```


Move the arm:
```
source devel/setup.bash
python barrett_controller.py
```