import argparse
import subprocess
import traceback

from config import g

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, help='port')

args = parser.parse_args()

if args.port in g.ports:
    while True:
        try:
            print(subprocess.check_output(
                'export ROS_MASTER_URI=http://localhost:%s && roslaunch graspit_interface graspit_interface.launch' %
                args.port,
                shell=True, stderr=subprocess.STDOUT))
        except:
            traceback.print_exc()
