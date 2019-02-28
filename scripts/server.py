import subprocess

while True:
    print(subprocess.check_output('python3 Citizen/src/Server.py', shell=True,
        stderr=subprocess.STDOUT))