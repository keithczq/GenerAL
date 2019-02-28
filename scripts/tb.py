import subprocess

while True:
    print(subprocess.check_output(
        'tensorboard --logdir=self:/home/bohan/Desktop/ws/results1,'
        'syros:/home/bohan/Desktop/results1/syros,fyn:/home/bohan/Desktop/results1/fyn,'
        'curacao:/home/bohan/Desktop/results1/curacao,delfino:/home/bohan/Desktop/results1/delfino --reload_interval '
        '10',
        shell=True, stderr=subprocess.STDOUT))
