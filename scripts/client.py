import subprocess

while True:
    try:
        print(subprocess.check_output('julia -e "using Unitful,Citizen; Citizen.Client.run()"', shell=True,
                                      stderr=subprocess.STDOUT))
    except:
        print("restarting")
