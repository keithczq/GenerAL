import json
import os
import time
import traceback

import tensorflow as tf
import yagmail

from config import d, g, n

if __name__ == "__main__":
    if g.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    try:
        for server in g.servers:
            if server != 'localhost' and g.simulator == 'graspit':
                os.system('make fs d=%s u=%s' % (server, g.server_user))
        module = __import__("algorithms.%s" % d.config.algo, fromlist=[d.config.algo])
        algo_class = getattr(module, d.config.algo)
        algo_class()
        subject = '%s: Job Succeeded!' % g.ip
        content = n() + '\n'
    except:
        subject = '%s: Job Failed!' % g.ip
        content = '%s\n%s' % (n(), traceback.format_exc())
        print(content)

    with open('credential.json', mode='r') as f:
        login = json.load(f)

    success = False
    while not success:
        time.sleep(1)
        try:
            print('Trying to send email')
            yag = yagmail.SMTP(login['from'], login['password'])
            yag.send(login["to"], subject=subject, contents=content)
            print("Email sent")
            success = True
        except:
            print(subject)
            print(content)
            print(traceback.format_exc())
            success = False
