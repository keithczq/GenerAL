from __future__ import print_function
import json
import os
import time
import traceback

import yagmail

from config import d, g, n

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    try:
        module = __import__("algorithms.%s" % d.config.algo, fromlist=[d.config.algo])
        algo_class = getattr(module, d.config.algo)
        algo_class(eval=True)
        with open('results.json', 'r') as f:
            j = json.load(f)
        subject = '%s: Job Succeeded! %s %s %s %s' % (
        g.ip, g.finger_closing_only, g.regrasp_only, g.pos_adjustment_only, g.ori_adjustment_only)
        content = str(
            ['test_single_seen_ret', 'test_single_novel_ret', 'test_multi_seen_ret, test_multi_novel_ret']) + '\n'
        content += str(j['mean']) + '\n'
        content += str(j['std']) + '\n' + g.path
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
            if g.statistics:
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
