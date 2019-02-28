import os
import time
import traceback

import numpy as np

from graspit_commander import GraspitCommander
from utils import gen_example

GraspitCommander.clearWorld()
# GraspitCommander.loadWorld("barrettTactileGlassDyn")
GraspitCommander.loadWorld("plannerMugTactile")

save_dir = os.path.expanduser('~/Desktop/critic_image_dataset')

batch_size = 10000

while True:
    states = []
    values = []
    images = []
    count = 0
    while count < batch_size:
        try:
            state, value, image = gen_example(GraspitCommander)
            states.append(state)
            values.append(value)
            images.append(image)
            count += 1
            print(count, batch_size)
        except:
            traceback.print_exc()
            pass
    time_str = str(int(time.time()))
    np.save(os.path.join(save_dir, time_str + '_x.npy'), np.array(states))
    np.save(os.path.join(save_dir, time_str + '_image.npy'), np.array(images))
    np.save(os.path.join(save_dir, time_str + '_y.npy'), np.array(values))
