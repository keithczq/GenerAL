import os

dir = os.path.expanduser('~/Desktop/graspit/models/objects')

for i in os.listdir(dir):
    if i.endswith('.iv') or i.endswith('.IV'):
        iv_path = os.path.join(dir, i)
        obj_path = os.path.join(dir, '%s.obj' % os.path.splitext(i)[0])
        os.system('ivcon %s %s' % (iv_path, obj_path))