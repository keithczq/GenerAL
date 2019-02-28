import os

dir = os.path.expanduser('~/Desktop/YCB')

for i in os.listdir(dir):
    if os.path.isdir(os.path.join(dir, i)):
        obj_path = os.path.join(dir, i, 'google_16k', 'textured.obj')
        iv_path = os.path.join(dir, '%s.iv' % i)
        xml_str = '''<?xml version="1.0" ?>
        <root>
            <material>plastic</material>
            <mass>400.0</mass>
            <cog>0 0 0</cog>
            <geometryFile type="Inventor">%s.iv</geometryFile>
        </root>''' % i
        xml_path = os.path.join(dir, '%s.xml' % i)
        os.system('ivcon %s %s' % (obj_path, iv_path))

        with open(xml_path, 'w') as f:
            f.write(xml_str)
            f.close()
