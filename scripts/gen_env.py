import os


def gen_xml(obj):
    return """<?xml version="1.0" ?>
<world>
    <obstacle>
        <filename>models/obstacles/table.xml</filename>
        <transform>
            <fullTransform>(+1 0 1 0)[0 -500 1000]</fullTransform>
        </transform>
    </obstacle>
    <graspableBody>
        <filename>models/graspable_bodies/%s.xml</filename>
        <transform>
            <fullTransform>(+1 0 0 0)[+8.37264e-06 +0.781957 +105.818]</fullTransform>
        </transform>
    </graspableBody>
    <robot>
        <filename>models/robots/BarrettBH8_280_Tactile/BarrettBH8_280_Tactile.xml</filename>
        <dofValues>0 0 +0 +0 0 +0 +0 0 +0 +0</dofValues>
        <transform>
            <fullTransform>(+1 0 0 0)[0 0 0]</fullTransform>
        </transform>
    </robot>
    <camera>
        <position>+512.727 +11.2119 +92.4714</position>
        <orientation>+0.494626 +0.482173 +0.511852 +0.510746</orientation>
        <focalDistance>+501</focalDistance>
    </camera>
</world>
""" % obj


src_path = '/Volumes/ubuntu/home/bohan/Desktop/graspit/models/graspable_bodies'
dst_path = '/Volumes/ubuntu/home/bohan/Desktop/graspit/worlds'

objects = list([i[:i.find('.')] for i in os.listdir(src_path)])

good_objects = set(
    [obj for obj in objects if '%s.xml' % obj in os.listdir(src_path) and '%s.iv' % obj in os.listdir(src_path)])
print(len(good_objects))
print(good_objects)

for obj in good_objects:
    xml = gen_xml(obj)
    with open('/Volumes/ubuntu/home/bohan/Desktop/graspit/worlds/planner%sTactile.xml' % obj.capitalize(), 'w') as f:
        f.write(xml)

print(['planner%sTactile' % obj.capitalize() for obj in good_objects])
