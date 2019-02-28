import os

from config import d

for di in os.listdir(d.pybullet_obj_dir):
    if os.path.isdir(os.path.join(d.pybullet_obj_dir, di)):
        for f in os.listdir(os.path.join(d.pybullet_obj_dir, di)):
            if '.stl' in f or '.obj' in f:
                urdf_filename = os.path.splitext(os.path.join(d.pybullet_obj_dir, di, f))[0] + '.urdf'
                obj = os.path.splitext(f)[0]
                urdf_str = """
                <robot name="%s">
                    <link name="base_link">
                        <contact>
                            <friction_anchor/>
                            <lateral_friction value="1.0"/>
                        </contact>
                        <visual>
                            <geometry><mesh filename="%s"/></geometry>
                        </visual>
                        <collision>
                            <geometry><mesh filename="%s"/></geometry>
                            <surface>
                                <friction>
                                  <torsional>
                                    <coefficient>1.0</coefficient>
                                    <surface_radius>0.5</surface_radius>
                                    <use_patch_radius>false</use_patch_radius>
                                  </torsional>
                                </friction>
                            </surface>
                        </collision>
                        <inertial>
                            <origin rpy="0 0 0" xyz="0 0 0"/>
                            <mass value="10"/>
                            <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
                        </inertial>
                    </link>
                </robot>""" % (obj, f, f)
                with open(urdf_filename, 'w') as f:
                    f.write(urdf_str)
                    f.close()
                print(urdf_filename)
