import rospy
import bhand_controller.msg
import bhand_controller.srv

print(bhand_controller.msg)
rospy.init_node('test')

hand_service_proxy = rospy.ServiceProxy('/bhand_node/actions', bhand_controller.srv.Actions)
custom_hand_service_proxy = rospy.ServiceProxy('/bhand_node/custom_actions', bhand_controller.srv.CustomActions)

# hand_service_proxy(bhand_controller.msg.Service.INIT_HAND)
# hand_service_proxy(bhand_controller.msg.Service.OPEN_GRASP)
import ipdb; ipdb.set_trace()
assert(False)


# hand_service_proxy(bhand_controller.msg.Service.CLOSE_HALF_GRASP, [])
# hand_service_proxy(bhand_controller.msg.Service.OPEN_GRASP, [])
# hand_service_proxy(bhand_controller.msg.Service.SET_GRASP_1, [])

# custom paremterized commands

# range of spread angle [0, 0.45]
spread_angle = 0.2
custom_hand_service_proxy(bhand_controller.msg.Service.SET_SPREAD, [spread_angle,spread_angle])


# this calls the velocity` control, the first three are the target joint angles while the last value is the velocity
# positive velocity closes the client, negative velocity opens the client
custom_hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY, [0.3, 0.3, 0.3, .1])

custom_hand_service_proxy(bhand_controller.msg.Service.CLOSE_HAND_VELOCITY, [0.01, 0.01, 0.01, -.5])