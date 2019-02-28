#include "ros/ros.h"
#include <geometry_msgs/Wrench.h>
#include <cereal_port/CerealPort.h>
#include <sstream>
#include <fstream>

int main(int argc, char **argv){
  ros::init(argc, argv, "force_torque_sensor");
  
  ros::NodeHandle n;
  ros::Publisher wrench_pub = n.advertise<geometry_msgs::Wrench>("force_torque_readings", 10);
  
  cereal::CerealPort cp;
  cp.open("/dev/ttyUSB0", 19200);
  cp.write("OA\r",3);

  //  register the callback
  ros::Rate loop_rate(300);
  
  while (ros::ok()){
    geometry_msgs::Wrench wrench;

    std::string serial_read_in;
    double valid_read;

    do{
      // read message into string
      cp.readLine(&serial_read_in,-1);
      std::stringstream ss;
      ss << serial_read_in;

      // parse message into valid_read, and message
      ss >> valid_read;
      // Forces in Newtons
      ss >> wrench.force.x;
      ss >> wrench.force.y;
      ss >> wrench.force.z;
      // Torques in Newton Meters
      ss >> wrench.torque.x;
      ss >> wrench.torque.y;
      ss >> wrench.torque.z;
  
    }while(valid_read != 0);

    //pubish the new ft sensor reading
    wrench_pub.publish(wrench);
    
    loop_rate.sleep();
  }

  return 0;
}

