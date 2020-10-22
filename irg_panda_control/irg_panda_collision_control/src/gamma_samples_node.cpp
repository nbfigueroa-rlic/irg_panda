#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <cmath>
#include <vector>

// Linear interpolation following MATLAB linspace
std::vector<double> LinearSpacedArray(double a, double b, std::size_t N)
{
    double h = (b - a) / static_cast<double>(N-1);
    std::vector<double> xs(N);
    std::vector<double>::iterator x;
    double val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h) {
        *x = val;
    }
    return xs;
}


int main( int argc, char** argv )
{
  ros::init(argc, argv, "spheres");
  ros::NodeHandle n;
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("gamma_samples_3D", 10);
  ros::Rate r(30);

  double grid_size   = 50;
  std::vector<double> y_row     = LinearSpacedArray(-0.8, 0.8, grid_size);
  std::vector<double> x_column  = LinearSpacedArray(0.1, 1.0, grid_size);
  std::vector<double> z_stacks  = LinearSpacedArray(0.55, 1.1, grid_size);        

  // float f = 0.0;
  while (ros::ok())
  {

    visualization_msgs::Marker sphere_list;
    sphere_list.header.frame_id= "/world";
    sphere_list.header.stamp= ros::Time::now();
    sphere_list.ns= "points";
    sphere_list.action= visualization_msgs::Marker::ADD;
    sphere_list.pose.orientation.w= 1.0;

    sphere_list.id = 0;
    sphere_list.type = visualization_msgs::Marker::SPHERE_LIST;

    // POINTS markers use x and y scale for width/height respectively
    sphere_list.scale.x = 0.01;
    sphere_list.scale.y = 0.01;
    sphere_list.scale.z = 0.01;

    for (uint32_t k = 0; k < grid_size; ++k){
      float z =  z_stacks[k];
        for (uint32_t i = 0; i < grid_size; ++i){
          float y =  y_row[i];
          
          for (uint32_t j = 0; j < grid_size; ++j){
            float x =  x_column[j];
            
            geometry_msgs::Point p;
            p.x = x;
            p.y = y;
            p.z = z;            

            std_msgs::ColorRGBA c;      
            // Set all the free space  
            c.r = 0.0;
            c.g = 1.0;
            c.b = 0.0;
            c.a = 0.01;

            // The table Top
            if (z < 0.625){
                c.r = 1.0;
                c.g = 0.0;              
                c.a = 0.075;
            }
            // The vertical wall
            if (x>= 0.3){
              // if (y>=-0.02 && y<=0.02){
               if (y>=-0.04 && y<=0.04){// Adding 2cm no the sides (to account for gripper)
                if (z >= 0.625 && z <= 1.025){
                  c.r = 1.0;
                  c.g = 0.0;
                  c.a = 0.075;
                }
              }
            }  
            // The horizontal wall
            if (x>= 0.3){
              // if (y>=-0.4 && y<=0.4){
              if (y>=-0.45 && y<=0.45){ // Adding 5cm no the sides (to account for gripper)
                // if (z >= 1.025 && z <= 1.065){
                  if (z >= 0.975 && z <= 1.065){ // Adding 5cm below the wall (to account for gripper)
                  c.r = 1.0;
                  c.g = 0.0;
                  c.a = 0.075;
                }
              }
            }         
            sphere_list.points.push_back(p);
            sphere_list.colors.push_back(c);      
            } 
        }
    }
    marker_pub.publish(sphere_list);

    r.sleep();

    // f += 0.04;
  }
}