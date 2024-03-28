#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>
#include <std_msgs/msg/string.hpp>

// NEW by JP:
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

// IF EIGEN issues: IN TERMINAL:
  // sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
  // sudo ln -s /usr/include/eigen3/unsupported/Eigen /usr/include/unsupported/Eigen


class PointCloudAligner : public rclcpp::Node
{
public:
    PointCloudAligner() : Node("pcl_align")
    {
        // Subscribe to the left and right point cloud topics
        subscription_L_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/input_cloud_1",
            10,
            std::bind(&PointCloudAligner::leftCallback, this, std::placeholders::_1));

        subscription_R_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/input_cloud_2",
            10,
            std::bind(&PointCloudAligner::rightCallback, this, std::placeholders::_1));

        // Publish the combined point cloud
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/output_cloud", 10);
    }

private:
    void leftCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        left_cloud_ = *msg;
        concatenateAndPublish();
    }

    void rightCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        right_cloud_ = *msg;
        // concatenateAndPublish();
    }

    void concatenateAndPublish()
    {
        if (!left_cloud_.data.empty() && !right_cloud_.data.empty())
        {   
            long int left_timestamp = left_cloud_.header.stamp.nanosec;
            long int right_timestamp = right_cloud_.header.stamp.nanosec;
            
            // Convert nanoseconds to seconds
            long double time_difference = std::abs(left_timestamp - right_timestamp) * 1e-9; 
            std::cout << "Time difference:" << time_difference << " s" << std::endl;

            if (time_difference < 0.001) // seconds
            {
                
                sensor_msgs::msg::PointCloud2 combined_cloud;
                if (pcl::concatenatePointCloud(left_cloud_, right_cloud_, combined_cloud)) {
                    combined_cloud.header = left_cloud_.header;
                    publisher_->publish(combined_cloud); }
                else {
                    std::cerr << "Error concatenating point clouds." << std::endl;
                }
            }
        }
    }

    sensor_msgs::msg::PointCloud2 left_cloud_;
    sensor_msgs::msg::PointCloud2 right_cloud_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_L_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_R_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};




// FUNCTION: align point clouds and measure processing time
pcl::PointCloud<pcl::PointXYZ>::Ptr align(boost::shared_ptr<pcl::Registration<pcl::PointXYZ, pcl::PointXYZ>> registration, 
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, 
                                          const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud ) {
  registration->setInputTarget(target_cloud);
  registration->setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

  rclcpp::Clock system_clock;

  ///auto t1 = ros::WallTime::now();
  auto t1 = system_clock.now();
  registration->align(*aligned);
  auto t2 = system_clock.now();
  std::cout << "single : " << (t2 - t1).seconds()* 1000 << "[msec]" << std::endl;

  for(int i=0; i<10; i++) {
    registration->align(*aligned);
  }
  auto t3 = system_clock.now();
  std::cout << "10times: " << (t3 - t2).seconds() * 1000 << "[msec]" << std::endl;
  std::cout << "fitness: " << registration->getFitnessScore() << std::endl << std::endl;

  return aligned;
}

int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: align target.pcd source.pcd" << std::endl;
    return 0;
  }

  std::string target_pcd = argv[1];
  std::string source_pcd = argv[2];

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if(pcl::io::loadPCDFile(target_pcd, *target_cloud)) {
    std::cerr << "failed to load " << target_pcd << std::endl;
    return 0;
  }
  if(pcl::io::loadPCDFile(source_pcd, *source_cloud)) {
    std::cerr << "failed to load " << source_pcd << std::endl;
    return 0;
  }

  // downsampling
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*downsampled);
  *target_cloud = *downsampled;

  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*downsampled);
  source_cloud = downsampled;

  // NEW
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned;

  std::vector<int> num_threads = {1, omp_get_max_threads()};
  std::vector<std::pair<std::string, pclomp::NeighborSearchMethod>> search_methods = {
    // {"DIRECT7", pclomp::DIRECT7},
    {"DIRECT1", pclomp::DIRECT1}
  };

  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
  ndt_omp->setResolution(1.0);

  for(int n : num_threads) {
    for(const auto& search_method : search_methods) {
      std::cout << "--- pclomp::NDT (" << search_method.first << ", " << n << " threads) ---" << std::endl;
      ndt_omp->setNumThreads(n);
      ndt_omp->setNeighborhoodSearchMethod(search_method.second);
      aligned = align(ndt_omp, target_cloud, source_cloud);
    }
  }

  // ADDED BY JP:
  const Eigen::Matrix<float, 4, 4>& trans_matrix = ndt_omp->printFinalTransformation();
  std::cout << "Transformation Matrix:" << std::endl;
  std::cout << trans_matrix << std::endl;
  // ADDED BY JP

  // visulization: Yellow is aligned to red as color blue
  pcl::visualization::PCLVisualizer vis("vis");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0);    // red
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 255.0, 255.0, 0.0);  // yelloy
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0);        // blue
  vis.addPointCloud(target_cloud, target_handler, "target");
  vis.addPointCloud(source_cloud, source_handler, "source");
  vis.addPointCloud(aligned, aligned_handler, "aligned");
  vis.spin();

  return 0;
}
