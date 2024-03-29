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

// IF EIGEN issues::
  // sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
  // sudo ln -s /usr/include/eigen3/unsupported/Eigen /usr/include/unsupported/Eigen


// ----------------------
// FUNCTION: align point clouds and measure processing time
// ----------------------

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

// ----------------------
// FUNCTION: Calculate TF
// ----------------------
using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

const Eigen::Matrix<float, 4, 4>& calculate_tf(const sensor_msgs::msg::PointCloud2& target_cloud_,
                                                const sensor_msgs::msg::PointCloud2& source_cloud_) {
    
    // Convert both to pcl::PointCloud<pcl::PointXYZ>
    pcl::PCLPointCloud2 pcl_pc2_target, pcl_pc2_source;
    pcl_conversions::toPCL(target_cloud_, pcl_pc2_target);
    pcl_conversions::toPCL(source_cloud_, pcl_pc2_source);

    PointCloudXYZ::Ptr target_cloud(new PointCloudXYZ());
    PointCloudXYZ::Ptr source_cloud(new PointCloudXYZ());

    pcl::fromPCLPointCloud2(pcl_pc2_target, *target_cloud);
    pcl::fromPCLPointCloud2(pcl_pc2_source, *source_cloud);

    // downsampling
    PointCloudXYZ::Ptr downsampled(new PointCloudXYZ());

    pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
    voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

    voxelgrid.setInputCloud(target_cloud);
    voxelgrid.filter(*downsampled);
    *target_cloud = *downsampled;

    voxelgrid.setInputCloud(source_cloud);
    voxelgrid.filter(*downsampled);
    *source_cloud = *downsampled;

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

    // GET TRANS_MATRIX AND PRINT
    const Eigen::Matrix<float, 4, 4>& trans_matrix = ndt_omp->printFinalTransformation();
    std::cout << "Transformation Matrix:" << std::endl;
    std::cout << trans_matrix << std::endl;

    // visulization: Yellow is aligned to red as color blue
    pcl::visualization::PCLVisualizer vis("vis");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0);    // red
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 255.0, 255.0, 0.0);  // yelloy
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0);        // blue
    vis.addPointCloud(target_cloud, target_handler, "target");
    vis.addPointCloud(source_cloud, source_handler, "source");
    vis.addPointCloud(aligned, aligned_handler, "aligned");
    vis.spin();

    return trans_matrix;
}

// ----------------------
// FUNCTION: Transform sensor_msgs::msg::PointCloud2
// ----------------------
sensor_msgs::PointCloud2 transformPointCloud(const sensor_msgs::PointCloud2& input_cloud, const Eigen::Matrix<float, 4, 4>& tf) {
    sensor_msgs::PointCloud2 output_cloud = input_cloud;

    // Extracting row and column dimensions from the input cloud
    int num_points = input_cloud.width * input_cloud.height;
    int point_step = input_cloud.point_step;

    // Ensure that the data size matches the expected format
    if (input_cloud.fields.size() < 3 ||
        input_cloud.fields[0].datatype != sensor_msgs::PointField::FLOAT32 ||
        input_cloud.fields[1].datatype != sensor_msgs::PointField::FLOAT32 ||
        input_cloud.fields[2].datatype != sensor_msgs::PointField::FLOAT32 ||
        point_step < 3 * sizeof(float)) {
        // Throw an error or handle the invalid input format as necessary
        return output_cloud;
    }

    // Accessing the raw data buffer
    const uint8_t* data_ptr = input_cloud.data.data();

    // Iterate through each point and apply the transformation
    for (int i = 0; i < num_points; ++i) {
        // Extracting the XYZ coordinates of the point
        float x = *reinterpret_cast<const float*>(data_ptr + input_cloud.fields[0].offset);
        float y = *reinterpret_cast<const float*>(data_ptr + input_cloud.fields[1].offset);
        float z = *reinterpret_cast<const float*>(data_ptr + input_cloud.fields[2].offset);

        // Apply the transformation
        Eigen::Vector4f point(x, y, z, 1.0);
        point = tf * point;

        // Update the point coordinates
        *reinterpret_cast<float*>(data_ptr + input_cloud.fields[0].offset) = point.x();
        *reinterpret_cast<float*>(data_ptr + input_cloud.fields[1].offset) = point.y();
        *reinterpret_cast<float*>(data_ptr + input_cloud.fields[2].offset) = point.z();

        // Move to the next point in the buffer
        data_ptr += point_step;
    }

    return output_cloud;
}


class PointCloudAligner : public rclcpp::Node
{
public:
    PointCloudAligner() : Node("pcl_concat")
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
        
        subscription_front_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/input_cloud_3",
            10,
            std::bind(&PointCloudAligner::frontCallback, this, std::placeholders::_1));

        // Publish the combined point cloud
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/output_cloud", 10);
    }



private:
    void leftCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        left_cloud_ = *msg;
        AlignAndPublish();
    }

    void rightCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        right_cloud_ = *msg;
    }

    void frontCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        front_cloud_ = *msg;
    }


    void AlignAndPublish()
    {
        if (!left_cloud_.data.empty() && !right_cloud_.data.empty() && !front_cloud_.data.empty())
        {   
            // Check if pointclouds are recorded close to each other:
            long int left_timestamp = left_cloud_.header.stamp.nanosec;
            long int right_timestamp = right_cloud_.header.stamp.nanosec;
            long int front_timestamp = front_cloud_.header.stamp.nanosec;
            long double max_difference = std::max({std::abs(left_timestamp - right_timestamp) * 1e-9,
                                                   std::abs(left_timestamp - front_timestamp) * 1e-9, 
                                                   std::abs(right_timestamp - front_timestamp) * 1e-9});
            std::cout << "Time difference:" << max_difference << " s" << std::endl;

            if (max_difference < 0.0015) // seconds
            {
            // 1) Concat helios L + helios R
                sensor_msgs::msg::PointCloud2 combined_cloud;
                if (pcl::concatenatePointCloud(left_cloud_, right_cloud_, combined_cloud)) {
                    combined_cloud.header = left_cloud_.header;
                    
                    }
                
                else {
                    std::cerr << "Error concatenating point clouds." << std::endl;
                }

            // 2) Calculate TF
                // Source = M1P , Target = heliosL+R
                const Eigen::Matrix<float, 4, 4>& tf = calculate_tf(combined_cloud, front_cloud_);

            // 3) Transform M1P 
            

            // 4) Concat HeliosL+R + M1P
                
                // publisher_->publish(combined_cloud); 
            }
        }
    }

    sensor_msgs::msg::PointCloud2 left_cloud_;
    sensor_msgs::msg::PointCloud2 right_cloud_;
    sensor_msgs::msg::PointCloud2 front_cloud_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_L_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_R_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_front_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudAligner>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

