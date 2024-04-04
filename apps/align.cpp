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
#include <rclcpp/qos.hpp>

// For the synchronizing part
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

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

//   rclcpp::Clock system_clock;

//   /auto t1 = ros::WallTime::now();
//   auto t1 = system_clock.now();
  registration->align(*aligned);
//   auto t2 = system_clock.now();
//   std::cout << "single : " << (t2 - t1).seconds()* 1000 << "[msec]" << std::endl;


  for(int i=0; i<10; i++) {
    registration->align(*aligned);
  }
//   auto t3 = system_clock.now();
//   std::cout << "10times: " << (t3 - t2).seconds() * 1000 << "[msec]" << std::endl;
//   std::cout << "fitness: " << registration->getFitnessScore() << std::endl << std::endl;

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
        // {"DIRECT26", pclomp::DIRECT26},
        {"DIRECT7", pclomp::DIRECT7},
        // {"DIRECT1", pclomp::DIRECT1},
    };

    pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>::Ptr ndt_omp(new pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>());
    ndt_omp->setResolution(1.0);

    for(int n : num_threads) {
        for(const auto& search_method : search_methods) {
            // std::cout << "--- pclomp::NDT (" << search_method.first << ", " << n << " threads) ---" << std::endl;
            ndt_omp->setNumThreads(n);
            ndt_omp->setNeighborhoodSearchMethod(search_method.second);
            aligned = align(ndt_omp, target_cloud, source_cloud);
        }
    }

    // GET TRANS_MATRIX AND PRINT
    const Eigen::Matrix<float, 4, 4>& trans_matrix = ndt_omp->printFinalTransformation();
    // std::cout << "Transformation Matrix:" << std::endl;
    // std::cout << trans_matrix << std::endl;

    // visulization: Yellow is aligned to red as color blue
    // pcl::visualization::PCLVisualizer vis("vis");
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0);    // red
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 255.0, 255.0, 0.0);  // yelloy
    // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0);        // blue
    // vis.addPointCloud(target_cloud, target_handler, "target");
    // vis.addPointCloud(source_cloud, source_handler, "source");
    // vis.addPointCloud(aligned, aligned_handler, "aligned");
    // vis.spin();

    return trans_matrix;
}

// ----------------------
// FUNCTION: Transform sensor_msgs::msg::PointCloud2
// ----------------------
sensor_msgs::msg::PointCloud2 transformPointCloud(const sensor_msgs::msg::PointCloud2& input_cloud, const Eigen::Matrix<float, 4, 4>& tf) {
    sensor_msgs::msg::PointCloud2 output_cloud = input_cloud;

    // Extracting row and column dimensions from the input cloud
    int num_points = input_cloud.width * input_cloud.height;
    int point_step = input_cloud.point_step;

    // Ensure that the data size matches the expected format
    if (input_cloud.fields.size() < 3 ||
        input_cloud.fields[0].datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        input_cloud.fields[1].datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        input_cloud.fields[2].datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        point_step < 3 * sizeof(float)) {
        // Throw an error or handle the invalid input format as necessary
        return output_cloud;
    }

    // Accessing the raw data buffer
    uint8_t* data_ptr = output_cloud.data.data();

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


class PointCloudAligner : public rclcpp::Node {
public:
    PointCloudAligner() : Node("lidar_sync_and_align") {
        
        // PUBLISHER: 
        rclcpp::QoS qos(10);
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/rslidar/combined", // topic_name
            qos
            );

        // SUBSCRIBERS
        auto rmw_qos_profile = qos.get_rmw_qos_profile();
        subscription_L_.subscribe(this, "/rslidar/helios_L",rmw_qos_profile);
        subscription_R_.subscribe(this, "/rslidar/helios_R", rmw_qos_profile);
        subscription_front_.subscribe(this, "/rslidar/M1P", rmw_qos_profile);

        // Initiate approximatetime syncer and run its callback
        sync_ = std::make_shared<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, 
                sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>>> (10, subscription_L_, subscription_R_, subscription_front_);
        sync_->registerCallback(&PointCloudAligner::pointcloudCallback, this);
    }

private:

    void pointcloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr& pc_msg_L,
                            const sensor_msgs::msg::PointCloud2::SharedPtr& pc_msg_R,
                            const sensor_msgs::msg::PointCloud2::SharedPtr& pc_msg_M1P) {
        // Debug
        // std::cout<<"Hello messages are being received, syncing them" << std::endl;

        left_cloud_ = *pc_msg_L;
        right_cloud_ = *pc_msg_R;
        front_cloud_ = *pc_msg_M1P;

        AlignAndPublish();
    }

    void AlignAndPublish()
    {
        if (!left_cloud_.data.empty() && !right_cloud_.data.empty() && !front_cloud_.data.empty())
        {   
            // Check if pointclouds are recorded close to each other:
            long int left_timestamp = left_cloud_.header.stamp.nanosec + left_cloud_.header.stamp.sec * 1e9;
            long int right_timestamp = right_cloud_.header.stamp.nanosec + right_cloud_.header.stamp.sec * 1e9;
            long int front_timestamp = front_cloud_.header.stamp.nanosec + front_cloud_.header.stamp.sec * 1e9;
            long double max_difference = std::max({std::abs(left_timestamp - right_timestamp) * 1e-9,
                                                   std::abs(left_timestamp - front_timestamp) * 1e-9, 
                                                   std::abs(right_timestamp - front_timestamp) * 1e-9});

            // std::cout << "TS left: " << left_timestamp << " && TS Right: " << right_timestamp << " && TS Front: " << right_timestamp << std::endl;
            // std::cout << "Max time difference:" << max_difference << " s" << std::endl;

            // if (max_difference < 0.0015) // seconds
            // TODO: IF DIFF between HELIOS_L and R < 0.0015 and between M1P and heliosses < 0.1:
            if (max_difference < 0.0015) {

                // 1) Concat helios L + helios R
                    sensor_msgs::msg::PointCloud2 combined_cloud_back;
                    if (pcl::concatenatePointCloud(left_cloud_, right_cloud_, combined_cloud_back)) {
                        combined_cloud_back.header = left_cloud_.header;
                        }
                    else {std::cerr << "Error concatenating point clouds." << std::endl;}

                // 2) Calculate TF
                    const Eigen::Matrix<float, 4, 4>& tf = calculate_tf(combined_cloud_back, front_cloud_);

                // 3) Transform M1P 
                    sensor_msgs::msg::PointCloud2 front_cloud = transformPointCloud(front_cloud_, tf);

                // 4) Concat HeliosL+R + M1P + PUBLISH
                    sensor_msgs::msg::PointCloud2 combined_cloud_all;
                    if (pcl::concatenatePointCloud(front_cloud, combined_cloud_back, combined_cloud_all)) {
                        combined_cloud_all.header.stamp = front_cloud_.header.stamp;
                        combined_cloud_all.header.frame_id = "rs_all";
                        publisher_->publish(combined_cloud_all); 
                        }
                    else {std::cerr << "Error concatenating point clouds." << std::endl;}
            }
        }
    }

    // Defining variables
    sensor_msgs::msg::PointCloud2 left_cloud_;
    sensor_msgs::msg::PointCloud2 right_cloud_;
    sensor_msgs::msg::PointCloud2 front_cloud_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> subscription_L_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> subscription_R_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> subscription_front_;

    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, 
                sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>>> sync_;

    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_L_;
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_R_;
    // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_front_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    // std::shared_ptr<TimeSynchronizer<sensor_msgs::msg::PointCloud2,
    //                                   sensor_msgs::msg::PointCloud2,
    //                                   sensor_msgs::msg::PointCloud2>> synchronizer_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudAligner>());
    rclcpp::shutdown();
    return 0;
}

