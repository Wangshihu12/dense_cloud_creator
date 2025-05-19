/* Copyright (C) 2024 David Skuddis - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: davidskuddis@web.de, or visit: https://opensource.org/license/mit/
 */
#include "dense_cloud_creator.h"
#include <ros/console.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#define foreach BOOST_FOREACH

using namespace Eigen;
using namespace pcl;

dense_cloud_creator::dense_cloud_creator()
{
    // 创建ROS节点句柄，前缀为"~"表示使用私有命名空间的参数
    ros::NodeHandle nh("~");

    // load parameters
    std::cout << "Current parameter config:\n";

    nh.getParam("lidar_topic", lidarSubTopicName);
    std::cout << "lidar_topic: " << lidarSubTopicName << std::endl;

    nh.getParam("pose_file_dir", pose_file_dir);
    std::cout << "pose_file_dir: " << pose_file_dir << std::endl;

    nh.getParam("result_dir", result_dir);
    std::cout << "result_dir: " << result_dir << std::endl;

    std::string bag_dirs;
    nh.getParam("bag_dirs", bag_dirs);
    std::cout << "bag_dirs: " << bag_dirs << std::endl;

    bagnames = splitIntoWords(bag_dirs);

    nh.getParam("grid_size", grid_size);
    std::cout << "grid_size: " << grid_size << std::endl;

    nh.getParam("max_num_points", max_num_points);
    std::cout << "max_num_points: " << max_num_points << std::endl;

    nh.getParam("min_dist", min_dist);
    std::cout << "min_dist: " << min_dist << std::endl;

    nh.getParam("max_dist", max_dist);
    std::cout << "max_dist: " << max_dist << std::endl;

    // init transform in imu frame
    imu2lidar = Matrix4f::Identity();

    Eigen::Quaternionf q;
    nh.getParam("q_x", q.x());
    nh.getParam("q_y", q.y());
    nh.getParam("q_z", q.z());
    nh.getParam("q_w", q.w());
    std::cout << "Quaternion from imu frame: " << q << std::endl;

    imu2lidar.block(0, 0, 3, 3) = q.normalized().toRotationMatrix();

    nh.getParam("t_x", imu2lidar(0, 3));
    nh.getParam("t_y", imu2lidar(1, 3));
    nh.getParam("t_z", imu2lidar(2, 3));

    std::cout << "Translation from imu frame: " << imu2lidar.block(0, 3, 3, 1) << std::endl;
    lidar2imu = imu2lidar.inverse();

    nh.getParam("sensor", sensor);
    std::cout << "sensor: " << sensor << std::endl;

    // DEBUG
    if (false)
    {
        std::string bag_dirs2 = "/home/david/Rosbags/Hilti/Additional_Seq/exp04_construction_upper_level.bag";

        bagnames = splitIntoWords(bag_dirs2);

        Eigen::Quaternionf q;

        q.x() = 0.7094397486399825;
        q.y() = -0.7047651311547696;
        q.z() = 0.001135774698382635;
        q.w() = -0.0002509459564800096;

        imu2lidar.block(0, 0, 3, 3) = q.normalized().toRotationMatrix();
        imu2lidar(0, 3) = 0.0;
        imu2lidar(1, 3) = 0.0;
        imu2lidar(2, 3) = 0.055;
        lidar2imu = imu2lidar.inverse();

        lidarSubTopicName = "/hesai/pandar";

        grid_size = 0.05;

        result_dir = "/home/david/optim";

        pose_file_dir = "/home/david/optim/Poses.txt";
    }

    pubCurrCloud = nh.advertise<sensor_msgs::PointCloud2>("/dense_cloud_creator/curr_pc", 1);

    globalPoints.points.reserve(max_num_points);
    filteredPoints.points.reserve(max_num_points);
}

dense_cloud_creator::~dense_cloud_creator()
{
}

void dense_cloud_creator::spin()
{
    // 输出用户提供的rosbag文件列表
    std::cout << "You entered the following rosbags:" << std::endl;
    for (const auto &rosbagdir : bagnames)
    {
        std::cout << rosbagdir << std::endl;
    };

    // 从文件加载姿态数据（位姿信息）
    Eigen::MatrixXd sparsePoses = readPosesFromFile(pose_file_dir);

    std::cout << "Loaded " << sparsePoses.rows() << " poses from " << pose_file_dir << std::endl;

    std::cout << "Create high resolution poses . . ." << std::endl;
    // 从稀疏姿态创建高分辨率姿态（通过插值）
    createHighResPoses(sparsePoses);

    // 创建话题列表，用于从rosbag中过滤消息
    std::vector<std::string> topics;
    // 添加LiDAR话题到列表
    topics.push_back(lidarSubTopicName);

    std::cout << "Process rosbag/s . . ." << std::endl;
    // 遍历所有rosbag文件
    for (auto rosbagDir : bagnames)
    {
        // 创建rosbag对象
        rosbag::Bag bag;

        try
        {
            // 尝试打开rosbag文件
            bag.open(rosbagDir);
        }
        catch (...)
        {
            // 如果打开失败，输出错误信息并退出函数
            std::cerr << "Rosbag directory (" << rosbagDir << ") is invalid, processing is aborted\n";
            return;
        }

        // 创建rosbag视图，只关注指定的话题（LiDAR数据）
        rosbag::View view(bag, rosbag::TopicQuery(topics));

        // 遍历视图中的所有消息（点云数据）
        for (rosbag::MessageInstance const m : view)
        {
            // 如果已达到最大点数限制，退出循环
            if (reachedMaxNumPoints)
                break;

            // 将消息实例化为PointCloud2类型
            sensor_msgs::PointCloud2::ConstPtr pc2Ptr = m.instantiate<sensor_msgs::PointCloud2>();

            // 如果实例化成功，处理点云数据
            if (pc2Ptr != nullptr)
                callbackPointCloud(pc2Ptr);
        }

        // 关闭当前rosbag文件
        bag.close();
    }

    std::cout << "Apply random grid filter last time before saving point cloud . . . " << std::endl;
    // 在保存前对全局点云进行网格下采样，减少冗余点
    closestGridDownsampling(globalPoints.makeShared(), filteredPoints, grid_size);

    // 准备保存点云到PCD文件
    std::string filename = result_dir + "/DensePointCloud.pcd";

    // 设置点云的宽度和高度属性
    filteredPoints.width = filteredPoints.points.size();
    filteredPoints.height = 1;  // 未组织的点云，高度为1

    std::cout << "Save accumulated points to " << filename << " . . ." << std::endl;
    // 保存点云到ASCII格式的PCD文件
    if (io::savePCDFileASCII(filename, filteredPoints) == -1)
    {
        PCL_ERROR("Failed to save PCD file\n");
    }

    std::cout << "Processing of rosbags/s finished . . . " << std::endl;

    // 创建ROS循环控制器，频率为1000Hz
    ros::Rate rate(1000);

    // 保持节点运行，直到ROS系统关闭
    while (ros::ok())
    {
        // 处理回调
        ros::spinOnce();
        // 按照指定频率休眠
        rate.sleep();
    }
}

void dense_cloud_creator::callbackPointCloud(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    // 创建新的增强型点云对象，用于存储处理后的点云数据
    PointCloudPlus::Ptr newPC(new PointCloudPlus);

    // 预分配内存空间，等于消息中的点数量
    newPC->resize(msg->height * msg->width);

    // 声明临时变量用于处理点云数据
    int arrayPosition;               // 用于计算点在原始数据中的位置
    uint8_t ring_tmp8;               // 8位激光雷达环形ID
    uint16_t ring_tmp;               // 16位激光雷达环形ID
    uint32_t relStampNano;           // 纳秒级相对时间戳
    double stampMsg = msg->header.stamp.toSec();  // 将ROS消息时间戳转换为秒

    // 检查消息时间戳是否在有效的高分辨率时间戳范围内，如果不在则丢弃该消息
    if (stampMsg - 0.3 < highResStamps(0))
        return;
    if (stampMsg + 0.3 > highResStamps(highResStamps.size() - 1))
        return;

    // 针对未知传感器类型的特殊处理
    if (sensor == "unknown" && lastPcMsgStamp < 0.0)
    {
        // 首次接收到消息，仅记录时间戳，不处理数据
        lastPcMsgStamp = stampMsg;
        return;
    }

    // 计算当前消息与上一消息的时间差，用于未知传感器类型的时间戳估计
    double deltaTPcs = stampMsg - lastPcMsgStamp;

    // 更新上一消息的时间戳
    lastPcMsgStamp = stampMsg;

    // 临时存储时间戳的变量
    float tmpStampFloat;
    double tmpStampDouble;

    // 遍历点云中的每个点
    for (uint k = 0; k < msg->height * msg->width; ++k)
    {
        // 计算当前点在原始数据数组中的位置
        arrayPosition = k * msg->point_step;

        // 初始化点为非静态点
        newPC->at(k).isStatic = 0;

        // 从原始数据中提取XYZ坐标
        memcpy(&newPC->at(k).x, &msg->data[arrayPosition + msg->fields[0].offset], sizeof(float));
        memcpy(&newPC->at(k).y, &msg->data[arrayPosition + msg->fields[1].offset], sizeof(float));
        memcpy(&newPC->at(k).z, &msg->data[arrayPosition + msg->fields[2].offset], sizeof(float));

        // 根据不同类型的激光雷达传感器提取时间戳和环形ID
        if (sensor == "hesai")
        {
            // 提取时间戳和环形ID
            memcpy(&tmpStampDouble, &msg->data[arrayPosition + msg->fields[4].offset], sizeof(double));
            memcpy(&ring_tmp, &msg->data[arrayPosition + msg->fields[5].offset], sizeof(uint16_t));

            // 为点赋值时间戳和环形ID
            newPC->at(k).stamp = tmpStampDouble;
            newPC->at(k).id = (int)ring_tmp;
        }
        else if (sensor == "ouster")
        {
            // Ouster激光雷达的时间戳是以纳秒为单位的相对时间
            memcpy(&relStampNano, &msg->data[arrayPosition + msg->fields[4].offset], sizeof(uint32_t));
            memcpy(&ring_tmp8, &msg->data[arrayPosition + msg->fields[6].offset], sizeof(uint8_t));

            // 将纳秒转换为秒并加上消息时间戳得到绝对时间
            tmpStampDouble = stampMsg + 1e-9 * (double)relStampNano;

            newPC->at(k).stamp = tmpStampDouble;
            newPC->at(k).id = (int)ring_tmp8;
        }
        else if (sensor == "robosense")
        {
            // RoboSense激光雷达的时间戳和环形ID提取
            memcpy(&tmpStampDouble, &msg->data[arrayPosition + msg->fields[5].offset], sizeof(double));
            memcpy(&ring_tmp, &msg->data[arrayPosition + msg->fields[4].offset], sizeof(uint16_t));

            newPC->at(k).stamp = tmpStampDouble;
            newPC->at(k).id = (int)ring_tmp;
        }
        else if (sensor == "velodyne")
        {
            // Velodyne激光雷达的时间戳是相对于消息时间戳的浮点数秒
            memcpy(&tmpStampFloat, &msg->data[arrayPosition + msg->fields[5].offset], sizeof(float));
            memcpy(&ring_tmp, &msg->data[arrayPosition + msg->fields[4].offset], sizeof(uint16_t));

            newPC->at(k).stamp = stampMsg + static_cast<double>(tmpStampFloat);
            newPC->at(k).id = (int)ring_tmp;
        }
        else if (sensor == "livoxXYZRTLT_s")
        {
            // Livox激光雷达（秒级时间戳）
            memcpy(&tmpStampDouble, &msg->data[arrayPosition + msg->fields[6].offset], sizeof(double));

            newPC->at(k).stamp = tmpStampDouble;

            // 由于Livox没有环形ID，创建人工环形ID
            newPC->at(k).id = k % 1000;
        }
        else if (sensor == "livoxXYZRTLT_ns")
        {
            // Livox激光雷达（纳秒级时间戳）
            memcpy(&tmpStampDouble, &msg->data[arrayPosition + msg->fields[6].offset], sizeof(double));

            // 将纳秒转换为秒，这是因为livox2驱动有bug
            newPC->at(k).stamp = 1e-9 * tmpStampDouble;

            // 创建人工环形ID
            newPC->at(k).id = k % 1000;
        }
        else if (sensor == "unknown")
        {
            // 对于未知传感器类型，使用启发式方法估计时间戳
            // 假设点是均匀分布的，根据消息间隔时间和点的索引计算
            newPC->at(k).stamp = stampMsg + deltaTPcs * (double)k / (double)(msg->height * msg->width);

            // 创建人工环形ID
            newPC->at(k).id = k % 1000;
        }

        // 设置点云数据的padding为1.0，用于齐次坐标变换
        newPC->at(k).data[3] = 1.0;

        // 将点从激光雷达坐标系转换到IMU坐标系
        newPC->at(k).getVector4fMap() = lidar2imu * newPC->at(k).getVector4fMap();
    }

    int tformId = startFromId;
    PointXYZI currPoint;
    float pointNorm;

    // 将点云从IMU坐标系转换到世界坐标系
    for (auto &point : newPC->points)
    {
        // 计算点到原点的距离（范数）
        pointNorm = point.getVector3fMap().norm();

        // 跳过距离不在有效范围内的点
        if (pointNorm < min_dist || pointNorm > max_dist)
        {
            // 将无效点的坐标设为零
            point.getVector3fMap().setZero();
            continue;
        }

        // 根据点的时间戳查找对应的变换矩阵
        for (tformId = startFromId; tformId < highResStamps.size(); ++tformId)
            if (point.stamp < highResStamps(tformId))
                break;

        // 将点转换到世界坐标系
        currPoint.data[3] = 1.0f;  // 设置齐次坐标
        currPoint.getVector4fMap() = highResTransforms[tformId] * point.getVector4fMap();
        currPoint.intensity = pointNorm;  // 用点的距离作为强度值
        point.getVector4fMap() = highResTransforms[tformId] * point.getVector4fMap();

        // 将变换后的点添加到全局点云中
        globalPoints.points.push_back(currPoint);

        // 检查是否达到最大点数限制
        if (static_cast<int>(globalPoints.points.size()) == max_num_points)
        {
            std::cout << "Maximum number of points reached, apply grid filter and continue . . . " << std::endl;

            // 应用网格下采样滤波器减少点数
            closestGridDownsampling(globalPoints.makeShared(), filteredPoints, grid_size);

            // 用滤波后的点云替换全局点云（不太高效，但作者暂时没有更好的方法）
            globalPoints = filteredPoints;

            // 如果滤波后点云仍达到最大点数，设置标志并返回
            if (static_cast<int>(globalPoints.points.size()) == max_num_points)
            {
                reachedMaxNumPoints = true;
                return;
            }
        }
    }

    // 更新下一帧点云处理的起始变换索引
    startFromId = tformId;

    // 增加已处理点云计数器
    ++processedPcCounter;

    // 发布当前处理的点云（可视化用途）
    sensor_msgs::PointCloud2 submapMsg;
    // 将PCL点云转换为ROS消息格式
    pcl::toROSMsg(*newPC, submapMsg);
    // 设置坐标系为"map"
    submapMsg.header.frame_id = "map";
    // 发布点云消息
    pubCurrCloud.publish(submapMsg);

    // 输出当前累积的总点数
    std::cout << "Num accumulated points: " << globalPoints.points.size() << std::endl;
}

void dense_cloud_creator::createHighResPoses(Eigen::MatrixXd &sparsePoses)
{
    // dirty hack to ensure that the interpolation works, even if the first pose is present twice in the pose file
    sparsePoses(0,0) -= 0.0001;

    double deltaTPoses = sparsePoses(sparsePoses.rows() - 1, 0) - sparsePoses(0, 0);
    int numPosesHighRes = std::round(deltaTPoses / dt) + 1;

    highResStamps = VectorXd::LinSpaced(numPosesHighRes, sparsePoses(0, 0), sparsePoses(sparsePoses.rows() - 1, 0));

    highResTransforms.resize(highResStamps.size());

    int lowerSparsePosesId = 0;

    std::vector<double> translX;
    std::vector<double> translY;
    std::vector<double> translZ;
    std::vector<double> stamps;

    int lowIdTranslInterp = 0;
    int upIdTranslInterp = 0;

    // generate pseudo values for init
    translX.push_back(0.0);
    translX.push_back(1.0);
    stamps.push_back(0.0);
    stamps.push_back(1.0);
    translX.push_back(2.0);
    stamps.push_back(2.0);

    boost::math::barycentric_rational<double> x = boost::math::barycentric_rational<double>(stamps.data(), translX.data(), stamps.size(), 2);
    boost::math::barycentric_rational<double> y = boost::math::barycentric_rational<double>(stamps.data(), translX.data(), stamps.size(), 2);
    boost::math::barycentric_rational<double> z = boost::math::barycentric_rational<double>(stamps.data(), translX.data(), stamps.size(), 2);

    Quaterniond q1, q2, q_interp;
    double dtQuat, dtCurr;

    // create high resolution poses
    for (int k = 0; k < highResTransforms.size(); ++k)
    {
        // update interpolation params
        if ((k == 0 || highResStamps(k) > sparsePoses(lowerSparsePosesId + 1, 0)) && lowerSparsePosesId < sparsePoses.rows() - 1)
        {
            if (k == 0)
                lowerSparsePosesId = 0;
            else
                ++lowerSparsePosesId;

            lowIdTranslInterp = std::max(lowerSparsePosesId - 2, 0);
            upIdTranslInterp = std::min(lowerSparsePosesId + 2, static_cast<int>(sparsePoses.rows()) - 1);

            translX.resize(0);
            translY.resize(0);
            translZ.resize(0);
            stamps.resize(0);

            for (int j = lowIdTranslInterp; j <= upIdTranslInterp; ++j)
            {
                stamps.push_back(sparsePoses(j, 0));
                translX.push_back(sparsePoses(j, 1));
                translY.push_back(sparsePoses(j, 2));
                translZ.push_back(sparsePoses(j, 3));
            }

            x = boost::math::barycentric_rational<double>(stamps.data(), translX.data(), stamps.size(), 2);
            y = boost::math::barycentric_rational<double>(stamps.data(), translY.data(), stamps.size(), 2);
            z = boost::math::barycentric_rational<double>(stamps.data(), translZ.data(), stamps.size(), 2);

            q1.x() = sparsePoses(lowerSparsePosesId, 4);
            q1.y() = sparsePoses(lowerSparsePosesId, 5);
            q1.z() = sparsePoses(lowerSparsePosesId, 6);
            q1.w() = sparsePoses(lowerSparsePosesId, 7);

            q2.x() = sparsePoses(lowerSparsePosesId + 1, 4);
            q2.y() = sparsePoses(lowerSparsePosesId + 1, 5);
            q2.z() = sparsePoses(lowerSparsePosesId + 1, 6);
            q2.w() = sparsePoses(lowerSparsePosesId + 1, 7);

            dtQuat = sparsePoses(lowerSparsePosesId + 1, 0) - sparsePoses(lowerSparsePosesId, 0);
        }

        // init transform
        highResTransforms[k].setIdentity();

        // interpolate rotation
        dtCurr = highResStamps(k) - sparsePoses(lowerSparsePosesId, 0);
        q_interp = q1.slerp(dtCurr / dtQuat, q2);

        AngleAxisd aa_interp(q_interp);
        Vector3d aa_interp_vec = aa_interp.axis() * aa_interp.angle();

        highResTransforms[k].block(0, 0, 3, 3) = axang2rotm(aa_interp_vec).cast<float>();

        // interpolate translation
        highResTransforms[k](0, 3) = static_cast<float>(x(highResStamps(k)));
        highResTransforms[k](1, 3) = static_cast<float>(y(highResStamps(k)));
        highResTransforms[k](2, 3) = static_cast<float>(z(highResStamps(k)));
    }
}
