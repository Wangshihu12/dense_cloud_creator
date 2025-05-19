/* Copyright (C) 2024 David Skuddis - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: davidskuddis@web.de, or visit: https://opensource.org/license/mit/
 */

#ifndef HELPERS_DCC_H
#define HELPERS_DCC_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>
#include "PointCloudPlus.h"

#define EPSILON_ROT 0.00001

using namespace pcl;
using namespace Eigen;



inline Eigen::MatrixXd readPosesFromFile(const std::string& directory) {
    std::ifstream infile(directory);
    if (!infile.is_open()) {
        std::cerr << "Failed to open poses.txt" << std::endl;
        return Eigen::MatrixXd();  // return an empty matrix
    }

    std::vector<std::vector<double>> data;
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double value;
        std::vector<double> row;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            data.push_back(row);
        }
    }

    if (data.empty() || data[0].empty()) {
        std::cerr << "The file is empty or improperly formatted." << std::endl;
        return Eigen::MatrixXd();  // return an empty matrix
    }

    Eigen::MatrixXd matrix(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

// Calculate slerp interpolation between two rotations defined by 3D axis-angle vectors
inline Vector3d slerp(const Vector3d &aa1, const Vector3d &aa2, double t)
{
    Quaterniond q1(AngleAxisd(aa1.norm(), aa1.normalized()));
    Quaterniond q2(AngleAxisd(aa2.norm(), aa2.normalized()));

    // Calculate slerp interpolation between q1 and q2
    Quaterniond q_interp = q1.slerp(t, q2);

    // Convert the interpolated quaternion to a 3D axis-angle vector
    AngleAxisd aa_interp(q_interp);
    Vector3d aa_interp_vec = aa_interp.axis() * aa_interp.angle();

    return aa_interp_vec;
}

inline Matrix3d skew(Vector3d vec)
{
    // calc skew symmetric
    Matrix3d skewSym;

    skewSym << 0.0, -vec(2), vec(1),
        vec(2), 0.0, -vec(0),
        -vec(1), vec(0), 0.0;

    return skewSym;
}

inline Matrix3d axang2rotm(Vector3d axang)
{
    if (axang.norm() < EPSILON_ROT)
        return Matrix3d::Identity();

    return skew(axang).exp();
}

inline Vector3d rotm2axang(Matrix3d rotm)
{

    Matrix3d skewSym = rotm.log();

    return Vector3d(skewSym(2, 1), skewSym(0, 2), skewSym(1, 0));
}

inline void closestGridDownsampling(PointCloud<PointXYZI>::Ptr rawPc, PointCloud<PointXYZI>& filteredPc, float gridSize)
{
    // 创建一个八叉树结构用于点云，设置体素大小为gridSize
    pcl::octree::OctreePointCloud<PointXYZI> octree(gridSize); // set voxel size

    // 设置八叉树的输入点云
    octree.setInputCloud(rawPc);

    // 构建八叉树结构（将点添加到八叉树中）
    octree.addPointsFromInputCloud();

    // 为过滤后的点云预分配内存，大小为八叉树的叶节点数量
    // 每个叶节点代表一个网格单元，将输出一个点
    filteredPc.resize(octree.getLeafCount());

    // 过滤后点云的点索引初始化为0
    int filId = 0;

    // 使用当前时间作为随机数生成器的种子
    // 注意：虽然有这行代码，但该函数并没有使用随机数
    srand(time(0));

    // 用于跟踪每个叶节点中最小强度值点的索引
    int minId;

    // 遍历八叉树的所有叶节点
    for (auto it = octree.leaf_depth_begin(); it != octree.leaf_depth_end(); ++it)
    {
        // 获取当前叶节点中所有点的索引
        std::vector<int> indices;
        it.getLeafContainer().getPointIndices(indices);

        // 初始化最小强度值点的索引为第一个点
        minId = 0;

        // 遍历叶节点中的所有点，寻找具有最小强度值的点
        for (int k = 0; k < indices.size(); ++k)
        {
            // 如果当前点的强度值小于已找到的最小强度值点，更新最小索引
            if (rawPc->points[indices[k]].intensity < rawPc->points[indices[minId]].intensity) minId = k;
        }

        // 将具有最小强度值的点添加到过滤后的点云中
        // 这样每个网格单元（体素）只保留一个点，从而实现下采样
        filteredPc.points[filId] = rawPc->points[indices[minId]];

        // 更新过滤后点云的索引，准备处理下一个体素
        ++filId;
    }
}

#endif
