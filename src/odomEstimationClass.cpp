// Author of FLOAM: Wang Han 
// Email wh200720041@gmail.com
// Homepage https://wanghan.pro

#include "odomEstimationClass.h"
#include "orbextractor.h"
#include <iostream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>

using namespace std;
using namespace cv;
int frame_count = 0;
int frame_start_repro = 10;

//设置特征点提取需要的一些参数
int nFeatures = 500;//图像金字塔上特征点的数量
int nLevels = 8;//图像金字塔层数
float fScaleFactor = 1.2;//金字塔比例因子
int fIniThFAST = 20; //检测fast角点阈值
int fMinThFAST = 8; //最低阈值


void OdomEstimationClass::init(lidar::Lidar lidar_param, double map_resolution){
    //init local map
    laserCloudCornerMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    laserCloudSurfMap = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());

    //downsampling size
    downSizeFilterEdge.setLeafSize(map_resolution, map_resolution, map_resolution);
    downSizeFilterSurf.setLeafSize(map_resolution * 2, map_resolution * 2, map_resolution * 2);

    //kd-tree
    kdtreeEdgeMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZI>());
    kdtreeSurfMap = pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr(new pcl::KdTreeFLANN<pcl::PointXYZI>());

    odom = Eigen::Isometry3d::Identity();
    last_odom = Eigen::Isometry3d::Identity();
    total = Eigen::Isometry3d::Identity();
    odom_inter = Eigen::Isometry3d::Identity();
    optimization_count=2;
}

void OdomEstimationClass::initMapWithPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in){
    *laserCloudCornerMap += *edge_in;
    *laserCloudSurfMap += *surf_in;
    optimization_count=12;
}

double number = 0;
void OdomEstimationClass::updatePointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_in
                                          , const sensor_msgs::ImageConstPtr& image_in, const int sequence_number){

    if(optimization_count>2)
        optimization_count--;

    Eigen::Isometry3d odom_prediction = odom * (last_odom.inverse() * odom);
    last_odom = odom;
    odom = odom_prediction;

    q_w_curr = Eigen::Quaterniond(odom.rotation());
    t_w_curr = odom.translation();

    Eigen::Quaterniond q_w_change;
    Eigen::Vector3d t_w_change;

    q_w_change.x() = -q_w_curr.y();
    q_w_change.y() = -q_w_curr.z();
    q_w_change.z() = q_w_curr.x();
    q_w_change.w() = q_w_curr.w();

    t_w_change.x() = -t_w_curr.y();
    t_w_change.y() = -t_w_curr.z();
    t_w_change.z() =  t_w_curr.x();

    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledEdgeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampledSurfCloud(new pcl::PointCloud<pcl::PointXYZI>());
    downSamplingToMap(edge_in,downsampledEdgeCloud,surf_in,downsampledSurfCloud);
    //ROS_WARN("point nyum%d,%d",(int)downsampledEdgeCloud->points.size(), (int)downsampledSurfCloud->points.size());
    if(laserCloudCornerMap->points.size()>10 && laserCloudSurfMap->points.size()>50){
        kdtreeEdgeMap->setInputCloud(laserCloudCornerMap);
        kdtreeSurfMap->setInputCloud(laserCloudSurfMap);

        for (int iterCount = 0; iterCount < optimization_count; iterCount++){
            ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
            ceres::Problem::Options problem_options;
            ceres::Problem problem(problem_options);

            problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());

            addEdgeCostFactor(downsampledEdgeCloud,laserCloudCornerMap,problem,loss_function);
            addSurfCostFactor(downsampledSurfCloud,laserCloudSurfMap,problem,loss_function);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.max_num_iterations = 4;
            options.minimizer_progress_to_stdout = false;
            options.check_gradients = false;
            options.gradient_check_relative_precision = 1e-4;
            ceres::Solver::Summary summary;

            ceres::Solve(options, &problem, &summary);

        }
    }else{
        printf("not enough points in map to associate, map error");
    }
    odom = Eigen::Isometry3d::Identity();
    odom.linear() = q_w_curr.toRotationMatrix();
    odom.translation() = t_w_curr;

    addPointsToMap(downsampledEdgeCloud,downsampledSurfCloud);
}

void OdomEstimationClass::pointAssociateToMap(pcl::PointXYZI const *const pi, pcl::PointXYZI *const po)
{
    Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
    Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
    po->x = point_w.x();
    po->y = point_w.y();
    po->z = point_w.z();
    po->intensity = pi->intensity;
    //po->intensity = 1.0;
}

void OdomEstimationClass::downSamplingToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& edge_pc_out, const pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_in, pcl::PointCloud<pcl::PointXYZI>::Ptr& surf_pc_out){
    downSizeFilterEdge.setInputCloud(edge_pc_in);
    downSizeFilterEdge.filter(*edge_pc_out);
    downSizeFilterSurf.setInputCloud(surf_pc_in);
    downSizeFilterSurf.filter(*surf_pc_out);    
}

void OdomEstimationClass::addEdgeCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function) {
    int corner_num ;
    int error_num = 0;//錯誤的線特徵有幾個
    for (int i = 0; i < (int)pc_in->points.size(); i++) {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);

        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        kdtreeEdgeMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);
        if (pointSearchSqDis[4] < 1.0) {
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0);
            for (int j = 0; j < 5; j++) {
                Eigen::Vector3d tmp(map_in->points[pointSearchInd[j]].x,
                                    map_in->points[pointSearchInd[j]].y,
                                    map_in->points[pointSearchInd[j]].z);
                center = center + tmp;
                nearCorners.push_back(tmp);
        }
            center = center / 5.0;

            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++) {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            Eigen::Vector3d point_a, point_b;

            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) {
                Eigen::Vector3d point_on_line = center;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;

                ceres::CostFunction *cost_function = new EdgeAnalyticCostFunction(curr_point, point_a, point_b);
                problem.AddResidualBlock(cost_function, loss_function, parameters);
                corner_num++;

                // 使用 Eigen::JacobiSVD 來計算最小平方的直線方程
                Eigen::MatrixXd A(nearCorners.size(), 2);
                Eigen::VectorXd b(nearCorners.size());
                for (double j = 0; j < nearCorners.size(); ++j) {
                    A(j, 0) = nearCorners[j].x();
                    A(j, 1) = 1;
                    b(j) = nearCorners[j].y();
                }
                Eigen::VectorXd x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

                // 計算 curr_point 到直線的距離
                double distance_fitted_line = std::abs(x(0) * curr_point.x() - curr_point.y() + x(1)) / std::sqrt(x(0) * x(0) + 1);

                // Calculate the distance from curr_point to the estimated line
                Eigen::Vector3d line_vector = point_b - point_a;
                double distance_estimated_line = (line_vector.cross(curr_point - point_a)).norm() / line_vector.norm();

                // Output the difference between the distances
                double distance_difference = distance_fitted_line - distance_estimated_line;
                // 檢查距離差異是否超過1並增加 error_num
                if (std::abs(distance_difference) > 1) {
                    error_num++;
                }
            }
        }
    }
    // std::cout << "corner = " << corner_num << std::endl;
    std::cout << "線特徵錯誤數量 = " << error_num << std::endl;
}



void OdomEstimationClass::addSurfCostFactor(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pc_in, const pcl::PointCloud<pcl::PointXYZI>::Ptr& map_in, ceres::Problem& problem, ceres::LossFunction *loss_function){
    int surf_num=0;
    int error_num = 0;
    //std::cout << "surf size = " << (int)pc_in->points.size() << std::endl;
    for (int i = 0; i < (int)pc_in->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&(pc_in->points[i]), &point_temp);
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;
        kdtreeSurfMap->nearestKSearch(point_temp, 5, pointSearchInd, pointSearchSqDis);

        Eigen::Matrix<double, 5, 3> matA0;
        Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones(); //5*1 都是-1
        if (pointSearchSqDis[4] < 1.0)
        {
            
            for (int j = 0; j < 5; j++)
            {
                matA0(j, 0) = map_in->points[pointSearchInd[j]].x;
                matA0(j, 1) = map_in->points[pointSearchInd[j]].y;
                matA0(j, 2) = map_in->points[pointSearchInd[j]].z;
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);
            double negative_OA_dot_norm = 1 / norm.norm();
            norm.normalize();

            bool planeValid = true;
            for (int j = 0; j < 5; j++)
            {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * map_in->points[pointSearchInd[j]].x +
                         norm(1) * map_in->points[pointSearchInd[j]].y +
                         norm(2) * map_in->points[pointSearchInd[j]].z + negative_OA_dot_norm) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }
            Eigen::Vector3d curr_point(pc_in->points[i].x, pc_in->points[i].y, pc_in->points[i].z);
            if (planeValid)
            {
                ceres::CostFunction *cost_function = new SurfNormAnalyticCostFunction(curr_point, norm, negative_OA_dot_norm);    
                problem.AddResidualBlock(cost_function, loss_function, parameters);
                            //verify by PCA
                // 1. 計算點的均值
                Eigen::Vector3d mean = matA0.colwise().mean();

                // 2. 中心化數據
                Eigen::MatrixXd centered = matA0.rowwise() - mean.transpose();

                // 3. 計算協方差矩陣
                Eigen::MatrixXd cov = centered.adjoint() * centered / double(matA0.rows() - 1);

                // 4. 進行特徵分解
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(cov);
                //Eigen::Vector3d eigenValues = eigenSolver.eigenvalues();
                Eigen::MatrixXd eigenVectors = eigenSolver.eigenvectors();

                // 5. 提取法向量 (最小特徵值對應的特徵向量)
                Eigen::Vector3d normal = eigenVectors.col(0);
                    surf_num++;
                                    // 正規化
                normal.normalize();  
                //比對誤差
                Eigen::Vector3d diff = norm - normal;
                if(diff.x() >= 1||diff.y() >= 1||diff.z() >= 1){
                    error_num = error_num + 1 ;
                }
            }
        }

    }
    // std::cout << "surf = " << surf_num << std::endl;
    std::cout << "平面錯誤特徵數量 = " << error_num << std::endl;
}

void OdomEstimationClass::addPointsToMap(const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledEdgeCloud, const pcl::PointCloud<pcl::PointXYZI>::Ptr& downsampledSurfCloud){

    for (int i = 0; i < (int)downsampledEdgeCloud->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledEdgeCloud->points[i], &point_temp);
        laserCloudCornerMap->push_back(point_temp); 
    }
    
    for (int i = 0; i < (int)downsampledSurfCloud->points.size(); i++)
    {
        pcl::PointXYZI point_temp;
        pointAssociateToMap(&downsampledSurfCloud->points[i], &point_temp);
        laserCloudSurfMap->push_back(point_temp);
    }
    
    double x_min = +odom.translation().x()-100;
    double y_min = +odom.translation().y()-100;
    double z_min = +odom.translation().z()-100;
    double x_max = +odom.translation().x()+100;
    double y_max = +odom.translation().y()+100;
    double z_max = +odom.translation().z()+100;
    
    //ROS_INFO("size : %f,%f,%f,%f,%f,%f", x_min, y_min, z_min,x_max, y_max, z_max);
    cropBoxFilter.setMin(Eigen::Vector4f(x_min, y_min, z_min, 1.0));
    cropBoxFilter.setMax(Eigen::Vector4f(x_max, y_max, z_max, 1.0));
    cropBoxFilter.setNegative(false);    

    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpCorner(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpSurf(new pcl::PointCloud<pcl::PointXYZI>());
    cropBoxFilter.setInputCloud(laserCloudSurfMap);
    cropBoxFilter.filter(*tmpSurf);
    cropBoxFilter.setInputCloud(laserCloudCornerMap);
    cropBoxFilter.filter(*tmpCorner);

    downSizeFilterSurf.setInputCloud(tmpSurf);
    downSizeFilterSurf.filter(*laserCloudSurfMap);
    downSizeFilterEdge.setInputCloud(tmpCorner);
    downSizeFilterEdge.filter(*laserCloudCornerMap);

}

void OdomEstimationClass::getMap(pcl::PointCloud<pcl::PointXYZI>::Ptr& laserCloudMap){
    
    *laserCloudMap += *laserCloudSurfMap;
    *laserCloudMap += *laserCloudCornerMap;
}

OdomEstimationClass::OdomEstimationClass(){

}
