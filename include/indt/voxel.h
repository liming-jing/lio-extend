#ifndef INDT_VOXEL_H_
#define INDT_VOXEL_H_

#include <vector>
#include "common_lib.h"

namespace faster_lio {
class Voxel {
 public:
  Voxel() : points_num_(0) {
    mean_ = common::V3D::Zero();
    cov_ = common::M3D::Zero();
    icov_ = common::M3D::Zero();
    voxel_cloud_ptr_.reset(new PointCloudType());
  }

  /**
   * @brief 增量式求解均值和方差
   *
   * @param point
   */
  void PushData(PointType point) {
    points_num_++;
    common::V3D new_data(point.x, point.y, point.z);
    common::V3D mean_differential = (new_data - mean_) / points_num_;
    common::V3D new_mean = mean_ + mean_differential;
    mean_ = new_mean;
    cov_ += new_data * new_data.transpose();
    voxel_cloud_ptr_->points.emplace_back(point);
  }

  bool Validate() {
    if (points_num_ < 5) {
      return false;  // 只有当voxel的点数超过5个时，才开始计算均值和协方差
    }

    common::V3D pt_sum = points_num_ * mean_;
    common::M3D icov = common::M3D::Zero();
    common::M3D cov = cov_;
    if (pt_sum, cov, icov, points_num_) {
      cov_ = cov;
      icov_ = icov;
      flg_estimated_ = true;
      return true;
    }
    flg_estimated_ = false;
    return false;
  }
  inline bool GetFlgEstimated() const { return flg_estimated_; }
  inline common::V3D GetMean() const { return mean_; }
  inline common::M3D GetCov() const { return cov_; }
  inline common::M3D GetICov() const { return icov_; }

 private:
  bool Core(const common::V3D& pt_sum, common::M3D& cov, common::M3D& icov,
            std::size_t data_num) {
    Eigen::SelfAdjointEigenSolver<common::M3D> eigensolver;
    common::M3D eigen_val;
    cov = (cov - 2 * pt_sum * mean_) / data_num + mean_ * mean_.transpose();
    cov *= (data_num - 1.0) / data_num;

    eigensolver.compute(cov);
    eigen_val = eigensolver.eigenvalues().asDiagonal();
    auto evecs = eigensolver.eigenvectors();

    if (eigen_val(0, 0) < 0 || eigen_val(1, 1) < 0 || eigen_val(2, 2) <= 0) {
      return false;
    }

    double min_covar_eigvalue = 0.01 * eigen_val(2, 2);
    if (eigen_val(0, 0) < min_covar_eigvalue) {
      eigen_val(0, 0) = min_covar_eigvalue;
      if (eigen_val(1, 1) < min_covar_eigvalue) {
        eigen_val(1, 1) = min_covar_eigvalue;
      }
      cov = evecs * eigen_val * evecs.inverse();
    }
    icov = cov.inverse();

    if (icov.maxCoeff() == std::numeric_limits<float>::infinity() ||
        icov.minCoeff() == -std::numeric_limits<float>::infinity()) {
      return false;
    }
    return true;
  }

 private:
  std::size_t points_num_ = 0;
  common::V3D mean_;
  common::M3D cov_;
  common::M3D icov_;
  CloudPtr voxel_cloud_ptr_;
  bool flg_estimated_ = false;  // 是否计算完成该voxel的均值和协方差：
                                // true计算成功， false计算失败
};
}  // namespace faster_lio

#endif