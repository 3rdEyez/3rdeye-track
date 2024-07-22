#ifndef __KALMAN_FILTER_H__
#define __KALMAN_FILTER_H__

#include "Eigen/Dense"
#include <vector>

template <int x_dim>
class KalmanFilter
{
public:
    KalmanFilter(
        Eigen::Matrix<float, x_dim, 1> x,
        float sigma_q, 
        float sigma_r, 
        Eigen::Matrix<float, x_dim, x_dim> A, 
        Eigen::Matrix<float, x_dim, x_dim> B, 
        Eigen::Matrix<float, x_dim, x_dim> H
    );
    ~KalmanFilter();
    Eigen::Matrix<float, x_dim, 1> predict(Eigen::Matrix<float, x_dim, 1> z);

private:
    Eigen::Matrix<float, x_dim, 1> x_prior, x_posterior;
    Eigen::Matrix<float, x_dim, x_dim> Q, R;
};

#endif
