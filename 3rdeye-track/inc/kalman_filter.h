#ifndef __KALMAN_FILTER_H__
#define __KALMAN_FILTER_H__

#include "Eigen/Dense"
#include <vector>

class KalmanFilter
{
public:
    KalmanFilter(
        int x_dim,
        Eigen::VectorXf x,
        float sigma_q, 
        float sigma_r, 
        Eigen::MatrixXf A, 
        Eigen::MatrixXf B,
        Eigen::MatrixXf H
    );
    ~KalmanFilter();
    Eigen::VectorXf predict(Eigen::VectorXf z);

// private:
    Eigen::VectorXf x_posterior;
    Eigen::MatrixXf Q, R, P;
    Eigen::MatrixXf A, B, H;
    Eigen::MatrixXf K;
};

#endif
