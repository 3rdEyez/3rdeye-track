#include "kalman_filter.h"

template<int x_dim>
KalmanFilter<x_dim>::KalmanFilter(
        Eigen::Matrix<float, x_dim, 1> x,
        float sigma_q, 
        float sigma_r, 
        Eigen::Matrix<float, x_dim, x_dim> A, 
        Eigen::Matrix<float, x_dim, x_dim> B, 
        Eigen::Matrix<float, x_dim, x_dim> H
    )
{
    this->x_prior = x;
    this->Q = Eigen::Matrix<float, x_dim, x_dim>::Identity() * sigma_q;
    this->R = Eigen::Matrix<float, x_dim, x_dim>::Identity() * sigma_r;
}


template<int x_dim>
KalmanFilter<x_dim>::~KalmanFilter()
{
    
}