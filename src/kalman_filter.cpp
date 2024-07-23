#include "kalman_filter.h"
#include <iostream> 

KalmanFilter::KalmanFilter(
        int x_dim,
        Eigen::VectorXf x,
        float sigma_q, 
        float sigma_r, 
        Eigen::MatrixXf A, 
        Eigen::MatrixXf B,
        Eigen::MatrixXf H
    )
{
    this->x_posterior = x;
    this->Q = Eigen::MatrixXf::Identity(x_dim, x_dim) * sigma_q;
    this->R = Eigen::MatrixXf::Identity(x_dim, x_dim) * sigma_r;
    this->A = A;
    this->B = B;
    this->H = H;
    this->P = Eigen::MatrixXf::Identity(x_dim, x_dim);
}


KalmanFilter::~KalmanFilter()
{

}

Eigen::VectorXf KalmanFilter::predict(Eigen::VectorXf z)
{
    // std::cout << "A = " << this->A << std::endl;
    // std::cout << "B = " << this->B << std::endl;
    // std::cout << "Q = " << this->Q << std::endl;
    // std::cout << "R = " << this->R << std::endl;
    auto x_prior = this->A * this->x_posterior + this->B;
    this->P = this->A * this->P * this->A.transpose() + this->Q;
    this->K = this->P * this->H.transpose() * (this->H * this->P * this->H.transpose() + this->R).inverse();
    this->x_posterior = x_prior + this->K * (z - this->H * x_prior);
    auto I = Eigen::MatrixXf::Identity(this->x_posterior.size(), this->x_posterior.size());
    this->P = (I - this->K * this->H) * this->P;
    return this->x_posterior * 1.0f;
}