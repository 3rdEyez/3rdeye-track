#include "kalman_filter.h"
#include <iostream>


int main(int argc, char const *argv[])
{
    auto x_initial = Eigen::VectorXf(7);
    auto A = Eigen::MatrixXf::Identity(7, 7);
    auto B = Eigen::MatrixXf::Zero(7, 7);
    auto H = Eigen::MatrixXf::Identity(7, 7);
    KalmanFilter kf(7, x_initial, 0.1f, 1.0f, A, B, H);

    // print all matrixs
    std::cout << "x_initial: \n" << x_initial << std::endl;
    std::cout << "A: \n" << A << std::endl;
    std::cout << "B: \n" << B << std::endl;
    std::cout << "H: \n" << H << std::endl;
    return 0;
}
