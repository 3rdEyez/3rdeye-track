#include "kalman_filter.h"
#include <cstdio>


int main(int argc, char const *argv[])
{
    Eigen::Matrix<float, 7, 1> x_initial;
    auto A = Eigen::Matrix<float, 7, 7>::Identity();
    auto B = Eigen::Matrix<float, 7, 7>::Zero();
    auto H = Eigen::Matrix<float, 7, 7>::Zero();
    KalmanFilter<7> kf(x_initial, 0.1f, 1.0f, A, B, H);
    return 0;
}
