#include <Eigen/Dense>
#include <iostream>

int main() {
    // 声明一个 3x3 的单位矩阵，数据类型为 double
    Eigen::Matrix<double, 3, 3> eye3d = Eigen::Matrix<double, 3, 3>::Identity();

    // 声明一个 4x4 的单位矩阵，数据类型为 float
    Eigen::Matrix<float, 4, 4> eye4f = Eigen::Matrix<float, 4, 4>::Identity();

    // 使用 Eigen 的 IO 流操作符输出矩阵
    std::cout << "3x3 identity matrix:\n" << eye3d << std::endl;
    std::cout << "4x4 identity matrix:\n" << eye4f << std::endl;

    // 生成一个 3x3 的随机矩阵
    auto mat = Eigen::MatrixXf::Random(3, 3);
    std::cout << "Random matrix:\n" << mat << std::endl;

    // 向量测试
    auto x = Eigen::VectorXf::Random(3);
    std::cout << "Random vector:\n" << x << std::endl;

    // 模拟kalmanFilter中的运算
    auto X = Eigen::VectorXf::Random(7);
    auto A = Eigen::MatrixXf::Random(7, 7);
    auto B = Eigen::MatrixXf::Zero(7, 1);
    auto xp = A * X + B;
    std::cout << "Kalman filter prediction:\n" << xp << std::endl;

    auto P = Eigen::MatrixXf::Random(7, 7);
    auto Q = Eigen::MatrixXf::Random(7, 7);
    auto Pp = A * P * A.transpose() + Q;
    std::cout << "Kalman filter prediction covariance:\n" << Pp << std::endl;

    return 0;
}