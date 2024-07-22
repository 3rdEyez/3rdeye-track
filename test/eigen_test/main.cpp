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

    return 0;
}