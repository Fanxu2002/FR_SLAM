#pragma once

#include <algorithm>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

// hat(vector -> matrix)
inline Eigen::Matrix3d Skew(const Eigen::Vector3d &vector)
{
        return Sophus::SO3d::hat(vector);
}

// Vector space to Maniflod space
inline Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &rotation_vector)
{
        return Sophus::SO3d::exp(rotation_vector).matrix();
}

// Maniflod  space to Vector space
inline Eigen::Vector3d LogSO3(const Eigen::Matrix3d &rotation_matrix)
{
        return Sophus::SO3d(rotation_matrix).log();
}

// Right Jacobian
// R_true = R_nominal * Exp(delta_theta)
inline Eigen::Matrix3d RightJacobianSO3(const Eigen::Vector3d &rotation_vector)
{
        const double theta = rotation_vector.norm();

        const Eigen::Matrix3d A = Skew(rotation_vector);

        const Eigen::Matrix3d A2 = A * A;

        // 出现了极小值的情况
        if (theta < 1.0e-8)
        {
                return Eigen::Matrix3d::Identity() - 0.5 * A + (1.0 / 6.0) * A2;
        }

        const double theta2 = theta * theta;

        const double theta3 = theta2 * theta;

        return Eigen::Matrix3d::Identity() - ((1.0 - std::cos(theta)) / theta2) * A + ((theta - std::sin(theta)) / theta3) * A2;
}

// inverse of the right Jacobian
inline Eigen::Matrix3d RightJacobianInverseSO3(const Eigen::Vector3d &rotation_vector)
{
        const double theta = rotation_vector.norm();
        const Eigen::Matrix3d A = Skew(rotation_vector);
        const Eigen::Matrix3d A2 = A * A;
        if (theta < 1.0e-8)
        {
                return Eigen::Matrix3d::Identity() + 0.5 * A + (1.0 / 12.0) * A2;
        }
        const double theta2 = theta * theta;

        const double coefficient = (1.0 / theta2) - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));

        return Eigen::Matrix3d::Identity() + 0.5 * A + coefficient * A2;
}

// safe quaternion normalization
inline Eigen::Quaterniond NormalizeQuaternion(const Eigen::Quaterniond &quaternion)
{
        Eigen::Quaterniond normalized = quaternion;
        if (!normalized.coeffs().allFinite() || normalized.norm() < 1.0e-12)
        {
                return Eigen::Quaterniond::Identity();
        }
        normalized.normalize();

        return normalized;
}

// Gravity S2 maniflod
inline Eigen::Matrix<double, 3, 2> GravityTangentBiasis(const Eigen::Vector3d &gravity_w)
{
        Eigen::Vector3d direction = gravity_w;
        if (!direction.allFinite() || direction.norm() < 1.0e-12)
        {
                direction = Eigen::Vector3d(0.0, 0.0, -1.0);
        }
        else
        {
                direction.normalize();
        }
        Eigen::Vector3d reference_axis;
        if (std::abs(direction.z()) < 0.9)
        {
                reference_axis = Eigen::Vector3d::UnitZ();
        }
        else
        {
                reference_axis = Eigen::Vector3d::UnitX();
        }
        Eigen::Vector3d basis_1 = reference_axis.cross(direction);
        if (basis_1.norm() < 1.0e-12)
        {
                reference_axis = Eigen::Vector3d::UnitY();
                basis_1 = reference_axis.cross(direction);
        }
        basis_1.normalize();
        Eigen::Vector3d basis_2 = direction.cross(basis_1);
        basis_2.normalize();
        Eigen::Matrix<double, 3, 2> basis;
        basis.col(0) = basis_1;
        basis.col(1) = basis_2;
        return basis;
}

// delta G = E_g * delta_gamma
// E_g = -|g| * [u_g]^ * B_g
// u_g = gravity_w /|gravity_w|
inline Eigen::Matrix<double, 3, 2> GravityErrorJacobian(const Eigen::Vector3d &gravity_w)
{
        const double gravity_magnitude = gravity_w.norm();
        if (!std::isfinite(gravity_magnitude) || gravity_magnitude < 1.0e-12)
        {
                return Eigen::Matrix<double, 3, 2>::Zero();
        }
        const Eigen::Vector3d gravity_direction = gravity_w / gravity_magnitude;
        const Eigen::Matrix<double, 3, 2> basis = GravityTangentBiasis(gravity_w);
        return -gravity_magnitude * Skew(gravity_direction) * basis;
}

// gravity boxplus
inline Eigen::Vector3d GravityBoxPlus(const Eigen::Vector3d &gravity_W, const Eigen::Vector2d &delta_gamma)
{
        const double gravity_magnitude = gravity_W.norm();

        if (!std::isfinite(gravity_magnitude) || gravity_magnitude < 1.0e-12)
        {
                return gravity_W;
        }
        const Eigen::Vector3d gravity_direction = gravity_W / gravity_magnitude;
        const Eigen::Matrix<double, 3, 2> basis = GravityTangentBiasis(gravity_W);
        const Eigen::Vector3d rotation_vector = basis * delta_gamma;
        Eigen::Vector3d new_direction = ExpSO3(rotation_vector) * gravity_direction;
        if (!new_direction.allFinite() || new_direction.norm() < 1.0e-12)
        {
                return gravity_W;
        }
        new_direction.normalize();
        return gravity_magnitude * new_direction;
}

// gravity boxmius
inline Eigen::Vector2d GravityBoxMinus(const Eigen::Vector3d &target_gravity, const Eigen::Vector3d &reference_gravity)
{
        if (!target_gravity.allFinite() ||
            !reference_gravity.allFinite() ||
            target_gravity.norm() < 1.0e-12 ||
            reference_gravity.norm() < 1.0e-12)
        {
                return Eigen::Vector2d::Zero();
        }
        const Eigen::Vector3d target_direction = target_gravity.normalized();
        const Eigen::Vector3d reference_direction = reference_gravity.normalized();
        Eigen::Quaterniond relative_rotation = Eigen::Quaterniond::FromTwoVectors(reference_direction, target_direction);
        relative_rotation.normalize();
        const Eigen::Vector3d rotation_vector = LogSO3(relative_rotation.toRotationMatrix());
        const Eigen::Matrix<double, 3, 2> basis = GravityTangentBiasis(reference_gravity);
        return basis.transpose() * rotation_vector;
}
