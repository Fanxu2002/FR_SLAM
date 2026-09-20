#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

double DegToRad(const double degrees)
{
    return degrees * kPi / 180.0;
}

double RadToDeg(const double radians)
{
    return radians * 180.0 / kPi;
}

struct MotionPair
{
    double start_timestamp = 0.0;
    double end_timestamp = 0.0;

    Eigen::Matrix3d delta_R_lidar =
        Eigen::Matrix3d::Identity();

    Eigen::Vector3d delta_t_lidar =
        Eigen::Vector3d::Zero();

    Eigen::Matrix3d delta_R_imu =
        Eigen::Matrix3d::Identity();

    Eigen::Vector3d delta_t_imu =
        Eigen::Vector3d::Zero();

    bool has_translation = false;
};

struct Options
{
    std::string input_path;

    std::string output_path =
        "lidar_imu_rotation_calibration.yaml";

    double min_rotation_deg = 1.0;

    double max_angle_difference_deg = 5.0;

    double rotation_huber_delta_deg = 2.0;

    double translation_huber_delta_m = 0.05;

    double minimum_translation_observability_ratio = 0.01;

    int max_irls_iterations = 20;

    bool self_test = false;
};

struct CalibrationResult
{
    // IMU -> LiDAR
    Eigen::Matrix3d R_LI =
        Eigen::Matrix3d::Identity();

    Eigen::Vector3d P_LI =
        Eigen::Vector3d::Zero();

    // LiDAR -> IMU, directly usable by FR-SLAM.
    Eigen::Matrix3d R_IL =
        Eigen::Matrix3d::Identity();

    Eigen::Vector3d P_IL =
        Eigen::Vector3d::Zero();

    std::vector<double> rotation_residuals_deg;

    std::vector<double> translation_residuals_m;

    Eigen::Vector3d rotation_excitation_eigenvalues =
        Eigen::Vector3d::Zero();

    Eigen::Vector3d translation_singular_values =
        Eigen::Vector3d::Zero();

    std::size_t total_pair_count = 0;

    std::size_t used_pair_count = 0;

    std::size_t translation_pair_count = 0;

    std::size_t rejected_small_rotation_count = 0;

    std::size_t rejected_angle_mismatch_count = 0;

    double mean_rotation_residual_deg = 0.0;

    double median_rotation_residual_deg = 0.0;

    double max_rotation_residual_deg = 0.0;

    double rotation_second_to_first_excitation_ratio = 0.0;

    bool rotation_excitation_sufficient = false;

    bool translation_available = false;

    double translation_rmse_m = 0.0;

    double translation_median_residual_m = 0.0;

    double translation_max_residual_m = 0.0;

    double translation_observability_ratio = 0.0;

    double translation_condition_number =
        std::numeric_limits<double>::infinity();

    bool translation_sufficient = false;
};

std::string BuildTimestamp()
{
    const auto now =
        std::chrono::system_clock::now();

    const std::time_t time =
        std::chrono::system_clock::to_time_t(now);

    std::tm local_time{};

#if defined(_WIN32)
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif

    std::ostringstream stream;

    stream
        << std::put_time(
               &local_time,
               "%Y%m%d_%H%M%S");

    return stream.str();
}

std::string ResolveNonOverwritingOutputPath(
    const std::string &requested_output_path)
{
    namespace fs = std::filesystem;

    const fs::path requested_path(
        requested_output_path);

    const fs::path parent_path =
        requested_path.parent_path();

    if (!parent_path.empty())
    {
        std::error_code error;

        fs::create_directories(
            parent_path,
            error);

        if (error)
        {
            throw std::runtime_error(
                "Cannot create output directory '" +
                parent_path.string() +
                "': " +
                error.message());
        }
    }

    std::error_code exists_error;

    if (!fs::exists(
            requested_path,
            exists_error))
    {
        if (exists_error)
        {
            throw std::runtime_error(
                "Cannot inspect output path.");
        }

        return requested_path.string();
    }

    const std::string timestamp =
        BuildTimestamp();

    const std::string stem =
        requested_path.stem().string();

    const std::string extension =
        requested_path.extension().string();

    for (std::size_t suffix = 0;
         suffix < 10000U;
         ++suffix)
    {
        std::ostringstream filename;

        filename
            << stem
            << "_"
            << timestamp;

        if (suffix > 0U)
        {
            filename
                << "_"
                << suffix;
        }

        filename << extension;

        const fs::path candidate =
            parent_path /
            filename.str();

        std::error_code candidate_error;

        if (!fs::exists(
                candidate,
                candidate_error))
        {
            if (candidate_error)
            {
                throw std::runtime_error(
                    "Cannot inspect candidate output path.");
            }

            std::cout
                << "Requested output already exists; preserving it.\n"
                << "Writing new result to:\n  "
                << candidate.string()
                << "\n";

            return candidate.string();
        }
    }

    throw std::runtime_error(
        "Unable to allocate output path.");
}

Eigen::Vector3d LogSO3(
    const Eigen::Matrix3d &R)
{
    const Eigen::AngleAxisd angle_axis(R);

    if (!std::isfinite(angle_axis.angle()) ||
        angle_axis.angle() < 1.0e-12)
    {
        return Eigen::Vector3d::Zero();
    }

    return
        angle_axis.angle() *
        angle_axis.axis();
}

double RotationAngle(
    const Eigen::Matrix3d &R)
{
    return LogSO3(R).norm();
}

Eigen::Matrix3d ProjectToSO3(
    const Eigen::Matrix3d &matrix)
{
    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        matrix,
        Eigen::ComputeFullU |
            Eigen::ComputeFullV);

    Eigen::Matrix3d U =
        svd.matrixU();

    const Eigen::Matrix3d V =
        svd.matrixV();

    Eigen::Matrix3d correction =
        Eigen::Matrix3d::Identity();

    correction(2, 2) =
        (U * V.transpose()).determinant();

    return
        U *
        correction *
        V.transpose();
}

Eigen::Matrix3d SolveWeightedWahba(
    const std::vector<Eigen::Vector3d>
        &lidar_rotation_vectors,
    const std::vector<Eigen::Vector3d>
        &imu_rotation_vectors,
    const std::vector<double>
        &weights)
{
    Eigen::Matrix3d cross_covariance =
        Eigen::Matrix3d::Zero();

    for (std::size_t i = 0;
         i < weights.size();
         ++i)
    {
        cross_covariance.noalias() +=
            weights[i] *
            lidar_rotation_vectors[i] *
            imu_rotation_vectors[i].transpose();
    }

    return
        ProjectToSO3(
            cross_covariance);
}

double HuberWeight(
    const double residual,
    const double delta)
{
    if (residual <= delta ||
        residual < 1.0e-12)
    {
        return 1.0;
    }

    return
        delta /
        residual;
}

double Median(
    std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end());

    const std::size_t middle =
        values.size() / 2U;

    if ((values.size() % 2U) == 0U)
    {
        return
            0.5 *
            (
                values[middle - 1U] +
                values[middle]
            );
    }

    return values[middle];
}

bool ParseNumericCsvRow(
    const std::string &line,
    std::vector<double> &values)
{
    values.clear();

    std::stringstream stream(line);

    std::string cell;

    while (std::getline(
        stream,
        cell,
        ','))
    {
        const std::size_t first =
            cell.find_first_not_of(
                " \t\r\n");

        const std::size_t last =
            cell.find_last_not_of(
                " \t\r\n");

        if (first == std::string::npos)
        {
            return false;
        }

        const std::string trimmed =
            cell.substr(
                first,
                last - first + 1U);

        try
        {
            std::size_t parsed_count = 0;

            const double value =
                std::stod(
                    trimmed,
                    &parsed_count);

            if (parsed_count !=
                    trimmed.size() ||
                !std::isfinite(value))
            {
                return false;
            }

            values.push_back(value);
        }
        catch (const std::exception &)
        {
            return false;
        }
    }

    return true;
}

std::vector<MotionPair>
LoadMotionPairs(
    const std::string &path)
{
    std::ifstream input(path);

    if (!input.is_open())
    {
        throw std::runtime_error(
            "Cannot open input CSV: " +
            path);
    }

    std::vector<MotionPair> pairs;

    std::string line;

    std::size_t line_number = 0;

    while (std::getline(
        input,
        line))
    {
        ++line_number;

        const std::size_t first =
            line.find_first_not_of(
                " \t\r\n");

        if (first == std::string::npos ||
            line[first] == '#')
        {
            continue;
        }

        std::vector<double> values;

        if (!ParseNumericCsvRow(
                line,
                values))
        {
            // One textual header is allowed.
            if (pairs.empty())
            {
                continue;
            }

            throw std::runtime_error(
                "Invalid numeric CSV row at line " +
                std::to_string(
                    line_number));
        }

        MotionPair pair;

        // ========================================================
        // New full SE(3) format:
        //
        // start,end,
        // lidar_txyz,lidar_qxyzw,
        // imu_txyz,imu_qxyzw
        // ========================================================
        if (values.size() == 16U)
        {
            pair.start_timestamp =
                values[0];

            pair.end_timestamp =
                values[1];

            pair.delta_t_lidar =
                Eigen::Vector3d(
                    values[2],
                    values[3],
                    values[4]);

            const Eigen::Quaterniond q_lidar(
                values[8],
                values[5],
                values[6],
                values[7]);

            pair.delta_t_imu =
                Eigen::Vector3d(
                    values[9],
                    values[10],
                    values[11]);

            const Eigen::Quaterniond q_imu(
                values[15],
                values[12],
                values[13],
                values[14]);

            if (q_lidar.norm() < 1.0e-8 ||
                q_imu.norm() < 1.0e-8)
            {
                throw std::runtime_error(
                    "Near-zero quaternion at line " +
                    std::to_string(
                        line_number));
            }

            pair.delta_R_lidar =
                q_lidar.normalized()
                    .toRotationMatrix();

            pair.delta_R_imu =
                q_imu.normalized()
                    .toRotationMatrix();

            pair.has_translation = true;
        }
        // ========================================================
        // Legacy rotation-only formats.
        // Kept only for backwards-compatible diagnostics.
        // ========================================================
        else if (
            values.size() == 10U ||
            values.size() == 9U)
        {
            const std::size_t offset =
                values.size() == 10U
                    ? 2U
                    : 1U;

            pair.start_timestamp =
                values.size() == 10U
                    ? values[0]
                    : values[0];

            pair.end_timestamp =
                values.size() == 10U
                    ? values[1]
                    : values[0];

            const Eigen::Quaterniond q_lidar(
                values[offset + 3U],
                values[offset + 0U],
                values[offset + 1U],
                values[offset + 2U]);

            const Eigen::Quaterniond q_imu(
                values[offset + 7U],
                values[offset + 4U],
                values[offset + 5U],
                values[offset + 6U]);

            if (q_lidar.norm() < 1.0e-8 ||
                q_imu.norm() < 1.0e-8)
            {
                throw std::runtime_error(
                    "Near-zero quaternion at line " +
                    std::to_string(
                        line_number));
            }

            pair.delta_R_lidar =
                q_lidar.normalized()
                    .toRotationMatrix();

            pair.delta_R_imu =
                q_imu.normalized()
                    .toRotationMatrix();

            pair.has_translation = false;
        }
        else
        {
            throw std::runtime_error(
                "Expected 16-column SE(3) CSV "
                "or legacy 9/10-column rotation CSV "
                "at line " +
                std::to_string(line_number) +
                ", got " +
                std::to_string(values.size()));
        }

        pairs.push_back(pair);
    }

    if (pairs.empty())
    {
        throw std::runtime_error(
            "No motion pairs loaded from: " +
            path);
    }

    return pairs;
}

CalibrationResult CalibrateExtrinsic(
    const std::vector<MotionPair> &pairs,
    const Options &options)
{
    CalibrationResult result;

    result.total_pair_count =
        pairs.size();

    std::vector<MotionPair>
        accepted_pairs;

    std::vector<Eigen::Vector3d>
        lidar_rotation_vectors;

    std::vector<Eigen::Vector3d>
        imu_rotation_vectors;

    const double minimum_rotation =
        DegToRad(
            options.min_rotation_deg);

    const double maximum_angle_difference =
        DegToRad(
            options.max_angle_difference_deg);

    for (const MotionPair &pair :
         pairs)
    {
        const Eigen::Vector3d lidar_vector =
            LogSO3(
                pair.delta_R_lidar);

        const Eigen::Vector3d imu_vector =
            LogSO3(
                pair.delta_R_imu);

        const double lidar_angle =
            lidar_vector.norm();

        const double imu_angle =
            imu_vector.norm();

        if (std::max(
                lidar_angle,
                imu_angle) <
            minimum_rotation)
        {
            ++result
                .rejected_small_rotation_count;

            continue;
        }

        if (std::abs(
                lidar_angle -
                imu_angle) >
            maximum_angle_difference)
        {
            ++result
                .rejected_angle_mismatch_count;

            continue;
        }

        accepted_pairs.push_back(pair);

        lidar_rotation_vectors.push_back(
            lidar_vector);

        imu_rotation_vectors.push_back(
            imu_vector);
    }

    result.used_pair_count =
        accepted_pairs.size();

    if (accepted_pairs.size() < 3U)
    {
        throw std::runtime_error(
            "Fewer than three usable motion pairs.");
    }

    // ============================================================
    // 1. Rotation: AX = XB, X = T_LI.
    // ============================================================
    std::vector<double> rotation_weights(
        accepted_pairs.size(),
        1.0);

    Eigen::Matrix3d R_LI =
        SolveWeightedWahba(
            lidar_rotation_vectors,
            imu_rotation_vectors,
            rotation_weights);

    const double rotation_huber_delta =
        DegToRad(
            options.rotation_huber_delta_deg);

    for (int iteration = 0;
         iteration <
             options.max_irls_iterations;
         ++iteration)
    {
        for (std::size_t i = 0;
             i < accepted_pairs.size();
             ++i)
        {
            const double residual =
                (
                    lidar_rotation_vectors[i] -
                    R_LI *
                        imu_rotation_vectors[i]
                ).norm();

            rotation_weights[i] =
                HuberWeight(
                    residual,
                    rotation_huber_delta);
        }

        const Eigen::Matrix3d updated_R_LI =
            SolveWeightedWahba(
                lidar_rotation_vectors,
                imu_rotation_vectors,
                rotation_weights);

        const double update_angle =
            RotationAngle(
                R_LI.transpose() *
                updated_R_LI);

        R_LI =
            updated_R_LI;

        if (update_angle < 1.0e-10)
        {
            break;
        }
    }

    result.R_LI = R_LI;

    // ============================================================
    // Rotation diagnostics.
    // ============================================================
    for (const MotionPair &pair :
         accepted_pairs)
    {
        const Eigen::Matrix3d error_rotation =
            pair.delta_R_lidar *
            R_LI *
            pair.delta_R_imu.transpose() *
            R_LI.transpose();

        result.rotation_residuals_deg.push_back(
            RadToDeg(
                RotationAngle(
                    error_rotation)));
    }

    const double rotation_residual_sum =
        std::accumulate(
            result.rotation_residuals_deg.begin(),
            result.rotation_residuals_deg.end(),
            0.0);

    result.mean_rotation_residual_deg =
        rotation_residual_sum /
        static_cast<double>(
            result.rotation_residuals_deg.size());

    result.median_rotation_residual_deg =
        Median(
            result.rotation_residuals_deg);

    result.max_rotation_residual_deg =
        *std::max_element(
            result.rotation_residuals_deg.begin(),
            result.rotation_residuals_deg.end());

    Eigen::Matrix3d rotation_excitation =
        Eigen::Matrix3d::Zero();

    for (const Eigen::Vector3d &vector :
         imu_rotation_vectors)
    {
        rotation_excitation.noalias() +=
            vector *
            vector.transpose();
    }

    const Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix3d>
        rotation_eigensolver(
            rotation_excitation);

    if (rotation_eigensolver.info() !=
        Eigen::Success)
    {
        throw std::runtime_error(
            "Rotation excitation decomposition failed.");
    }

    result.rotation_excitation_eigenvalues =
        rotation_eigensolver
            .eigenvalues()
            .reverse();

    if (result
            .rotation_excitation_eigenvalues
            .x() >
        1.0e-12)
    {
        result
            .rotation_second_to_first_excitation_ratio =
            result
                .rotation_excitation_eigenvalues
                .y() /
            result
                .rotation_excitation_eigenvalues
                .x();
    }

    result.rotation_excitation_sufficient =
        result
            .rotation_second_to_first_excitation_ratio >=
        0.05;

    // ============================================================
    // 2. Translation.
    //
    // A X = X B, X = T_LI
    //
    //     (R_L - I) P_LI
    //       =
    //     R_LI t_I - t_L
    //
    // Stack all accepted motion pairs:
    //
    //     A p = b
    // ============================================================
    std::vector<const MotionPair *>
        translation_pairs;

    for (const MotionPair &pair :
         accepted_pairs)
    {
        if (pair.has_translation &&
            pair.delta_t_lidar.allFinite() &&
            pair.delta_t_imu.allFinite())
        {
            translation_pairs.push_back(
                &pair);
        }
    }

    result.translation_pair_count =
        translation_pairs.size();

    result.translation_available =
        !translation_pairs.empty();

    if (translation_pairs.size() >= 3U)
    {
        const Eigen::Index row_count =
            static_cast<Eigen::Index>(
                3U *
                translation_pairs.size());

        Eigen::MatrixXd A(
            row_count,
            3);

        Eigen::VectorXd b(
            row_count);

        for (std::size_t i = 0;
             i < translation_pairs.size();
             ++i)
        {
            const MotionPair &pair =
                *translation_pairs[i];

            A.block<3, 3>(
                static_cast<Eigen::Index>(
                    3U * i),
                0) =
                pair.delta_R_lidar -
                Eigen::Matrix3d::Identity();

            b.segment<3>(
                static_cast<Eigen::Index>(
                    3U * i)) =
                R_LI *
                    pair.delta_t_imu -
                pair.delta_t_lidar;
        }

        // Translation observability from unweighted system.
        const Eigen::JacobiSVD<
            Eigen::MatrixXd>
            observability_svd(
                A,
                Eigen::ComputeThinU |
                    Eigen::ComputeThinV);

        const Eigen::VectorXd singular_values =
            observability_svd
                .singularValues();

        if (singular_values.size() >= 3)
        {
            result.translation_singular_values =
                singular_values.head<3>();

            const double largest =
                result
                    .translation_singular_values
                    .x();

            const double smallest =
                result
                    .translation_singular_values
                    .z();

            if (largest > 1.0e-12)
            {
                result
                    .translation_observability_ratio =
                    smallest /
                    largest;
            }

            if (smallest > 1.0e-12)
            {
                result
                    .translation_condition_number =
                    largest /
                    smallest;
            }
        }

        // ========================================================
        // Robust translation IRLS.
        // Each motion pair has one scalar robust weight applied
        // to its three translation equations.
        // ========================================================
        std::vector<double>
            translation_weights(
                translation_pairs.size(),
                1.0);

        Eigen::Vector3d P_LI =
            observability_svd.solve(b);

        for (int iteration = 0;
             iteration <
                 options.max_irls_iterations;
             ++iteration)
        {
            Eigen::MatrixXd weighted_A =
                A;

            Eigen::VectorXd weighted_b =
                b;

            for (std::size_t i = 0;
                 i < translation_pairs.size();
                 ++i)
            {
                const Eigen::Vector3d residual =
                    A.block<3, 3>(
                        static_cast<Eigen::Index>(
                            3U * i),
                        0) *
                        P_LI -
                    b.segment<3>(
                        static_cast<Eigen::Index>(
                            3U * i));

                translation_weights[i] =
                    HuberWeight(
                        residual.norm(),
                        options
                            .translation_huber_delta_m);

                const double sqrt_weight =
                    std::sqrt(
                        translation_weights[i]);

                weighted_A
                    .block<3, 3>(
                        static_cast<Eigen::Index>(
                            3U * i),
                        0) *=
                    sqrt_weight;

                weighted_b
                    .segment<3>(
                        static_cast<Eigen::Index>(
                            3U * i)) *=
                    sqrt_weight;
            }

            const Eigen::JacobiSVD<
                Eigen::MatrixXd>
                weighted_svd(
                    weighted_A,
                    Eigen::ComputeThinU |
                        Eigen::ComputeThinV);

            const Eigen::Vector3d
                updated_P_LI =
                    weighted_svd.solve(
                        weighted_b);

            const double update_norm =
                (
                    updated_P_LI -
                    P_LI
                ).norm();

            P_LI =
                updated_P_LI;

            if (update_norm < 1.0e-10)
            {
                break;
            }
        }

        result.P_LI =
            P_LI;

        double squared_residual_sum =
            0.0;

        for (std::size_t i = 0;
             i < translation_pairs.size();
             ++i)
        {
            const Eigen::Vector3d residual =
                A.block<3, 3>(
                    static_cast<Eigen::Index>(
                        3U * i),
                    0) *
                    P_LI -
                b.segment<3>(
                    static_cast<Eigen::Index>(
                        3U * i));

            const double residual_norm =
                residual.norm();

            result
                .translation_residuals_m
                .push_back(
                    residual_norm);

            squared_residual_sum +=
                residual.squaredNorm();
        }

        result.translation_rmse_m =
            std::sqrt(
                squared_residual_sum /
                static_cast<double>(
                    3U *
                    translation_pairs.size()));

        result.translation_median_residual_m =
            Median(
                result.translation_residuals_m);

        result.translation_max_residual_m =
            *std::max_element(
                result
                    .translation_residuals_m
                    .begin(),
                result
                    .translation_residuals_m
                    .end());

        result.translation_sufficient =
            result
                .translation_observability_ratio >=
            options
                .minimum_translation_observability_ratio;
    }

    // ============================================================
    // 3. Convert IMU->LiDAR result to FR-SLAM LiDAR->IMU form.
    //
    //     T_IL = T_LI^{-1}
    //
    //     R_IL = R_LI^T
    //     P_IL = -R_IL * P_LI
    // ============================================================
    result.R_IL =
        result.R_LI.transpose();

    result.P_IL =
        -result.R_IL *
        result.P_LI;

    return result;
}

Eigen::Vector3d MatrixToRpyZyx(
    const Eigen::Matrix3d &R)
{
    const double pitch =
        std::asin(
            std::clamp(
                -R(2, 0),
                -1.0,
                1.0));

    const double cos_pitch =
        std::cos(pitch);

    double roll = 0.0;

    double yaw = 0.0;

    if (std::abs(cos_pitch) >
        1.0e-8)
    {
        roll =
            std::atan2(
                R(2, 1),
                R(2, 2));

        yaw =
            std::atan2(
                R(1, 0),
                R(0, 0));
    }
    else
    {
        roll =
            std::atan2(
                -R(1, 2),
                R(1, 1));
    }

    return
        Eigen::Vector3d(
            roll,
            pitch,
            yaw);
}

void WriteYaml(
    const CalibrationResult &result,
    const Options &options)
{
    std::ofstream output(
        options.output_path);

    if (!output.is_open())
    {
        throw std::runtime_error(
            "Cannot open output YAML: " +
            options.output_path);
    }

    const Eigen::Quaterniond q_LI(
        result.R_LI);

    const Eigen::Quaterniond q_IL(
        result.R_IL);

    const Eigen::Vector3d rpy_LI_deg =
        MatrixToRpyZyx(
            result.R_LI)
            .unaryExpr(
                [](const double value)
                {
                    return RadToDeg(value);
                });

    output
        << std::setprecision(12);

    // Keep the historical root key so existing FR-SLAM
    // launch loaders remain backwards compatible.
    output
        << "lidar_imu_rotation_calibration:\n";

    output
        << "  calibration_type: \"SE3\"\n";

    output
        << "  convention: "
        << "\"p_L = R_LI * p_I + P_LI\"\n";

    output
        << "  relative_motion_equation: "
        << "\"delta_T_L * T_LI = "
        << "T_LI * delta_T_I\"\n";

    output
        << "  quaternion_xyzw: ["
        << q_LI.x() << ", "
        << q_LI.y() << ", "
        << q_LI.z() << ", "
        << q_LI.w() << "]\n";

    output
        << "  translation_xyz_m: ["
        << result.P_LI.x() << ", "
        << result.P_LI.y() << ", "
        << result.P_LI.z() << "]\n";

    output
        << "  rpy_deg_zyx: ["
        << rpy_LI_deg.x() << ", "
        << rpy_LI_deg.y() << ", "
        << rpy_LI_deg.z() << "]\n";

    output
        << "  rotation_matrix:\n";

    for (int row = 0;
         row < 3;
         ++row)
    {
        output
            << "    - ["
            << result.R_LI(row, 0)
            << ", "
            << result.R_LI(row, 1)
            << ", "
            << result.R_LI(row, 2)
            << "]\n";
    }

    output
        << "  inverse_lidar_to_imu:\n";

    output
        << "    convention: "
        << "\"p_I = R_IL * p_L + P_IL\"\n";

    output
        << "    quaternion_xyzw: ["
        << q_IL.x() << ", "
        << q_IL.y() << ", "
        << q_IL.z() << ", "
        << q_IL.w() << "]\n";

    output
        << "    translation_xyz_m: ["
        << result.P_IL.x() << ", "
        << result.P_IL.y() << ", "
        << result.P_IL.z() << "]\n";

    output
        << "    rotation_matrix:\n";

    for (int row = 0;
         row < 3;
         ++row)
    {
        output
            << "      - ["
            << result.R_IL(row, 0)
            << ", "
            << result.R_IL(row, 1)
            << ", "
            << result.R_IL(row, 2)
            << "]\n";
    }

    output
        << "  diagnostics:\n";

    output
        << "    total_pairs: "
        << result.total_pair_count
        << "\n";

    output
        << "    used_pairs: "
        << result.used_pair_count
        << "\n";

    output
        << "    rejected_small_rotation: "
        << result
               .rejected_small_rotation_count
        << "\n";

    output
        << "    rejected_angle_mismatch: "
        << result
               .rejected_angle_mismatch_count
        << "\n";

    output
        << "    mean_residual_deg: "
        << result
               .mean_rotation_residual_deg
        << "\n";

    output
        << "    median_residual_deg: "
        << result
               .median_rotation_residual_deg
        << "\n";

    output
        << "    max_residual_deg: "
        << result
               .max_rotation_residual_deg
        << "\n";

    output
        << "    excitation_eigenvalues_desc: ["
        << result
               .rotation_excitation_eigenvalues
               .x()
        << ", "
        << result
               .rotation_excitation_eigenvalues
               .y()
        << ", "
        << result
               .rotation_excitation_eigenvalues
               .z()
        << "]\n";

    output
        << "    excitation_second_to_first_ratio: "
        << result
               .rotation_second_to_first_excitation_ratio
        << "\n";

    output
        << "    excitation_sufficient: "
        << (
               result
                   .rotation_excitation_sufficient
                   ? "true"
                   : "false"
           )
        << "\n";

    output
        << "    translation_available: "
        << (
               result.translation_available
                   ? "true"
                   : "false"
           )
        << "\n";

    output
        << "    translation_pairs: "
        << result.translation_pair_count
        << "\n";

    output
        << "    translation_rmse_m: "
        << result.translation_rmse_m
        << "\n";

    output
        << "    translation_median_residual_m: "
        << result
               .translation_median_residual_m
        << "\n";

    output
        << "    translation_max_residual_m: "
        << result
               .translation_max_residual_m
        << "\n";

    output
        << "    translation_singular_values_desc: ["
        << result
               .translation_singular_values
               .x()
        << ", "
        << result
               .translation_singular_values
               .y()
        << ", "
        << result
               .translation_singular_values
               .z()
        << "]\n";

    output
        << "    translation_observability_ratio: "
        << result
               .translation_observability_ratio
        << "\n";

    output
        << "    translation_condition_number: "
        << result
               .translation_condition_number
        << "\n";

    output
        << "    translation_sufficient: "
        << (
               result.translation_sufficient
                   ? "true"
                   : "false"
           )
        << "\n";
}

void PrintResult(
    const CalibrationResult &result,
    const std::string &output_path)
{
    const Eigen::Quaterniond q_LI(
        result.R_LI);

    const Eigen::Quaterniond q_IL(
        result.R_IL);

    std::cout
        << std::fixed
        << std::setprecision(8);

    std::cout
        << "\n===== LiDAR-IMU SE(3) EXTRINSIC =====\n";

    std::cout
        << "\nIMU -> LiDAR\n";

    std::cout
        << "q_LI [x y z w] = "
        << q_LI.x() << " "
        << q_LI.y() << " "
        << q_LI.z() << " "
        << q_LI.w() << "\n";

    std::cout
        << "P_LI [m] = "
        << result.P_LI.transpose()
        << "\n";

    std::cout
        << "\nLiDAR -> IMU (FR-SLAM)\n";

    std::cout
        << "q_IL [x y z w] = "
        << q_IL.x() << " "
        << q_IL.y() << " "
        << q_IL.z() << " "
        << q_IL.w() << "\n";

    std::cout
        << "P_IL [m] = "
        << result.P_IL.transpose()
        << "\n";

    std::cout
        << "\nRotation residual mean/median/max [deg] = "
        << result.mean_rotation_residual_deg
        << " / "
        << result.median_rotation_residual_deg
        << " / "
        << result.max_rotation_residual_deg
        << "\n";

    std::cout
        << "Rotation excitation ratio = "
        << result
               .rotation_second_to_first_excitation_ratio
        << " -> "
        << (
               result
                   .rotation_excitation_sufficient
                   ? "PASS"
                   : "FAIL"
           )
        << "\n";

    std::cout
        << "\nTranslation pairs = "
        << result.translation_pair_count
        << "\n";

    std::cout
        << "Translation RMSE [m] = "
        << result.translation_rmse_m
        << "\n";

    std::cout
        << "Translation median/max residual [m] = "
        << result.translation_median_residual_m
        << " / "
        << result.translation_max_residual_m
        << "\n";

    std::cout
        << "Translation singular values = "
        << result
               .translation_singular_values
               .transpose()
        << "\n";

    std::cout
        << "Translation observability ratio = "
        << result.translation_observability_ratio
        << "\n";

    std::cout
        << "Translation condition number = "
        << result.translation_condition_number
        << "\n";

    std::cout
        << "Translation observability = "
        << (
               result.translation_sufficient
                   ? "PASS"
                   : "FAIL"
           )
        << "\n";

    std::cout
        << "\nYAML written to: "
        << output_path
        << "\n";
}

std::vector<MotionPair>
GenerateSelfTestPairs(
    const Eigen::Matrix3d &true_R_LI,
    const Eigen::Vector3d &true_P_LI)
{
    std::mt19937 generator(42U);

    std::normal_distribution<double>
        rotation_noise(
            0.0,
            DegToRad(0.02));

    std::normal_distribution<double>
        translation_noise(
            0.0,
            0.0005);

    std::uniform_real_distribution<double>
        angle_distribution(
            DegToRad(3.0),
            DegToRad(30.0));

    std::normal_distribution<double>
        axis_distribution(
            0.0,
            1.0);

    std::normal_distribution<double>
        translation_distribution(
            0.0,
            0.15);

    std::vector<MotionPair> pairs;

    pairs.reserve(150U);

    for (std::size_t i = 0;
         i < 150U;
         ++i)
    {
        Eigen::Vector3d axis(
            axis_distribution(generator),
            axis_distribution(generator),
            axis_distribution(generator));

        axis.normalize();

        const double angle =
            angle_distribution(generator);

        const Eigen::Matrix3d R_I =
            Eigen::AngleAxisd(
                angle,
                axis)
                .toRotationMatrix();

        const Eigen::Vector3d t_I(
            translation_distribution(generator),
            translation_distribution(generator),
            translation_distribution(generator));

        Eigen::Matrix3d R_L =
            true_R_LI *
            R_I *
            true_R_LI.transpose();

        Eigen::Vector3d t_L =
            true_R_LI *
                t_I +
            true_P_LI -
            R_L *
                true_P_LI;

        Eigen::Vector3d rotational_noise_vector(
            rotation_noise(generator),
            rotation_noise(generator),
            rotation_noise(generator));

        const double noise_angle =
            rotational_noise_vector.norm();

        if (noise_angle > 1.0e-12)
        {
            R_L =
                R_L *
                Eigen::AngleAxisd(
                    noise_angle,
                    rotational_noise_vector /
                        noise_angle)
                    .toRotationMatrix();
        }

        t_L +=
            Eigen::Vector3d(
                translation_noise(generator),
                translation_noise(generator),
                translation_noise(generator));

        MotionPair pair;

        pair.start_timestamp =
            static_cast<double>(i);

        pair.end_timestamp =
            static_cast<double>(i + 1U);

        pair.delta_R_imu =
            R_I;

        pair.delta_t_imu =
            t_I;

        pair.delta_R_lidar =
            R_L;

        pair.delta_t_lidar =
            t_L;

        pair.has_translation =
            true;

        pairs.push_back(pair);
    }

    return pairs;
}

void PrintUsage(
    const char *program)
{
    std::cout
        << "Usage:\n"
        << "  "
        << program
        << " --input motion_pairs.csv "
        << "[--output result.yaml]\n"
        << "  "
        << program
        << " --self-test "
        << "[--output result.yaml]\n\n"

        << "Options:\n"
        << "  --min-angle-deg VALUE             default 1.0\n"
        << "  --max-angle-diff-deg VALUE        default 5.0\n"
        << "  --huber-deg VALUE                 default 2.0\n"
        << "  --translation-huber-m VALUE       default 0.05\n"
        << "  --translation-observability VALUE default 0.01\n"
        << "  --irls-iterations VALUE           default 20\n\n"

        << "Full SE(3) CSV:\n"
        << "start_timestamp,end_timestamp,"
        << "lidar_tx,lidar_ty,lidar_tz,"
        << "lidar_qx,lidar_qy,lidar_qz,lidar_qw,"
        << "imu_tx,imu_ty,imu_tz,"
        << "imu_qx,imu_qy,imu_qz,imu_qw\n";
}

Options ParseArguments(
    const int argc,
    char **argv)
{
    Options options;

    for (int i = 1;
         i < argc;
         ++i)
    {
        const std::string argument(
            argv[i]);

        auto require_value =
            [&](const std::string &name)
            -> std::string
        {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    "Missing value after " +
                    name);
            }

            return
                std::string(
                    argv[++i]);
        };

        if (argument == "--input")
        {
            options.input_path =
                require_value(argument);
        }
        else if (argument == "--output")
        {
            options.output_path =
                require_value(argument);
        }
        else if (
            argument ==
            "--min-angle-deg")
        {
            options.min_rotation_deg =
                std::stod(
                    require_value(argument));
        }
        else if (
            argument ==
            "--max-angle-diff-deg")
        {
            options
                .max_angle_difference_deg =
                std::stod(
                    require_value(argument));
        }
        else if (
            argument ==
            "--huber-deg")
        {
            options
                .rotation_huber_delta_deg =
                std::stod(
                    require_value(argument));
        }
        else if (
            argument ==
            "--translation-huber-m")
        {
            options
                .translation_huber_delta_m =
                std::stod(
                    require_value(argument));
        }
        else if (
            argument ==
            "--translation-observability")
        {
            options
                .minimum_translation_observability_ratio =
                std::stod(
                    require_value(argument));
        }
        else if (
            argument ==
            "--irls-iterations")
        {
            options.max_irls_iterations =
                std::stoi(
                    require_value(argument));
        }
        else if (
            argument ==
            "--self-test")
        {
            options.self_test =
                true;
        }
        else if (
            argument == "--help" ||
            argument == "-h")
        {
            PrintUsage(
                argv[0]);

            std::exit(0);
        }
        else
        {
            throw std::runtime_error(
                "Unknown argument: " +
                argument);
        }
    }

    if (!options.self_test &&
        options.input_path.empty())
    {
        throw std::runtime_error(
            "Provide --input CSV "
            "or use --self-test.");
    }

    if (
        options.min_rotation_deg < 0.0 ||
        options.max_angle_difference_deg <= 0.0 ||
        options.rotation_huber_delta_deg <= 0.0 ||
        options.translation_huber_delta_m <= 0.0 ||
        options.minimum_translation_observability_ratio <= 0.0 ||
        options.max_irls_iterations <= 0)
    {
        throw std::runtime_error(
            "Calibration thresholds must be positive.");
    }

    return options;
}

}  // namespace

int main(
    const int argc,
    char **argv)
{
    try
    {
        Options options =
            ParseArguments(
                argc,
                argv);

        options.output_path =
            ResolveNonOverwritingOutputPath(
                options.output_path);

        std::vector<MotionPair> pairs;

        Eigen::Matrix3d true_R_LI =
            Eigen::Matrix3d::Identity();

        Eigen::Vector3d true_P_LI =
            Eigen::Vector3d::Zero();

        if (options.self_test)
        {
            true_R_LI =
                (
                    Eigen::AngleAxisd(
                        DegToRad(12.0),
                        Eigen::Vector3d::UnitZ()) *
                    Eigen::AngleAxisd(
                        DegToRad(-4.0),
                        Eigen::Vector3d::UnitY()) *
                    Eigen::AngleAxisd(
                        DegToRad(2.5),
                        Eigen::Vector3d::UnitX())
                ).toRotationMatrix();

            true_P_LI =
                Eigen::Vector3d(
                    0.08,
                    -0.035,
                    0.12);

            pairs =
                GenerateSelfTestPairs(
                    true_R_LI,
                    true_P_LI);
        }
        else
        {
            pairs =
                LoadMotionPairs(
                    options.input_path);
        }

        const CalibrationResult result =
            CalibrateExtrinsic(
                pairs,
                options);

        WriteYaml(
            result,
            options);

        PrintResult(
            result,
            options.output_path);

        if (options.self_test)
        {
            const double rotation_error_deg =
                RadToDeg(
                    RotationAngle(
                        true_R_LI.transpose() *
                        result.R_LI));

            const double translation_error_m =
                (
                    true_P_LI -
                    result.P_LI
                ).norm();

            std::cout
                << "\nSelf-test rotation error [deg] = "
                << rotation_error_deg
                << "\n";

            std::cout
                << "Self-test translation error [m] = "
                << translation_error_m
                << "\n";

            if (
                rotation_error_deg > 0.10 ||
                translation_error_m > 0.01)
            {
                std::cerr
                    << "Self-test FAILED\n";

                return 2;
            }

            std::cout
                << "Self-test PASSED\n";
        }

        if (!result
                 .rotation_excitation_sufficient)
        {
            std::cerr
                << "WARNING: rotational excitation insufficient. "
                << "Use rotations around at least two "
                << "non-parallel axes.\n";

            return 3;
        }

        if (
            result.translation_available &&
            !result.translation_sufficient)
        {
            std::cerr
                << "WARNING: translation is poorly observable. "
                << "Use richer 3-D rotational excitation.\n";

            return 4;
        }

        if (!result.translation_available)
        {
            std::cerr
                << "WARNING: legacy rotation-only CSV detected. "
                << "P_IL was NOT calibrated.\n";

            return 5;
        }

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "Calibration failed: "
            << error.what()
            << "\n";

        PrintUsage(
            argv[0]);

        return 1;
    }
}
