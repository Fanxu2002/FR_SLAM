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

double DegToRad(const double degrees) { return degrees * kPi / 180.0; }
double RadToDeg(const double radians) { return radians * 180.0 / kPi; }

struct RotationPair {
  double timestamp = 0.0;
  Eigen::Matrix3d delta_R_lidar = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d delta_R_imu = Eigen::Matrix3d::Identity();
};

struct Options {
  std::string input_path;
  std::string output_path = "lidar_imu_rotation_calibration.yaml";
  double min_rotation_deg = 1.0;
  double max_angle_difference_deg = 5.0;
  double huber_delta_deg = 2.0;
  int max_irls_iterations = 20;
  bool self_test = false;
};

struct CalibrationResult {
  Eigen::Matrix3d R_LI = Eigen::Matrix3d::Identity();
  std::vector<double> residuals_deg;
  Eigen::Vector3d excitation_eigenvalues = Eigen::Vector3d::Zero();
  std::size_t total_pair_count = 0;
  std::size_t used_pair_count = 0;
  std::size_t rejected_small_rotation_count = 0;
  std::size_t rejected_angle_mismatch_count = 0;
  double mean_residual_deg = 0.0;
  double median_residual_deg = 0.0;
  double max_residual_deg = 0.0;
  double second_to_first_excitation_ratio = 0.0;
  bool excitation_sufficient = false;
};

std::string BuildTimestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &time);
#else
  localtime_r(&time, &local_time);
#endif

  std::ostringstream stream;
  stream << std::put_time(&local_time, "%Y%m%d_%H%M%S");
  return stream.str();
}

std::string ResolveNonOverwritingOutputPath(
    const std::string &requested_output_path) {
  namespace fs = std::filesystem;

  const fs::path requested_path(requested_output_path);
  const fs::path parent_path = requested_path.parent_path();
  if (!parent_path.empty()) {
    std::error_code directory_error;
    fs::create_directories(parent_path, directory_error);
    if (directory_error) {
      throw std::runtime_error(
          "Cannot create output directory '" + parent_path.string() +
          "': " + directory_error.message());
    }
  }

  std::error_code exists_error;
  const bool requested_exists = fs::exists(requested_path, exists_error);
  if (exists_error) {
    throw std::runtime_error(
        "Cannot inspect output path '" + requested_path.string() +
        "': " + exists_error.message());
  }
  if (!requested_exists) {
    return requested_path.string();
  }

  const std::string timestamp = BuildTimestamp();
  const std::string stem = requested_path.stem().string();
  const std::string extension = requested_path.extension().string();

  for (std::size_t suffix = 0; suffix < 10000U; ++suffix) {
    std::ostringstream filename;
    filename << stem << "_" << timestamp;
    if (suffix > 0U) {
      filename << "_" << suffix;
    }
    filename << extension;

    const fs::path candidate = parent_path / filename.str();
    std::error_code candidate_error;
    const bool candidate_exists = fs::exists(candidate, candidate_error);
    if (candidate_error) {
      throw std::runtime_error(
          "Cannot inspect output path '" + candidate.string() +
          "': " + candidate_error.message());
    }
    if (!candidate_exists) {
      std::cout
          << "Requested output already exists and will be preserved:\n  "
          << requested_path.string() << "\n"
          << "Writing this result to a new file instead:\n  "
          << candidate.string() << "\n";
      return candidate.string();
    }
  }

  throw std::runtime_error(
      "Unable to allocate a non-overwriting calibration output path.");
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R) {
  const Eigen::AngleAxisd angle_axis(R);
  if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() < 1e-12) {
    return Eigen::Vector3d::Zero();
  }
  return angle_axis.angle() * angle_axis.axis();
}

double RotationAngle(const Eigen::Matrix3d &R) {
  return LogSO3(R).norm();
}

Eigen::Matrix3d ProjectToSO3(const Eigen::Matrix3d &matrix) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
  correction(2, 2) = (U * V.transpose()).determinant();
  return U * correction * V.transpose();
}

Eigen::Matrix3d SolveWeightedWahba(
    const std::vector<Eigen::Vector3d> &lidar_rotation_vectors,
    const std::vector<Eigen::Vector3d> &imu_rotation_vectors,
    const std::vector<double> &weights) {
  Eigen::Matrix3d cross_covariance = Eigen::Matrix3d::Zero();
  for (std::size_t i = 0; i < weights.size(); ++i) {
    cross_covariance.noalias() +=
        weights[i] * lidar_rotation_vectors[i] *
        imu_rotation_vectors[i].transpose();
  }
  return ProjectToSO3(cross_covariance);
}

double HuberWeight(const double residual, const double delta) {
  if (residual <= delta || residual < 1e-12) {
    return 1.0;
  }
  return delta / residual;
}

CalibrationResult CalibrateRotation(const std::vector<RotationPair> &pairs,
                                    const Options &options) {
  CalibrationResult result;
  result.total_pair_count = pairs.size();

  std::vector<RotationPair> accepted_pairs;
  std::vector<Eigen::Vector3d> lidar_rotation_vectors;
  std::vector<Eigen::Vector3d> imu_rotation_vectors;
  accepted_pairs.reserve(pairs.size());
  lidar_rotation_vectors.reserve(pairs.size());
  imu_rotation_vectors.reserve(pairs.size());

  const double min_rotation = DegToRad(options.min_rotation_deg);
  const double max_angle_difference =
      DegToRad(options.max_angle_difference_deg);

  for (const RotationPair &pair : pairs) {
    const Eigen::Vector3d lidar_vector = LogSO3(pair.delta_R_lidar);
    const Eigen::Vector3d imu_vector = LogSO3(pair.delta_R_imu);
    const double lidar_angle = lidar_vector.norm();
    const double imu_angle = imu_vector.norm();

    if (std::max(lidar_angle, imu_angle) < min_rotation) {
      ++result.rejected_small_rotation_count;
      continue;
    }
    if (std::abs(lidar_angle - imu_angle) > max_angle_difference) {
      ++result.rejected_angle_mismatch_count;
      continue;
    }

    accepted_pairs.push_back(pair);
    lidar_rotation_vectors.push_back(lidar_vector);
    imu_rotation_vectors.push_back(imu_vector);
  }

  result.used_pair_count = accepted_pairs.size();
  if (accepted_pairs.size() < 3U) {
    throw std::runtime_error(
        "Fewer than three usable rotation pairs remain after filtering.");
  }

  std::vector<double> weights(accepted_pairs.size(), 1.0);
  Eigen::Matrix3d R_LI = SolveWeightedWahba(
      lidar_rotation_vectors, imu_rotation_vectors, weights);
  const double huber_delta = DegToRad(options.huber_delta_deg);

  for (int iteration = 0; iteration < options.max_irls_iterations;
       ++iteration) {
    for (std::size_t i = 0; i < accepted_pairs.size(); ++i) {
      const double residual =
          (lidar_rotation_vectors[i] - R_LI * imu_rotation_vectors[i])
              .norm();
      weights[i] = HuberWeight(residual, huber_delta);
    }

    const Eigen::Matrix3d updated_R_LI = SolveWeightedWahba(
        lidar_rotation_vectors, imu_rotation_vectors, weights);
    const double update_angle =
        RotationAngle(R_LI.transpose() * updated_R_LI);
    R_LI = updated_R_LI;
    if (update_angle < 1e-10) {
      break;
    }
  }

  result.R_LI = R_LI;
  result.residuals_deg.reserve(accepted_pairs.size());
  for (const RotationPair &pair : accepted_pairs) {
    // AX = XB with A=delta_R_lidar, B=delta_R_imu, X=R_LI.
    const Eigen::Matrix3d error_rotation =
        pair.delta_R_lidar * R_LI * pair.delta_R_imu.transpose() *
        R_LI.transpose();
    result.residuals_deg.push_back(RadToDeg(RotationAngle(error_rotation)));
  }

  const double residual_sum =
      std::accumulate(result.residuals_deg.begin(),
                      result.residuals_deg.end(), 0.0);
  result.mean_residual_deg = residual_sum / result.residuals_deg.size();
  std::vector<double> sorted_residuals = result.residuals_deg;
  std::sort(sorted_residuals.begin(), sorted_residuals.end());
  const std::size_t middle = sorted_residuals.size() / 2U;
  if (sorted_residuals.size() % 2U == 0U) {
    result.median_residual_deg =
        0.5 * (sorted_residuals[middle - 1U] + sorted_residuals[middle]);
  } else {
    result.median_residual_deg = sorted_residuals[middle];
  }
  result.max_residual_deg = sorted_residuals.back();

  Eigen::Matrix3d excitation = Eigen::Matrix3d::Zero();
  for (const Eigen::Vector3d &vector : imu_rotation_vectors) {
    excitation.noalias() += vector * vector.transpose();
  }
  const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(excitation);
  if (eigensolver.info() != Eigen::Success) {
    throw std::runtime_error("Excitation eigenvalue decomposition failed.");
  }
  result.excitation_eigenvalues = eigensolver.eigenvalues().reverse();
  if (result.excitation_eigenvalues.x() > 1e-12) {
    result.second_to_first_excitation_ratio =
        result.excitation_eigenvalues.y() /
        result.excitation_eigenvalues.x();
  }
  // Two non-parallel rotation axes are the minimum required to determine R_LI.
  result.excitation_sufficient =
      result.second_to_first_excitation_ratio >= 0.05;
  return result;
}

bool ParseNumericCsvRow(const std::string &line,
                        std::vector<double> &values) {
  values.clear();
  std::stringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ',')) {
    const std::size_t first = cell.find_first_not_of(" \t\r\n");
    const std::size_t last = cell.find_last_not_of(" \t\r\n");
    if (first == std::string::npos) {
      return false;
    }
    const std::string trimmed = cell.substr(first, last - first + 1U);
    try {
      std::size_t parsed_count = 0;
      const double value = std::stod(trimmed, &parsed_count);
      if (parsed_count != trimmed.size() || !std::isfinite(value)) {
        return false;
      }
      values.push_back(value);
    } catch (const std::exception &) {
      return false;
    }
  }
  return true;
}

std::vector<RotationPair> LoadRotationPairs(const std::string &path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open input CSV: " + path);
  }

  std::vector<RotationPair> pairs;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const std::size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos || line[first] == '#') {
      continue;
    }

    std::vector<double> values;
    if (!ParseNumericCsvRow(line, values)) {
      // A single textual header is allowed.
      if (pairs.empty()) {
        continue;
      }
      throw std::runtime_error("Invalid numeric CSV row at line " +
                               std::to_string(line_number));
    }
    if (values.size() != 9U && values.size() != 10U) {
      throw std::runtime_error(
          "Expected 9 or 10 CSV columns at line " +
          std::to_string(line_number) + ", got " +
          std::to_string(values.size()));
    }

    // Supported formats:
    //   9 columns: end_time, lidar_xyzw, imu_xyzw
    //  10 columns: start_time, end_time, lidar_xyzw, imu_xyzw
    const std::size_t quaternion_offset =
        values.size() == 10U ? 2U : 1U;

    const Eigen::Quaterniond q_lidar(
        values[quaternion_offset + 3U],
        values[quaternion_offset + 0U],
        values[quaternion_offset + 1U],
        values[quaternion_offset + 2U]);

    const Eigen::Quaterniond q_imu(
        values[quaternion_offset + 7U],
        values[quaternion_offset + 4U],
        values[quaternion_offset + 5U],
        values[quaternion_offset + 6U]);
    if (q_lidar.norm() < 1e-8 || q_imu.norm() < 1e-8) {
      throw std::runtime_error("Near-zero quaternion at line " +
                               std::to_string(line_number));
    }

    RotationPair pair;
    pair.timestamp =
        values.size() == 10U
            ? values[1]
            : values[0];
    pair.delta_R_lidar = q_lidar.normalized().toRotationMatrix();
    pair.delta_R_imu = q_imu.normalized().toRotationMatrix();
    pairs.push_back(pair);
  }

  if (pairs.empty()) {
    throw std::runtime_error("No rotation pairs were loaded from: " + path);
  }
  return pairs;
}

Eigen::Vector3d MatrixToRpyZyx(const Eigen::Matrix3d &R) {
  const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));
  const double cos_pitch = std::cos(pitch);
  double roll = 0.0;
  double yaw = 0.0;
  if (std::abs(cos_pitch) > 1e-8) {
    roll = std::atan2(R(2, 1), R(2, 2));
    yaw = std::atan2(R(1, 0), R(0, 0));
  } else {
    roll = std::atan2(-R(1, 2), R(1, 1));
  }
  return Eigen::Vector3d(roll, pitch, yaw);
}

void WriteYaml(const CalibrationResult &result, const Options &options) {
  std::ofstream output(options.output_path);
  if (!output.is_open()) {
    throw std::runtime_error("Cannot open output YAML: " +
                             options.output_path);
  }

  const Eigen::Quaterniond q_LI(result.R_LI);
  const Eigen::Matrix3d R_IL =
      result.R_LI.transpose();
  const Eigen::Quaterniond q_IL(R_IL);
  const Eigen::Vector3d rpy_deg =
      MatrixToRpyZyx(result.R_LI).unaryExpr(
          [](const double value) { return RadToDeg(value); });

  output << std::setprecision(12);
  output << "lidar_imu_rotation_calibration:\n";
  output << "  convention: \"v_L = R_LI * v_I\"\n";
  output << "  relative_motion_equation: \"delta_R_L = R_LI * delta_R_I * R_LI^T\"\n";
  output << "  quaternion_xyzw: [" << q_LI.x() << ", " << q_LI.y()
         << ", " << q_LI.z() << ", " << q_LI.w() << "]\n";
  output << "  rpy_deg_zyx: [" << rpy_deg.x() << ", " << rpy_deg.y()
         << ", " << rpy_deg.z() << "]\n";
  output << "  rotation_matrix:\n";
  for (int row = 0; row < 3; ++row) {
    output << "    - [" << result.R_LI(row, 0) << ", "
           << result.R_LI(row, 1) << ", " << result.R_LI(row, 2)
           << "]\n";
  }
  output << "  inverse_lidar_to_imu:\n";
  output << "    convention: \"v_I = R_IL * v_L\"\n";
  output << "    quaternion_xyzw: [" << q_IL.x() << ", " << q_IL.y()
         << ", " << q_IL.z() << ", " << q_IL.w() << "]\n";
  output << "    rotation_matrix:\n";
  for (int row = 0; row < 3; ++row) {
    output << "      - [" << R_IL(row, 0) << ", "
           << R_IL(row, 1) << ", " << R_IL(row, 2)
           << "]\n";
  }
  output << "  diagnostics:\n";
  output << "    total_pairs: " << result.total_pair_count << "\n";
  output << "    used_pairs: " << result.used_pair_count << "\n";
  output << "    rejected_small_rotation: "
         << result.rejected_small_rotation_count << "\n";
  output << "    rejected_angle_mismatch: "
         << result.rejected_angle_mismatch_count << "\n";
  output << "    mean_residual_deg: " << result.mean_residual_deg << "\n";
  output << "    median_residual_deg: " << result.median_residual_deg << "\n";
  output << "    max_residual_deg: " << result.max_residual_deg << "\n";
  output << "    excitation_eigenvalues_desc: ["
         << result.excitation_eigenvalues.x() << ", "
         << result.excitation_eigenvalues.y() << ", "
         << result.excitation_eigenvalues.z() << "]\n";
  output << "    excitation_second_to_first_ratio: "
         << result.second_to_first_excitation_ratio << "\n";
  output << "    excitation_sufficient: "
         << (result.excitation_sufficient ? "true" : "false") << "\n";
}

void PrintResult(const CalibrationResult &result,
                 const std::string &output_path) {
  const Eigen::Quaterniond q_LI(result.R_LI);
  const Eigen::Quaterniond q_IL(
      result.R_LI.transpose());
  const Eigen::Vector3d rpy_deg =
      MatrixToRpyZyx(result.R_LI).unaryExpr(
          [](const double value) { return RadToDeg(value); });

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\nR_LI (v_L = R_LI * v_I):\n" << result.R_LI << "\n";
  std::cout << "q_LI [x y z w]: " << q_LI.x() << " " << q_LI.y() << " "
            << q_LI.z() << " " << q_LI.w() << "\n";
  std::cout << "q_IL [x y z w] for FR-SLAM parameters: "
            << q_IL.x() << " " << q_IL.y() << " " << q_IL.z() << " "
            << q_IL.w() << "\n";
  std::cout << "RPY ZYX [deg]: " << rpy_deg.transpose() << "\n";
  std::cout << "Pairs total/used: " << result.total_pair_count << "/"
            << result.used_pair_count << "\n";
  std::cout << "Residual mean/median/max [deg]: "
            << result.mean_residual_deg << "/" << result.median_residual_deg
            << "/" << result.max_residual_deg << "\n";
  std::cout << "Excitation eigenvalues: "
            << result.excitation_eigenvalues.transpose() << "\n";
  std::cout << "Excitation second/first ratio: "
            << result.second_to_first_excitation_ratio << " -> "
            << (result.excitation_sufficient ? "PASS" : "INSUFFICIENT")
            << "\n";
  std::cout << "YAML written to: " << output_path << "\n";
}

std::vector<RotationPair> GenerateSelfTestPairs(
    const Eigen::Matrix3d &true_R_LI) {
  std::mt19937 generator(42U);
  std::normal_distribution<double> noise(0.0, DegToRad(0.03));
  std::uniform_real_distribution<double> angle_distribution(
      DegToRad(2.0), DegToRad(25.0));
  std::normal_distribution<double> axis_distribution(0.0, 1.0);

  std::vector<RotationPair> pairs;
  pairs.reserve(120U);
  for (std::size_t i = 0; i < 120U; ++i) {
    Eigen::Vector3d axis(axis_distribution(generator),
                         axis_distribution(generator),
                         axis_distribution(generator));
    axis.normalize();
    const double angle = angle_distribution(generator);
    const Eigen::Matrix3d delta_R_imu =
        Eigen::AngleAxisd(angle, axis).toRotationMatrix();
    Eigen::Vector3d lidar_noise(noise(generator), noise(generator),
                                noise(generator));
    const double lidar_noise_angle = lidar_noise.norm();
    Eigen::Matrix3d noise_rotation = Eigen::Matrix3d::Identity();
    if (lidar_noise_angle > 1e-12) {
      noise_rotation =
          Eigen::AngleAxisd(lidar_noise_angle,
                            lidar_noise / lidar_noise_angle)
              .toRotationMatrix();
    }

    RotationPair pair;
    pair.timestamp = static_cast<double>(i);
    pair.delta_R_imu = delta_R_imu;
    pair.delta_R_lidar =
        true_R_LI * delta_R_imu * true_R_LI.transpose() * noise_rotation;
    pairs.push_back(pair);
  }
  return pairs;
}

void PrintUsage(const char *program) {
  std::cout
      << "Usage:\n"
      << "  " << program
      << " --input rotation_pairs.csv [--output result.yaml]\n"
      << "  " << program << " --self-test [--output result.yaml]\n\n"
      << "Options:\n"
      << "  --min-angle-deg VALUE       default 1.0\n"
      << "  --max-angle-diff-deg VALUE  default 5.0\n"
      << "  --huber-deg VALUE           default 2.0\n"
      << "  --irls-iterations VALUE     default 20\n\n"
      << "CSV columns written by FR-SLAM (quaternions are x,y,z,w):\n"
      << "start_timestamp,end_timestamp,"
         "lidar_qx,lidar_qy,lidar_qz,lidar_qw,"
         "imu_qx,imu_qy,imu_qz,imu_qw\n";
}

Options ParseArguments(const int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument(argv[i]);
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value after " + name);
      }
      return std::string(argv[++i]);
    };

    if (argument == "--input") {
      options.input_path = require_value(argument);
    } else if (argument == "--output") {
      options.output_path = require_value(argument);
    } else if (argument == "--min-angle-deg") {
      options.min_rotation_deg = std::stod(require_value(argument));
    } else if (argument == "--max-angle-diff-deg") {
      options.max_angle_difference_deg = std::stod(require_value(argument));
    } else if (argument == "--huber-deg") {
      options.huber_delta_deg = std::stod(require_value(argument));
    } else if (argument == "--irls-iterations") {
      options.max_irls_iterations = std::stoi(require_value(argument));
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--help" || argument == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown argument: " + argument);
    }
  }

  if (!options.self_test && options.input_path.empty()) {
    throw std::runtime_error("Provide --input CSV or use --self-test.");
  }
  if (options.min_rotation_deg < 0.0 ||
      options.max_angle_difference_deg <= 0.0 ||
      options.huber_delta_deg <= 0.0 || options.max_irls_iterations <= 0) {
    throw std::runtime_error("Calibration thresholds must be positive.");
  }
  return options;
}

}  // namespace

int main(const int argc, char **argv) {
  try {
    Options options = ParseArguments(argc, argv);
    options.output_path =
        ResolveNonOverwritingOutputPath(options.output_path);
    std::vector<RotationPair> pairs;
    Eigen::Matrix3d true_R_LI = Eigen::Matrix3d::Identity();

    if (options.self_test) {
      true_R_LI =
          (Eigen::AngleAxisd(DegToRad(12.0), Eigen::Vector3d::UnitZ()) *
           Eigen::AngleAxisd(DegToRad(-4.0), Eigen::Vector3d::UnitY()) *
           Eigen::AngleAxisd(DegToRad(2.5), Eigen::Vector3d::UnitX()))
              .toRotationMatrix();
      pairs = GenerateSelfTestPairs(true_R_LI);
    } else {
      pairs = LoadRotationPairs(options.input_path);
    }

    const CalibrationResult result = CalibrateRotation(pairs, options);
    WriteYaml(result, options);
    PrintResult(result, options.output_path);

    if (options.self_test) {
      const double estimation_error_deg =
          RadToDeg(RotationAngle(true_R_LI.transpose() * result.R_LI));
      std::cout << "Self-test R_LI error [deg]: " << estimation_error_deg
                << "\n";
      if (estimation_error_deg > 0.1) {
        std::cerr << "Self-test FAILED\n";
        return 2;
      }
      std::cout << "Self-test PASSED\n";
    }

    if (!result.excitation_sufficient) {
      std::cerr
          << "WARNING: motion excitation is weak. Record rotations around at "
             "least two non-parallel axes before trusting R_LI.\n";
      return 3;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Calibration failed: " << error.what() << "\n";
    PrintUsage(argv[0]);
    return 1;
  }
}
