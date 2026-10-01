#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kRadToDeg = 180.0 / kPi;

struct Pose {
    double t = 0.0;
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

struct MatchedPose {
    Pose est_raw;
    Pose gt_lidar;
};

struct Alignment {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

struct Alignment2D {
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    Eigen::Vector2d t = Eigen::Vector2d::Zero();
    double yaw_rad = 0.0;
};

struct ErrorSample {
    double trans_m = 0.0;
    double rot_deg = 0.0;

    // Signed translation residuals:
    //     est_aligned - gt
    double x_m = 0.0;
    double y_m = 0.0;
    double z_m = 0.0;

    // Signed ZYX attitude residuals.
    double roll_deg = 0.0;
    double pitch_deg = 0.0;
    double yaw_deg = 0.0;
};

struct Stats {
    std::size_t n = 0;
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double mean = std::numeric_limits<double>::quiet_NaN();
    double median = std::numeric_limits<double>::quiet_NaN();
    double p95 = std::numeric_limits<double>::quiet_NaN();
    double max = std::numeric_limits<double>::quiet_NaN();
};

struct RpeStats {
    double target_distance_m = 0.0;
    std::size_t n = 0;
    Stats translation;
    Stats rotation_deg;
};

struct Options {
    fs::path est_path;
    fs::path gt_path;
    fs::path output_dir;
    std::string extrinsic = "hortimulti";
    double max_gt_gap_s = 0.5;
    bool make_plots = true;
};

std::string Trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitCsv(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
        out.push_back(Trim(cell));
    }
    if (!line.empty() && line.back() == ',') {
        out.emplace_back();
    }
    return out;
}

std::unordered_map<std::string, std::size_t> HeaderMap(
    const std::vector<std::string>& header) {
    std::unordered_map<std::string, std::size_t> map;
    for (std::size_t i = 0; i < header.size(); ++i) {
        map[Trim(header[i])] = i;
    }
    return map;
}

std::size_t RequireColumn(
    const std::unordered_map<std::string, std::size_t>& h,
    const std::string& name) {
    const auto it = h.find(name);
    if (it == h.end()) {
        throw std::runtime_error("Missing required CSV column: " + name);
    }
    return it->second;
}

double ParseDouble(const std::string& text, const fs::path& path, std::size_t line_no) {
    try {
        std::size_t used = 0;
        const double value = std::stod(text, &used);
        if (used != text.size()) {
            throw std::runtime_error("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Failed to parse numeric value '" + text + "' at " +
            path.string() + ":" + std::to_string(line_no));
    }
}

std::vector<Pose> LoadPoseCsv(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Cannot open CSV: " + path.string());
    }

    std::string line;
    if (!std::getline(in, line)) {
        throw std::runtime_error("Empty CSV: " + path.string());
    }

    const auto header = SplitCsv(line);
    const auto h = HeaderMap(header);

    const bool has_timestamp = h.find("timestamp") != h.end();
    const bool has_time = h.find("time") != h.end();
    if (!has_timestamp && !has_time) {
        throw std::runtime_error(
            "CSV must contain either 'timestamp' or 'time': " + path.string());
    }

    const std::size_t c_t = has_timestamp ? h.at("timestamp") : h.at("time");
    const std::size_t c_x = RequireColumn(h, "x");
    const std::size_t c_y = RequireColumn(h, "y");
    const std::size_t c_z = RequireColumn(h, "z");
    const std::size_t c_qx = RequireColumn(h, "qx");
    const std::size_t c_qy = RequireColumn(h, "qy");
    const std::size_t c_qz = RequireColumn(h, "qz");
    const std::size_t c_qw = RequireColumn(h, "qw");

    const std::size_t required_max = std::max(
        {c_t, c_x, c_y, c_z, c_qx, c_qy, c_qz, c_qw});

    std::vector<Pose> poses;
    std::size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (Trim(line).empty()) {
            continue;
        }
        const auto cells = SplitCsv(line);
        if (cells.size() <= required_max) {
            throw std::runtime_error(
                "Not enough CSV fields at " + path.string() + ":" +
                std::to_string(line_no));
        }

        Pose p;
        p.t = ParseDouble(cells[c_t], path, line_no);
        p.p.x() = ParseDouble(cells[c_x], path, line_no);
        p.p.y() = ParseDouble(cells[c_y], path, line_no);
        p.p.z() = ParseDouble(cells[c_z], path, line_no);

        const double qx = ParseDouble(cells[c_qx], path, line_no);
        const double qy = ParseDouble(cells[c_qy], path, line_no);
        const double qz = ParseDouble(cells[c_qz], path, line_no);
        const double qw = ParseDouble(cells[c_qw], path, line_no);
        p.q = Eigen::Quaterniond(qw, qx, qy, qz);
        const double norm = p.q.norm();
        if (!(norm > 1e-12) || !std::isfinite(norm)) {
            throw std::runtime_error(
                "Invalid quaternion at " + path.string() + ":" +
                std::to_string(line_no));
        }
        p.q.normalize();
        poses.push_back(p);
    }

    if (poses.size() < 2) {
        throw std::runtime_error("Need at least 2 poses in: " + path.string());
    }

    std::sort(poses.begin(), poses.end(), [](const Pose& a, const Pose& b) {
        return a.t < b.t;
    });

    for (std::size_t i = 1; i < poses.size(); ++i) {
        if (!(poses[i].t > poses[i - 1].t)) {
            throw std::runtime_error(
                "Pose timestamps are not strictly increasing after sorting: " +
                path.string());
        }
    }

    return poses;
}

bool InterpolatePose(
    const std::vector<Pose>& poses,
    double t,
    double max_gap_s,
    Pose& out,
    bool& rejected_for_gap) {
    rejected_for_gap = false;
    if (t < poses.front().t || t > poses.back().t) {
        return false;
    }

    const auto it = std::lower_bound(
        poses.begin(), poses.end(), t,
        [](const Pose& p, double value) { return p.t < value; });

    if (it == poses.begin()) {
        out = *it;
        out.t = t;
        return true;
    }
    if (it == poses.end()) {
        out = poses.back();
        out.t = t;
        return true;
    }
    if (std::abs(it->t - t) < 1e-9) {
        out = *it;
        return true;
    }

    const Pose& b = *it;
    const Pose& a = *(it - 1);
    const double dt = b.t - a.t;
    if (!(dt > 0.0)) {
        return false;
    }
    if (dt > max_gap_s) {
        rejected_for_gap = true;
        return false;
    }

    const double alpha = (t - a.t) / dt;
    out.t = t;
    out.p = (1.0 - alpha) * a.p + alpha * b.p;
    out.q = a.q.slerp(alpha, b.q).normalized();
    return true;
}

struct Extrinsic {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

Extrinsic GetExtrinsic(const std::string& name) {
    Extrinsic e;
    if (name == "identity") {
        return e;
    }
    if (name != "hortimulti") {
        throw std::runtime_error(
            "Unknown --extrinsic value: " + name +
            " (supported: hortimulti, identity)");
    }

    e.R <<
         0.99988877,  0.01206341, -0.00877090,
        -0.01220330,  0.99979631, -0.01607509,
         0.00857519,  0.01618033,  0.99983232;
    e.p << 0.14486527, 0.04820110, -0.28193628;
    return e;
}

Pose ApplyBodyToSensorExtrinsic(const Pose& T_WO, const Extrinsic& T_OL) {
    Pose out;
    out.t = T_WO.t;
    const Eigen::Quaterniond q_ol(T_OL.R);
    out.q = (T_WO.q * q_ol).normalized();
    out.p = T_WO.p + T_WO.q * T_OL.p;
    return out;
}

Alignment FirstPoseAlignment(const Pose& est0, const Pose& gt0) {
    Alignment a;
    a.R = gt0.q.toRotationMatrix() * est0.q.toRotationMatrix().transpose();
    a.t = gt0.p - a.R * est0.p;
    return a;
}

Alignment GlobalSe3Alignment(const std::vector<MatchedPose>& matched) {
    if (matched.size() < 3) {
        throw std::runtime_error("Need at least 3 matched poses for SE(3) alignment");
    }

    Eigen::MatrixXd src(3, static_cast<Eigen::Index>(matched.size()));
    Eigen::MatrixXd dst(3, static_cast<Eigen::Index>(matched.size()));
    for (std::size_t i = 0; i < matched.size(); ++i) {
        src.col(static_cast<Eigen::Index>(i)) = matched[i].est_raw.p;
        dst.col(static_cast<Eigen::Index>(i)) = matched[i].gt_lidar.p;
    }

    const Eigen::Matrix4d T = Eigen::umeyama(src, dst, false);
    Alignment a;
    a.R = T.block<3, 3>(0, 0);
    a.t = T.block<3, 1>(0, 3);

    if (a.R.determinant() < 0.0) {
        throw std::runtime_error("SE(3) alignment produced a reflection");
    }
    return a;
}

Alignment2D GlobalSe2XyAlignment(const std::vector<MatchedPose>& matched) {
    if (matched.size() < 2) {
        throw std::runtime_error("Need at least 2 matched poses for XY alignment");
    }

    Eigen::Vector2d src_mean = Eigen::Vector2d::Zero();
    Eigen::Vector2d dst_mean = Eigen::Vector2d::Zero();
    for (const MatchedPose& m : matched) {
        src_mean += m.est_raw.p.head<2>();
        dst_mean += m.gt_lidar.p.head<2>();
    }
    src_mean /= static_cast<double>(matched.size());
    dst_mean /= static_cast<double>(matched.size());

    double dot_sum = 0.0;
    double cross_sum = 0.0;
    double src_energy = 0.0;
    for (const MatchedPose& m : matched) {
        const Eigen::Vector2d s = m.est_raw.p.head<2>() - src_mean;
        const Eigen::Vector2d d = m.gt_lidar.p.head<2>() - dst_mean;
        dot_sum += s.x() * d.x() + s.y() * d.y();
        cross_sum += s.x() * d.y() - s.y() * d.x();
        src_energy += s.squaredNorm();
    }

    if (!(src_energy > 1e-12) ||
        (!std::isfinite(dot_sum)) ||
        (!std::isfinite(cross_sum))) {
        throw std::runtime_error("Degenerate data for global XY alignment");
    }

    Alignment2D a;
    a.yaw_rad = std::atan2(cross_sum, dot_sum);
    const double c = std::cos(a.yaw_rad);
    const double sn = std::sin(a.yaw_rad);
    a.R << c, -sn,
           sn,  c;
    a.t = dst_mean - a.R * src_mean;
    return a;
}

Eigen::Vector2d ApplyAlignment2D(
    const Eigen::Vector3d& p,
    const Alignment2D& a) {
    return a.R * p.head<2>() + a.t;
}

double XyErrorM(
    const Eigen::Vector2d& est_xy,
    const Eigen::Vector3d& gt_p) {
    return (est_xy - gt_p.head<2>()).norm();
}

Pose ApplyAlignment(const Pose& pose, const Alignment& a) {
    Pose out = pose;
    out.p = a.R * pose.p + a.t;
    out.q = (Eigen::Quaterniond(a.R) * pose.q).normalized();
    return out;
}

double RotationAngleDeg(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd aa(R);
    return std::abs(aa.angle()) * kRadToDeg;
}

double WrapDeg(double deg) {
    while (deg > 180.0) deg -= 360.0;
    while (deg < -180.0) deg += 360.0;
    return deg;
}

ErrorSample ComputeError(const Pose& est, const Pose& gt) {
    ErrorSample e;

    // --------------------------------------------------------
    // Translation residual in the common evaluation world frame.
    //
    //     dp = p_est_aligned - p_gt
    //
    // Therefore x/y/z keep their sign.
    // --------------------------------------------------------
    const Eigen::Vector3d dp = est.p - gt.p;

    e.trans_m = dp.norm();
    e.x_m = dp.x();
    e.y_m = dp.y();
    e.z_m = dp.z();

    // --------------------------------------------------------
    // Rotation residual:
    //
    //     R_err = R_gt^T * R_est
    //
    // Extract signed ZYX Euler residual:
    //     R_err = Rz(yaw) * Ry(pitch) * Rx(roll)
    //
    // These component plots are diagnostics; rot_deg below
    // remains the coordinate-independent rotation-angle APE.
    // --------------------------------------------------------
    const Eigen::Matrix3d R_err =
        gt.q.toRotationMatrix().transpose() *
        est.q.toRotationMatrix();

    e.rot_deg = RotationAngleDeg(R_err);

    const double sin_pitch =
        std::clamp(
            -R_err(2, 0),
            -1.0,
            1.0);

    e.roll_deg = WrapDeg(
        std::atan2(
            R_err(2, 1),
            R_err(2, 2)) *
        kRadToDeg);

    e.pitch_deg =
        std::asin(sin_pitch) *
        kRadToDeg;

    e.yaw_deg = WrapDeg(
        std::atan2(
            R_err(1, 0),
            R_err(0, 0)) *
        kRadToDeg);

    return e;
}

double QuantileSorted(const std::vector<double>& sorted, double q) {
    if (sorted.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted.size() == 1) {
        return sorted.front();
    }
    const double pos = q * static_cast<double>(sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double a = pos - static_cast<double>(lo);
    return (1.0 - a) * sorted[lo] + a * sorted[hi];
}

Stats ComputeStats(const std::vector<double>& values) {
    Stats s;
    s.n = values.size();
    if (values.empty()) {
        return s;
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    for (double v : values) {
        sum += v;
        sum_sq += v * v;
    }
    s.mean = sum / static_cast<double>(values.size());
    s.rmse = std::sqrt(sum_sq / static_cast<double>(values.size()));

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    s.median = QuantileSorted(sorted, 0.5);
    s.p95 = QuantileSorted(sorted, 0.95);
    s.max = sorted.back();
    return s;
}

double PathLength(const std::vector<Pose>& poses) {
    double length = 0.0;
    for (std::size_t i = 1; i < poses.size(); ++i) {
        length += (poses[i].p - poses[i - 1].p).norm();
    }
    return length;
}

double MatchedGtPathLength(const std::vector<MatchedPose>& matched) {
    double length = 0.0;
    for (std::size_t i = 1; i < matched.size(); ++i) {
        length += (matched[i].gt_lidar.p - matched[i - 1].gt_lidar.p).norm();
    }
    return length;
}

struct RelativePose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d p = Eigen::Vector3d::Zero();
};

RelativePose Relative(const Pose& a, const Pose& b) {
    RelativePose r;
    const Eigen::Matrix3d R_a = a.q.toRotationMatrix();
    r.R = R_a.transpose() * b.q.toRotationMatrix();
    r.p = R_a.transpose() * (b.p - a.p);
    return r;
}

RpeStats ComputeRpe(const std::vector<MatchedPose>& matched, double target_distance_m) {
    RpeStats result;
    result.target_distance_m = target_distance_m;

    if (matched.size() < 2) {
        return result;
    }

    std::vector<double> cumulative(matched.size(), 0.0);
    for (std::size_t i = 1; i < matched.size(); ++i) {
        cumulative[i] = cumulative[i - 1] +
            (matched[i].gt_lidar.p - matched[i - 1].gt_lidar.p).norm();
    }

    std::vector<double> trans_errors;
    std::vector<double> rot_errors;

    for (std::size_t i = 0; i + 1 < matched.size(); ++i) {
        const double target = cumulative[i] + target_distance_m;
        const auto it = std::lower_bound(
            cumulative.begin() + static_cast<std::ptrdiff_t>(i + 1),
            cumulative.end(), target);
        if (it == cumulative.end()) {
            continue;
        }
        const std::size_t j = static_cast<std::size_t>(
            std::distance(cumulative.begin(), it));

        const RelativePose gt_rel = Relative(matched[i].gt_lidar, matched[j].gt_lidar);
        const RelativePose est_rel = Relative(matched[i].est_raw, matched[j].est_raw);

        const Eigen::Matrix3d R_err = gt_rel.R.transpose() * est_rel.R;
        const Eigen::Vector3d p_err =
            gt_rel.R.transpose() * (est_rel.p - gt_rel.p);

        trans_errors.push_back(p_err.norm());
        rot_errors.push_back(RotationAngleDeg(R_err));
    }

    result.n = trans_errors.size();
    result.translation = ComputeStats(trans_errors);
    result.rotation_deg = ComputeStats(rot_errors);
    return result;
}

std::string ShellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::string GnuplotQuote(const fs::path& p) {
    std::string s = p.string();
    std::string out = "'";
    for (char c : s) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '\'') {
            out += "\\'";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

void WriteStatsText(std::ostream& out, const std::string& title, const Stats& s, const std::string& unit) {
    out << title << "\n";
    out << "  N      : " << s.n << "\n";
    out << "  RMSE   : " << s.rmse << " " << unit << "\n";
    out << "  Mean   : " << s.mean << " " << unit << "\n";
    out << "  Median : " << s.median << " " << unit << "\n";
    out << "  P95    : " << s.p95 << " " << unit << "\n";
    out << "  Max    : " << s.max << " " << unit << "\n";
}

void WriteMetricCsv(std::ofstream& out, const std::string& key, double value, const std::string& unit) {
    out << key << "," << std::setprecision(15) << value << "," << unit << "\n";
}

void WriteMetricCsv(std::ofstream& out, const std::string& key, std::size_t value, const std::string& unit) {
    out << key << "," << value << "," << unit << "\n";
}

void WritePlotScript(
    const fs::path& script_path,
    const fs::path& trajectory_csv,
    const fs::path& rpe_csv,
    const fs::path& plots_dir) {
    std::ofstream gp(script_path);
    if (!gp) {
        throw std::runtime_error("Cannot write gnuplot script: " + script_path.string());
    }

    const std::string data = GnuplotQuote(trajectory_csv);
    const std::string rpe = GnuplotQuote(rpe_csv);
    const auto png = [&](const std::string& name) {
        return GnuplotQuote(plots_dir / name);
    };

    // Important: explicitly reset aspect-ratio / 3D state between figures.
    // The old script used `set size ratio -1` for XY and the state leaked into
    // later figures, which compressed APE/RPE plots into a thin horizontal band.
    gp << "set datafile separator ','\n";
    gp << "set terminal pngcairo size 1600,900 enhanced font 'Sans,16'\n";
    gp << "set border linewidth 1.2\n";
    gp << "set tics out nomirror\n";
    gp << "set grid xtics ytics back lw 1 dt 3\n";
    gp << "set key top center horizontal samplen 2 spacing 1.2\n";
    gp << "set style line 1 lw 2.6\n";
    gp << "set style line 2 lw 2.4\n";
    gp << "set style line 3 lw 2.4\n";

    // XY: readable presentation plot. Do NOT force equal physical scale here;
    // long agricultural trajectories otherwise become a very thin strip.
    gp << "set output " << png("trajectory_xy.png") << "\n";
    gp << "set title 'FR-SLAM Pure LIO vs Ground Truth - XY'\n";
    gp << "set xlabel 'X [m]'\nset ylabel 'Y [m]'\n";
    gp << "set size noratio\nset autoscale\n";
    gp << "set margins 10,4,6,5\n";
    gp << "plot " << data << " every ::1 using 6:7 with lines ls 1 title 'GT LiDAR', \\\n"
       << "     " << data << " every ::1 using 9:10 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 12:13 with lines ls 3 title 'SE(3) aligned'\n";

    // A second XY figure keeps true metric aspect ratio for geometric inspection.
    gp << "set output " << png("trajectory_xy_equal.png") << "\n";
    gp << "set title 'FR-SLAM Pure LIO vs Ground Truth - XY (equal metric scale)'\n";
    gp << "set xlabel 'X [m]'\nset ylabel 'Y [m]'\n";
    gp << "set size ratio -1\nset autoscale\n";
    gp << "plot " << data << " every ::1 using 6:7 with lines ls 1 title 'GT LiDAR', \\\n"
       << "     " << data << " every ::1 using 9:10 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 12:13 with lines ls 3 title 'SE(3) aligned'\n";

    // Lab-style XY rigid alignment used for direct mathematical comparison
    // with the laboratory paper (same-run fit here; scale fixed to 1).
    gp << "set output " << png("trajectory_xy_lab.png") << "\n";
    gp << "set title 'GT vs Lab-style XY Alignment (SE(2), scale=1)'\n";
    gp << "set xlabel 'X [m]'\nset ylabel 'Y [m]'\n";
    gp << "set size noratio\nset autoscale\n";
    gp << "set margins 10,4,6,5\n";
    gp << "plot " << data << " every ::1 using 6:7 with lines ls 1 title 'GT LiDAR', \\\n"
       << "     " << data << " every ::1 using 23:24 with lines ls 3 title 'Lab-style XY aligned'\n";

    gp << "set output " << png("xy_ate_lab.png") << "\n";
    gp << "set title 'Lab-style XY ATE vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'XY ATE [m]'\n";
    gp << "set size noratio\nset autoscale x\nset yrange [0:*]\n";
    gp << "set format y '%.2f'\n";
    gp << "plot " << data << " every ::1 using 2:25 with lines ls 3 title 'XY ATE'\n";
    gp << "set format y '%g'\n";

    // XZ: intentionally use non-equal axes so centimeter/decimeter height drift
    // is visible across a tens-of-meters horizontal trajectory.
    gp << "set output " << png("trajectory_xz.png") << "\n";
    gp << "set title 'FR-SLAM Pure LIO vs Ground Truth - XZ'\n";
    gp << "set xlabel 'X [m]'\nset ylabel 'Z [m]'\n";
    gp << "set size noratio\nset autoscale\n";
    gp << "set format y '%.2f'\n";
    gp << "plot " << data << " every ::1 using 6:8 with lines ls 1 title 'GT LiDAR', \\\n"
       << "     " << data << " every ::1 using 9:11 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 12:14 with lines ls 3 title 'SE(3) aligned'\n";
    gp << "set format y '%g'\n";

    // 3D: use a readable box rather than equal xyz scaling; equal xyz makes this
    // particular long/flat agricultural trajectory visually collapse.
    gp << "set output " << png("trajectory_3d.png") << "\n";
    gp << "set title 'FR-SLAM Pure LIO vs Ground Truth - 3D'\n";
    gp << "set xlabel 'X [m]'\nset ylabel 'Y [m]'\nset zlabel 'Z [m]' offset 1,0\n";
    gp << "set size noratio\nset autoscale\n";
    gp << "set view 62,28,1.0,1.35\n";
    gp << "set xyplane relative 0\n";
    gp << "set grid xtics ytics ztics back lw 1 dt 3\n";
    gp << "splot " << data << " every ::1 using 6:7:8 with lines ls 1 title 'GT LiDAR', \\\n"
       << "      " << data << " every ::1 using 9:10:11 with lines ls 2 title 'First-pose aligned', \\\n"
       << "      " << data << " every ::1 using 12:13:14 with lines ls 3 title 'SE(3) aligned'\n";

    // Return to clean 2D state before all time-series plots.
    gp << "set grid xtics ytics back lw 1 dt 3\n";
    gp << "set size noratio\n";
    gp << "set key top center horizontal\n";
    gp << "set margins 10,4,6,5\n";

    gp << "set output " << png("translation_ape.png") << "\n";
    gp << "set title 'Translation APE vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Translation error [m]'\n";
    gp << "set autoscale x\nset yrange [0:*]\nset format y '%.2f'\n";
    gp << "plot " << data << " every ::1 using 2:15 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 2:19 with lines ls 3 title 'SE(3) aligned'\n";

    gp << "set output " << png("rotation_ape.png") << "\n";
    gp << "set title 'Rotation APE vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Rotation error [deg]'\n";
    gp << "set autoscale x\nset yrange [0:*]\nset format y '%.2f'\n";
    gp << "plot " << data << " every ::1 using 2:16 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 2:20 with lines ls 3 title 'SE(3) aligned'\n";

    gp << "set output " << png("z_error.png") << "\n";
    gp << "set title 'Signed Z Error vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Z error [m]'\n";
    gp << "set autoscale\nset format y '%.2f'\nset yzeroaxis lw 1 dt 2\n";
    gp << "plot " << data << " every ::1 using 2:17 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 2:21 with lines ls 3 title 'SE(3) aligned'\n";

    gp << "set output " << png("yaw_error.png") << "\n";
    gp << "set title 'Signed Yaw Error vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Yaw error [deg]'\n";
    gp << "set autoscale\nset format y '%.2f'\nset yzeroaxis lw 1 dt 2\n";
    gp << "plot " << data << " every ::1 using 2:18 with lines ls 2 title 'First-pose aligned', \\\n"
       << "     " << data << " every ::1 using 2:22 with lines ls 3 title 'SE(3) aligned'\n";

    // ========================================================
    // Signed XYZ residuals.
    // FIRST-pose alignment preserves accumulated trajectory
    // drift and is therefore the most useful diagnostic view.
    // ========================================================

    gp << "set output " << png("first_xyz_residual.png") << "\n";
    gp << "set title 'FIRST-pose Signed XYZ Residual vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Position residual [m]'\n";
    gp << "set autoscale\nset format y '%.2f'\nset yzeroaxis lw 1 dt 2\n";
    gp << "plot " << data << " every ::1 using 2:26 with lines lw 2.2 title 'X residual', \\\n"
       << "     " << data << " every ::1 using 2:27 with lines lw 2.2 title 'Y residual', \\\n"
       << "     " << data << " every ::1 using 2:17 with lines lw 2.2 title 'Z residual'\n";

    gp << "set output " << png("se3_xyz_residual.png") << "\n";
    gp << "set title 'GLOBAL SE(3) Signed XYZ Residual vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Position residual [m]'\n";
    gp << "set autoscale\nset format y '%.2f'\nset yzeroaxis lw 1 dt 2\n";
    gp << "plot " << data << " every ::1 using 2:28 with lines lw 2.2 title 'X residual', \\\n"
       << "     " << data << " every ::1 using 2:29 with lines lw 2.2 title 'Y residual', \\\n"
       << "     " << data << " every ::1 using 2:21 with lines lw 2.2 title 'Z residual'\n";

    // ========================================================
    // Signed RPY residuals.
    //
    // These are ZYX Euler diagnostic components of
    // R_gt^T * R_est. The existing rot_deg metric remains the
    // invariant SO(3) angular APE.
    // ========================================================

    gp << "set output " << png("first_rpy_residual.png") << "\n";
    gp << "set title 'FIRST-pose Signed RPY Residual vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Attitude residual [deg]'\n";
    gp << "set autoscale\nset format y '%.2f'\nset yzeroaxis lw 1 dt 2\n";
    gp << "plot " << data << " every ::1 using 2:30 with lines lw 2.2 title 'Roll residual', \\\n"
       << "     " << data << " every ::1 using 2:31 with lines lw 2.2 title 'Pitch residual', \\\n"
       << "     " << data << " every ::1 using 2:18 with lines lw 2.2 title 'Yaw residual'\n";

    gp << "set output " << png("se3_rpy_residual.png") << "\n";
    gp << "set title 'GLOBAL SE(3) Signed RPY Residual vs Time'\n";
    gp << "set xlabel 'Elapsed time [s]'\nset ylabel 'Attitude residual [deg]'\n";
    gp << "set autoscale\nset format y '%.2f'\nset yzeroaxis lw 1 dt 2\n";
    gp << "plot " << data << " every ::1 using 2:32 with lines lw 2.2 title 'Roll residual', \\\n"
       << "     " << data << " every ::1 using 2:33 with lines lw 2.2 title 'Pitch residual', \\\n"
       << "     " << data << " every ::1 using 2:22 with lines lw 2.2 title 'Yaw residual'\n";

    // RPE: explicitly clear the previous title/state so it cannot leak into the
    // multiplot panels.
    gp << "unset title\n";
    gp << "unset yzeroaxis\n";
    gp << "set format y '%g'\n";
    gp << "set output " << png("rpe.png") << "\n";
    gp << "set multiplot layout 2,1 rowsfirst title 'Distance-based Relative Pose Error' font ',18'\n";
    gp << "set size noratio\nset key top left\n";
    gp << "set title 'Translation RPE'\n";
    gp << "set xlabel 'Interval [m]'\nset ylabel 'Translation RMSE [m]'\n";
    gp << "set xrange [0.5:10.5]\nset yrange [0:*]\nset format y '%.3f'\n";
    gp << "plot " << rpe << " every ::1 using 1:3 with linespoints lw 2.5 pt 7 ps 1.2 title 'RMSE'\n";
    gp << "set title 'Rotation RPE'\n";
    gp << "set xlabel 'Interval [m]'\nset ylabel 'Rotation RMSE [deg]'\n";
    gp << "set xrange [0.5:10.5]\nset yrange [0:*]\nset format y '%.2f'\n";
    gp << "plot " << rpe << " every ::1 using 1:8 with linespoints lw 2.5 pt 7 ps 1.2 title 'RMSE'\n";
    gp << "unset multiplot\n";
}

void PrintUsage(const char* argv0) {
    std::cerr
        << "Usage:\n  " << argv0
        << " --est frontend_trajectory.csv"
        << " --gt GT_trajectory.csv"
        << " --output output_directory"
        << " [--extrinsic hortimulti|identity]"
        << " [--max-gt-gap 0.5]"
        << " [--no-plots]\n";
}

Options ParseOptions(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value after " + name);
            }
            return argv[++i];
        };

        if (arg == "--est") {
            o.est_path = require_value(arg);
        } else if (arg == "--gt") {
            o.gt_path = require_value(arg);
        } else if (arg == "--output") {
            o.output_dir = require_value(arg);
        } else if (arg == "--extrinsic") {
            o.extrinsic = require_value(arg);
        } else if (arg == "--max-gt-gap") {
            o.max_gt_gap_s = std::stod(require_value(arg));
        } else if (arg == "--no-plots") {
            o.make_plots = false;
        } else if (arg == "-h" || arg == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (o.est_path.empty() || o.gt_path.empty() || o.output_dir.empty()) {
        throw std::runtime_error("--est, --gt and --output are required");
    }
    if (!(o.max_gt_gap_s > 0.0)) {
        throw std::runtime_error("--max-gt-gap must be > 0");
    }
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);

        const std::vector<Pose> est = LoadPoseCsv(options.est_path);
        const std::vector<Pose> gt_os = LoadPoseCsv(options.gt_path);
        const Extrinsic T_OL = GetExtrinsic(options.extrinsic);

        std::vector<MatchedPose> matched;
        matched.reserve(est.size());
        std::size_t rejected_outside_range = 0;
        std::size_t rejected_large_gap = 0;

        for (const Pose& e : est) {
            Pose gt_interp_os;
            bool rejected_gap = false;
            if (!InterpolatePose(
                    gt_os,
                    e.t,
                    options.max_gt_gap_s,
                    gt_interp_os,
                    rejected_gap)) {
                if (rejected_gap) {
                    ++rejected_large_gap;
                } else {
                    ++rejected_outside_range;
                }
                continue;
            }
            matched.push_back({e, ApplyBodyToSensorExtrinsic(gt_interp_os, T_OL)});
        }

        if (matched.size() < 3) {
            throw std::runtime_error("Too few matched poses after timestamp association");
        }

        const Alignment first_align =
            FirstPoseAlignment(matched.front().est_raw, matched.front().gt_lidar);
        const Alignment se3_align = GlobalSe3Alignment(matched);
        const Alignment2D lab_xy_align = GlobalSe2XyAlignment(matched);

        std::vector<Pose> est_first;
        std::vector<Pose> est_se3;
        est_first.reserve(matched.size());
        est_se3.reserve(matched.size());

        std::vector<ErrorSample> first_errors;
        std::vector<ErrorSample> se3_errors;
        first_errors.reserve(matched.size());
        se3_errors.reserve(matched.size());

        std::vector<double> first_trans, first_rot, first_z_abs, first_yaw_abs;
        std::vector<double> se3_trans, se3_rot, se3_z_abs, se3_yaw_abs;
        std::vector<Eigen::Vector2d> est_lab_xy;
        std::vector<double> lab_xy_errors;
        first_trans.reserve(matched.size());
        first_rot.reserve(matched.size());
        first_z_abs.reserve(matched.size());
        first_yaw_abs.reserve(matched.size());
        se3_trans.reserve(matched.size());
        se3_rot.reserve(matched.size());
        se3_z_abs.reserve(matched.size());
        se3_yaw_abs.reserve(matched.size());
        est_lab_xy.reserve(matched.size());
        lab_xy_errors.reserve(matched.size());

        for (const MatchedPose& m : matched) {
            const Pose f = ApplyAlignment(m.est_raw, first_align);
            const Pose s = ApplyAlignment(m.est_raw, se3_align);
            const Eigen::Vector2d lxy =
                ApplyAlignment2D(m.est_raw.p, lab_xy_align);
            const ErrorSample ef = ComputeError(f, m.gt_lidar);
            const ErrorSample es = ComputeError(s, m.gt_lidar);
            const double lab_xy_error = XyErrorM(lxy, m.gt_lidar.p);

            est_first.push_back(f);
            est_se3.push_back(s);
            first_errors.push_back(ef);
            se3_errors.push_back(es);

            first_trans.push_back(ef.trans_m);
            first_rot.push_back(ef.rot_deg);
            first_z_abs.push_back(std::abs(ef.z_m));
            first_yaw_abs.push_back(std::abs(ef.yaw_deg));
            se3_trans.push_back(es.trans_m);
            se3_rot.push_back(es.rot_deg);
            se3_z_abs.push_back(std::abs(es.z_m));
            se3_yaw_abs.push_back(std::abs(es.yaw_deg));
            est_lab_xy.push_back(lxy);
            lab_xy_errors.push_back(lab_xy_error);
        }

        const Stats first_trans_stats = ComputeStats(first_trans);
        const Stats first_rot_stats = ComputeStats(first_rot);
        const Stats first_z_stats = ComputeStats(first_z_abs);
        const Stats first_yaw_stats = ComputeStats(first_yaw_abs);
        const Stats se3_trans_stats = ComputeStats(se3_trans);
        const Stats se3_rot_stats = ComputeStats(se3_rot);
        const Stats se3_z_stats = ComputeStats(se3_z_abs);
        const Stats se3_yaw_stats = ComputeStats(se3_yaw_abs);
        const Stats lab_xy_stats = ComputeStats(lab_xy_errors);

        const std::vector<double> rpe_distances = {1.0, 5.0, 10.0};
        std::vector<RpeStats> rpe_results;
        for (double d : rpe_distances) {
            rpe_results.push_back(ComputeRpe(matched, d));
        }

        // ============================================================
        // HORTIMULTI BENCHMARK METRICS
        //
        // ATE [m] : global SE(3), scale=1, translation APE RMSE.
        // RPE [%] : 1 m translational RPE RMSE / 1 m * 100.
        //
        // NOTE: HortiMulti reports ATE [m] and RPE [%] and cites
        // Sturm et al. The exact unpublished evaluation script is
        // not assumed here; the RPE normalization is stated explicitly.
        // ============================================================
        const RpeStats& hortimulti_rpe = rpe_results.front();
        const double hortimulti_ate_rmse_m = se3_trans_stats.rmse;
        const double hortimulti_rpe_percent =
            100.0 * hortimulti_rpe.translation.rmse /
            hortimulti_rpe.target_distance_m;

        fs::create_directories(options.output_dir);
        const fs::path plots_dir = options.output_dir / "plots";
        if (options.make_plots) {
            fs::create_directories(plots_dir);
        }

        const double eval_duration_s =
            matched.back().est_raw.t - matched.front().est_raw.t;
        const double eval_gt_path_m = MatchedGtPathLength(matched);
        std::vector<Pose> gt_lidar_full;
        gt_lidar_full.reserve(gt_os.size());
        for (const Pose& p : gt_os) {
            gt_lidar_full.push_back(ApplyBodyToSensorExtrinsic(p, T_OL));
        }
        const double full_gt_path_m = PathLength(gt_lidar_full);

        const double est_closure_m =
            (matched.back().est_raw.p - matched.front().est_raw.p).norm();
        const double gt_closure_m =
            (matched.back().gt_lidar.p - matched.front().gt_lidar.p).norm();

        const ErrorSample final_first = first_errors.back();
        const ErrorSample final_se3 = se3_errors.back();
        const double endpoint_drift_percent =
            eval_gt_path_m > 1e-12 ? 100.0 * final_first.trans_m / eval_gt_path_m
                                  : std::numeric_limits<double>::quiet_NaN();

        const fs::path trajectory_csv = options.output_dir / "matched_trajectory.csv";
        {
            std::ofstream out(trajectory_csv);
            if (!out) {
                throw std::runtime_error("Cannot write: " + trajectory_csv.string());
            }
            out << std::setprecision(15);
            out << "timestamp,elapsed_s,"
                << "est_raw_x,est_raw_y,est_raw_z,"
                << "gt_x,gt_y,gt_z,"
                << "first_x,first_y,first_z,"
                << "se3_x,se3_y,se3_z,"
                << "first_trans_err_m,first_rot_err_deg,first_z_err_m,first_yaw_err_deg,"
                << "se3_trans_err_m,se3_rot_err_deg,se3_z_err_m,se3_yaw_err_deg,"
                << "lab_xy_x,lab_xy_y,lab_xy_ate_m,"
                << "first_x_err_m,first_y_err_m,"
                << "se3_x_err_m,se3_y_err_m,"
                << "first_roll_err_deg,first_pitch_err_deg,"
                << "se3_roll_err_deg,se3_pitch_err_deg\n";
            const double t0 = matched.front().est_raw.t;
            for (std::size_t i = 0; i < matched.size(); ++i) {
                out << matched[i].est_raw.t << ","
                    << matched[i].est_raw.t - t0 << ","
                    << matched[i].est_raw.p.x() << ","
                    << matched[i].est_raw.p.y() << ","
                    << matched[i].est_raw.p.z() << ","
                    << matched[i].gt_lidar.p.x() << ","
                    << matched[i].gt_lidar.p.y() << ","
                    << matched[i].gt_lidar.p.z() << ","
                    << est_first[i].p.x() << ","
                    << est_first[i].p.y() << ","
                    << est_first[i].p.z() << ","
                    << est_se3[i].p.x() << ","
                    << est_se3[i].p.y() << ","
                    << est_se3[i].p.z() << ","
                    << first_errors[i].trans_m << ","
                    << first_errors[i].rot_deg << ","
                    << first_errors[i].z_m << ","
                    << first_errors[i].yaw_deg << ","
                    << se3_errors[i].trans_m << ","
                    << se3_errors[i].rot_deg << ","
                    << se3_errors[i].z_m << ","
                    << se3_errors[i].yaw_deg << ","
                    << est_lab_xy[i].x() << ","
                    << est_lab_xy[i].y() << ","
                    << lab_xy_errors[i] << ","
                    << first_errors[i].x_m << ","
                    << first_errors[i].y_m << ","
                    << se3_errors[i].x_m << ","
                    << se3_errors[i].y_m << ","
                    << first_errors[i].roll_deg << ","
                    << first_errors[i].pitch_deg << ","
                    << se3_errors[i].roll_deg << ","
                    << se3_errors[i].pitch_deg << "\n";
            }
        }

        const fs::path errors_csv = options.output_dir / "errors.csv";
        {
            std::ofstream out(errors_csv);
            if (!out) {
                throw std::runtime_error("Cannot write: " + errors_csv.string());
            }
            out << std::setprecision(15);
            out << "timestamp,elapsed_s,"
                << "first_trans_err_m,first_rot_err_deg,first_z_err_m,first_yaw_err_deg,"
                << "se3_trans_err_m,se3_rot_err_deg,se3_z_err_m,se3_yaw_err_deg,"
                << "lab_xy_ate_m,"
                << "first_x_err_m,first_y_err_m,"
                << "se3_x_err_m,se3_y_err_m,"
                << "first_roll_err_deg,first_pitch_err_deg,"
                << "se3_roll_err_deg,se3_pitch_err_deg\n";
            const double t0 = matched.front().est_raw.t;
            for (std::size_t i = 0; i < matched.size(); ++i) {
                out << matched[i].est_raw.t << ","
                    << matched[i].est_raw.t - t0 << ","
                    << first_errors[i].trans_m << ","
                    << first_errors[i].rot_deg << ","
                    << first_errors[i].z_m << ","
                    << first_errors[i].yaw_deg << ","
                    << se3_errors[i].trans_m << ","
                    << se3_errors[i].rot_deg << ","
                    << se3_errors[i].z_m << ","
                    << se3_errors[i].yaw_deg << ","
                    << lab_xy_errors[i] << ","
                    << first_errors[i].x_m << ","
                    << first_errors[i].y_m << ","
                    << se3_errors[i].x_m << ","
                    << se3_errors[i].y_m << ","
                    << first_errors[i].roll_deg << ","
                    << first_errors[i].pitch_deg << ","
                    << se3_errors[i].roll_deg << ","
                    << se3_errors[i].pitch_deg << "\n";
            }
        }

        const fs::path rpe_csv = options.output_dir / "rpe.csv";
        {
            std::ofstream out(rpe_csv);
            if (!out) {
                throw std::runtime_error("Cannot write: " + rpe_csv.string());
            }
            out << std::setprecision(15);
            out << "distance_m,count,"
                << "trans_rmse_m,trans_mean_m,trans_median_m,trans_p95_m,trans_max_m,"
                << "rot_rmse_deg,rot_mean_deg,rot_median_deg,rot_p95_deg,rot_max_deg\n";
            for (const RpeStats& r : rpe_results) {
                out << r.target_distance_m << "," << r.n << ","
                    << r.translation.rmse << ","
                    << r.translation.mean << ","
                    << r.translation.median << ","
                    << r.translation.p95 << ","
                    << r.translation.max << ","
                    << r.rotation_deg.rmse << ","
                    << r.rotation_deg.mean << ","
                    << r.rotation_deg.median << ","
                    << r.rotation_deg.p95 << ","
                    << r.rotation_deg.max << "\n";
            }
        }

        const fs::path summary_txt = options.output_dir / "summary.txt";
        {
            std::ofstream out(summary_txt);
            if (!out) {
                throw std::runtime_error("Cannot write: " + summary_txt.string());
            }
            out << std::fixed << std::setprecision(6);
            out << "========== FR-SLAM LIO GT EVALUATION ==========\n\n";
            out << "Input\n";
            out << "  Estimator CSV              : " << options.est_path << "\n";
            out << "  Ground-truth CSV           : " << options.gt_path << "\n";
            out << "  GT->LiDAR extrinsic mode   : " << options.extrinsic << "\n";
            out << "  Estimator poses            : " << est.size() << "\n";
            out << "  GT poses                   : " << gt_os.size() << "\n";
            out << "  Matched poses              : " << matched.size() << "\n";
            out << "  Rejected outside GT range  : " << rejected_outside_range << "\n";
            out << "  Rejected large GT gap      : " << rejected_large_gap << "\n";
            out << "  Max GT interpolation gap   : " << options.max_gt_gap_s << " s\n";
            out << "  First matched timestamp    : " << matched.front().est_raw.t << "\n";
            out << "  Last matched timestamp     : " << matched.back().est_raw.t << "\n";
            out << "  Evaluation duration        : " << eval_duration_s << " s\n";
            out << "  Evaluation GT path length  : " << eval_gt_path_m << " m\n";
            out << "  Full GT path length        : " << full_gt_path_m << " m\n\n";

            out << "Closure over matched interval\n";
            out << "  GT start-to-end distance   : " << gt_closure_m << " m\n";
            out << "  EST start-to-end distance  : " << est_closure_m << " m\n\n";

            out << "HORTIMULTI BENCHMARK\n";
            out << "  ATE RMSE                   : "
                << hortimulti_ate_rmse_m << " m\n";
            out << "  RPE translation            : "
                << hortimulti_rpe_percent << " %\n";
            out << "  RPE source distance        : "
                << hortimulti_rpe.target_distance_m << " m\n";
            out << "  RPE samples                : "
                << hortimulti_rpe.n << "\n";
            out << "  RPE definition             : translational RMSE / travelled distance * 100\n\n";

            out << "FIRST-POSE ALIGNMENT\n";
            WriteStatsText(out, "Translation APE", first_trans_stats, "m");
            WriteStatsText(out, "Rotation APE", first_rot_stats, "deg");
            WriteStatsText(out, "|Z error|", first_z_stats, "m");
            WriteStatsText(out, "|Yaw error|", first_yaw_stats, "deg");
            out << "Final translation error      : " << final_first.trans_m << " m\n";
            out << "Final rotation error         : " << final_first.rot_deg << " deg\n";
            out << "Final signed Z error         : " << final_first.z_m << " m\n";
            out << "Final signed yaw error       : " << final_first.yaw_deg << " deg\n";
            out << "Endpoint drift / path length : " << endpoint_drift_percent << " %\n\n";

            out << "GLOBAL SE(3) ALIGNMENT (scale = 1)\n";
            WriteStatsText(out, "Translation APE", se3_trans_stats, "m");
            WriteStatsText(out, "Rotation APE", se3_rot_stats, "deg");
            WriteStatsText(out, "|Z error|", se3_z_stats, "m");
            WriteStatsText(out, "|Yaw error|", se3_yaw_stats, "deg");
            out << "Final translation error      : " << final_se3.trans_m << " m\n";
            out << "Final rotation error         : " << final_se3.rot_deg << " deg\n";
            out << "Final signed Z error         : " << final_se3.z_m << " m\n";
            out << "Final signed yaw error       : " << final_se3.yaw_deg << " deg\n\n";

            out << "LAB-STYLE GLOBAL XY ALIGNMENT (SE(2), scale = 1, same-run fit)\n";
            WriteStatsText(out, "XY ATE", lab_xy_stats, "m");
            out << "  Alignment yaw              : "
                << lab_xy_align.yaw_rad * kRadToDeg << " deg\n";
            out << "  Alignment tx               : " << lab_xy_align.t.x() << " m\n";
            out << "  Alignment ty               : " << lab_xy_align.t.y() << " m\n";
            out << "  Note                       : Same XY rigid-alignment mathematics as the lab paper;\n";
            out << "                               this evaluator fits on the evaluated run itself and removes no outliers.\n\n";

            out << "DISTANCE-BASED RPE (alignment-invariant)\n";
            for (const RpeStats& r : rpe_results) {
                out << "RPE @ " << r.target_distance_m << " m (N=" << r.n << ")\n";
                out << "  Translation RMSE : " << r.translation.rmse << " m\n";
                out << "  Translation P95  : " << r.translation.p95 << " m\n";
                out << "  Rotation RMSE    : " << r.rotation_deg.rmse << " deg\n";
                out << "  Rotation P95     : " << r.rotation_deg.p95 << " deg\n";
            }
        }

        const fs::path summary_csv = options.output_dir / "summary.csv";
        {
            std::ofstream out(summary_csv);
            if (!out) {
                throw std::runtime_error("Cannot write: " + summary_csv.string());
            }
            out << "metric,value,unit\n";
            WriteMetricCsv(out, "estimator_pose_count", est.size(), "count");
            WriteMetricCsv(out, "gt_pose_count", gt_os.size(), "count");
            WriteMetricCsv(out, "matched_pose_count", matched.size(), "count");
            WriteMetricCsv(out, "rejected_outside_gt_range", rejected_outside_range, "count");
            WriteMetricCsv(out, "rejected_large_gt_gap", rejected_large_gap, "count");
            WriteMetricCsv(out, "evaluation_duration", eval_duration_s, "s");
            WriteMetricCsv(out, "evaluation_gt_path_length", eval_gt_path_m, "m");
            WriteMetricCsv(out, "full_gt_path_length", full_gt_path_m, "m");
            WriteMetricCsv(out, "gt_closure_distance", gt_closure_m, "m");
            WriteMetricCsv(out, "est_closure_distance", est_closure_m, "m");

            WriteMetricCsv(out, "first_trans_ape_rmse", first_trans_stats.rmse, "m");
            WriteMetricCsv(out, "first_trans_ape_mean", first_trans_stats.mean, "m");
            WriteMetricCsv(out, "first_trans_ape_median", first_trans_stats.median, "m");
            WriteMetricCsv(out, "first_trans_ape_p95", first_trans_stats.p95, "m");
            WriteMetricCsv(out, "first_trans_ape_max", first_trans_stats.max, "m");
            WriteMetricCsv(out, "first_rot_ape_rmse", first_rot_stats.rmse, "deg");
            WriteMetricCsv(out, "first_rot_ape_p95", first_rot_stats.p95, "deg");
            WriteMetricCsv(out, "first_z_abs_rmse", first_z_stats.rmse, "m");
            WriteMetricCsv(out, "first_z_abs_p95", first_z_stats.p95, "m");
            WriteMetricCsv(out, "first_yaw_abs_rmse", first_yaw_stats.rmse, "deg");
            WriteMetricCsv(out, "first_yaw_abs_p95", first_yaw_stats.p95, "deg");
            WriteMetricCsv(out, "first_final_translation_error", final_first.trans_m, "m");
            WriteMetricCsv(out, "first_final_rotation_error", final_first.rot_deg, "deg");
            WriteMetricCsv(out, "first_final_z_error", final_first.z_m, "m");
            WriteMetricCsv(out, "first_final_yaw_error", final_first.yaw_deg, "deg");
            WriteMetricCsv(out, "endpoint_drift_over_path", endpoint_drift_percent, "%");

            WriteMetricCsv(out, "se3_trans_ape_rmse", se3_trans_stats.rmse, "m");
            WriteMetricCsv(out, "se3_trans_ape_mean", se3_trans_stats.mean, "m");
            WriteMetricCsv(out, "se3_trans_ape_median", se3_trans_stats.median, "m");
            WriteMetricCsv(out, "se3_trans_ape_p95", se3_trans_stats.p95, "m");
            WriteMetricCsv(out, "se3_trans_ape_max", se3_trans_stats.max, "m");
            WriteMetricCsv(out, "se3_rot_ape_rmse", se3_rot_stats.rmse, "deg");
            WriteMetricCsv(out, "se3_rot_ape_p95", se3_rot_stats.p95, "deg");
            WriteMetricCsv(out, "se3_z_abs_rmse", se3_z_stats.rmse, "m");
            WriteMetricCsv(out, "se3_yaw_abs_rmse", se3_yaw_stats.rmse, "deg");

            WriteMetricCsv(out, "lab_xy_ate_rmse", lab_xy_stats.rmse, "m");
            WriteMetricCsv(out, "lab_xy_ate_mean", lab_xy_stats.mean, "m");
            WriteMetricCsv(out, "lab_xy_ate_median", lab_xy_stats.median, "m");
            WriteMetricCsv(out, "lab_xy_ate_p95", lab_xy_stats.p95, "m");
            WriteMetricCsv(out, "lab_xy_ate_max", lab_xy_stats.max, "m");
            WriteMetricCsv(out, "lab_xy_alignment_yaw", lab_xy_align.yaw_rad * kRadToDeg, "deg");
            WriteMetricCsv(out, "lab_xy_alignment_tx", lab_xy_align.t.x(), "m");
            WriteMetricCsv(out, "lab_xy_alignment_ty", lab_xy_align.t.y(), "m");

            WriteMetricCsv(out, "hortimulti_ate_rmse",
                           hortimulti_ate_rmse_m, "m");
            WriteMetricCsv(out, "hortimulti_rpe_percent",
                           hortimulti_rpe_percent, "%");
            WriteMetricCsv(out, "hortimulti_rpe_distance",
                           hortimulti_rpe.target_distance_m, "m");
            WriteMetricCsv(out, "hortimulti_rpe_count",
                           hortimulti_rpe.n, "count");

            for (const RpeStats& r : rpe_results) {
                const std::string prefix = "rpe_" + std::to_string(static_cast<int>(r.target_distance_m)) + "m_";
                WriteMetricCsv(out, prefix + "count", r.n, "count");
                WriteMetricCsv(out, prefix + "trans_rmse", r.translation.rmse, "m");
                WriteMetricCsv(out, prefix + "trans_p95", r.translation.p95, "m");
                WriteMetricCsv(out, prefix + "rot_rmse", r.rotation_deg.rmse, "deg");
                WriteMetricCsv(out, prefix + "rot_p95", r.rotation_deg.p95, "deg");
            }
        }

        bool plots_generated = false;
        if (options.make_plots) {
            const int gnuplot_available =
                std::system("command -v gnuplot >/dev/null 2>&1");
            if (gnuplot_available == 0) {
                const fs::path script_path = options.output_dir / "plot_eval.gp";
                WritePlotScript(script_path, trajectory_csv, rpe_csv, plots_dir);
                const std::string command = "gnuplot " + ShellQuote(script_path.string());
                const int rc = std::system(command.c_str());
                plots_generated = (rc == 0);
                if (!plots_generated) {
                    std::cerr << "Warning: gnuplot returned non-zero status: " << rc << "\n";
                }
            } else {
                std::cerr << "Warning: gnuplot not found; skipping PNG generation.\n";
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "========== FR-SLAM LIO GT EVALUATION ==========\n";
        std::cout << "Estimator poses           : " << est.size() << "\n";
        std::cout << "GT poses                  : " << gt_os.size() << "\n";
        std::cout << "Matched poses             : " << matched.size() << "\n";
        std::cout << "Rejected outside GT range : " << rejected_outside_range << "\n";
        std::cout << "Rejected large GT gap     : " << rejected_large_gap << "\n";
        std::cout << "Evaluation duration       : " << eval_duration_s << " s\n";
        std::cout << "Evaluation GT path length : " << eval_gt_path_m << " m\n\n";

        std::cout << "HORTIMULTI BENCHMARK\n";
        std::cout << "  ATE RMSE                : "
                  << hortimulti_ate_rmse_m << " m\n";
        std::cout << "  RPE translation         : "
                  << hortimulti_rpe_percent << " %\n";
        std::cout << "  RPE source distance     : "
                  << hortimulti_rpe.target_distance_m << " m\n";
        std::cout << "  RPE samples             : "
                  << hortimulti_rpe.n << "\n\n";

        std::cout << "FIRST-POSE ALIGNMENT\n";
        std::cout << "  Translation APE RMSE    : " << first_trans_stats.rmse << " m\n";
        std::cout << "  Translation APE P95     : " << first_trans_stats.p95 << " m\n";
        std::cout << "  Rotation APE RMSE       : " << first_rot_stats.rmse << " deg\n";
        std::cout << "  |Z error| RMSE          : " << first_z_stats.rmse << " m\n";
        std::cout << "  |Yaw error| RMSE        : " << first_yaw_stats.rmse << " deg\n";
        std::cout << "  Final translation error : " << final_first.trans_m << " m\n";
        std::cout << "  Final Z error           : " << final_first.z_m << " m\n";
        std::cout << "  Final yaw error         : " << final_first.yaw_deg << " deg\n";
        std::cout << "  Endpoint/path drift     : " << endpoint_drift_percent << " %\n\n";

        std::cout << "GLOBAL SE(3) ALIGNMENT (scale=1)\n";
        std::cout << "  Translation APE RMSE    : " << se3_trans_stats.rmse << " m\n";
        std::cout << "  Translation APE P95     : " << se3_trans_stats.p95 << " m\n";
        std::cout << "  Rotation APE RMSE       : " << se3_rot_stats.rmse << " deg\n\n";

        std::cout << "LAB-STYLE GLOBAL XY ALIGNMENT (SE(2), scale=1, same-run fit)\n";
        std::cout << "  XY ATE RMSE             : " << lab_xy_stats.rmse << " m"
                  << " (" << 100.0 * lab_xy_stats.rmse << " cm)\n";
        std::cout << "  XY ATE P95              : " << lab_xy_stats.p95 << " m"
                  << " (" << 100.0 * lab_xy_stats.p95 << " cm)\n";
        std::cout << "  Alignment yaw           : "
                  << lab_xy_align.yaw_rad * kRadToDeg << " deg\n\n";

        std::cout << "RPE\n";
        for (const RpeStats& r : rpe_results) {
            std::cout << "  @ " << r.target_distance_m << " m"
                      << " | N=" << r.n
                      << " | trans RMSE=" << r.translation.rmse << " m"
                      << " | rot RMSE=" << r.rotation_deg.rmse << " deg\n";
        }

        std::cout << "\nOutput directory          : " << options.output_dir << "\n";
        std::cout << "Summary                   : " << summary_txt << "\n";
        std::cout << "Plots generated           : " << (plots_generated ? "yes" : "no") << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "lio_gt_evaluator ERROR: " << e.what() << "\n";
        PrintUsage(argv[0]);
        return 1;
    }
}
