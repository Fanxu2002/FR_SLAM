#include "fr_slam/loop/btc_adapter.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <chrono>

#ifndef FR_SLAM_BTC_CONFIG_DIR
#error "FR_SLAM_BTC_CONFIG_DIR must be defined by CMake."
#endif

namespace fr_slam_btc
{

    namespace
    {
        // Diagnostic only: exact-cloud yaw-180 BTC rotation self-test.
        // Runs once on the BTC query whose endpoint KF is 180.
        constexpr std::size_t kBtcRotationSelfTestEndpointKf = 180;
        constexpr bool kEnableBtcSelfTests = false;

        constexpr double kRadToDeg =
            180.0 / 3.14159265358979323846;
    }

    OfficialBtcAdapter::OfficialBtcAdapter(
        const std::string &config_profile,
        int skip_near_num,
        int proj_plane_num,
        std::size_t window_size,
        std::size_t window_stride)
        : btc_window_size_(
              std::max<std::size_t>(
                  1,
                  window_size)),
          btc_window_stride_(
              std::min(
                  btc_window_size_,
                  std::max<std::size_t>(
                      1,
                      window_stride)))
    {
        std::string normalized_profile =
            config_profile;

        std::transform(
            normalized_profile.begin(),
            normalized_profile.end(),
            normalized_profile.begin(),
            [](unsigned char character)
            {
                return static_cast<char>(
                    std::tolower(character));
            });

        std::string config_filename;

        if (normalized_profile == "indoor")
        {
            config_filename =
                "config_indoor.yaml";
        }
        else if (normalized_profile == "outdoor")
        {
            config_filename =
                "config_outdoor.yaml";
        }
        else
        {
            throw std::runtime_error(
                "Unsupported btc_config_profile='" +
                config_profile +
                "'. Expected 'indoor' or 'outdoor'.");
        }

        std::string config_path =
            std::string(
                FR_SLAM_BTC_CONFIG_DIR) +
            "/" +
            config_filename;

        load_config_setting(
            config_path,
            config_);

        // Official BTC descriptor values come from the selected profile.
        // Only FR-SLAM integration policy is overridden here.
        config_.skip_near_num_ =
            std::max(
                0,
                skip_near_num);

        config_.proj_plane_num_ =
            std::max(
                1,
                proj_plane_num);

        manager_ =
            std::make_unique<BtcDescManager>(
                config_);

        manager_->print_debug_info_ =
            true;
    }

    bool OfficialBtcAdapter::IsProcessedSubmap(
        std::size_t submap_id) const
    {
        std::lock_guard<std::mutex>
            lock(mutex_);

        return std::find(
                   internal_to_submap_id_.begin(),
                   internal_to_submap_id_.end(),
                   submap_id) !=
               internal_to_submap_id_.end();
    }

    std::size_t
    OfficialBtcAdapter::ProcessedSubmapCount() const
    {
        std::lock_guard<std::mutex>
            lock(mutex_);

        return internal_to_submap_id_.size();
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr
    OfficialBtcAdapter::ConvertSubmapCloud(
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_S)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr output(
            new pcl::PointCloud<pcl::PointXYZI>());

        if (!cloud_S)
        {
            return output;
        }

        output->reserve(
            cloud_S->size());

        for (const LIDAR_POINT &point :
             cloud_S->points)
        {
            if (!std::isfinite(point.x) ||
                !std::isfinite(point.y) ||
                !std::isfinite(point.z))
            {
                continue;
            }

            pcl::PointXYZI btc_point;
            btc_point.x = point.x;
            btc_point.y = point.y;
            btc_point.z = point.z;

            if (std::isfinite(
                    static_cast<double>(
                        point.intensity)))
            {
                btc_point.intensity =
                    static_cast<float>(
                        point.intensity);
            }
            else
            {
                btc_point.intensity = 0.0F;
            }

            output->push_back(
                btc_point);
        }

        output->width =
            static_cast<std::uint32_t>(
                output->size());
        output->height = 1;
        output->is_dense = true;

        return output;
    }

    void OfficialBtcAdapter::RotationToRpyDeg(
        const Eigen::Matrix3d &R,
        double &roll_deg,
        double &pitch_deg,
        double &yaw_deg)
    {
        if (!R.allFinite())
        {
            roll_deg =
                std::numeric_limits<double>::quiet_NaN();
            pitch_deg =
                std::numeric_limits<double>::quiet_NaN();
            yaw_deg =
                std::numeric_limits<double>::quiet_NaN();
            return;
        }

        const double sin_pitch =
            std::clamp(
                -R(2, 0),
                -1.0,
                1.0);

        const double pitch =
            std::asin(
                sin_pitch);

        const double cos_pitch =
            std::cos(
                pitch);

        double roll = 0.0;
        double yaw = 0.0;

        if (std::abs(cos_pitch) > 1e-8)
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
            yaw =
                std::atan2(
                    -R(0, 1),
                    R(1, 1));
        }

        roll_deg = roll * kRadToDeg;
        pitch_deg = pitch * kRadToDeg;
        yaw_deg = yaw * kRadToDeg;
    }

    OfficialBtcSubmapResult
    OfficialBtcAdapter::ProcessSubmap(
        std::size_t submap_id,
        const pcl::PointCloud<LIDAR_POINT>::ConstPtr &cloud_S)
    {
        std::lock_guard<std::mutex>
            lock(mutex_);

        OfficialBtcSubmapResult result;
        result.current_submap_id =
            submap_id;

        // V31.1: submap_id is the external BTC query endpoint KF id.
        // Every accepted query endpoint is processed at most once.  The official
        // BTC internal frame id remains a contiguous generation index so all
        // upstream history vectors (plane clouds, binary lists, etc.) keep their
        // original indexing semantics.
        if (std::find(
                internal_to_submap_id_.begin(),
                internal_to_submap_id_.end(),
                submap_id) !=
            internal_to_submap_id_.end())
        {
            return result;
        }

        const pcl::PointCloud<pcl::PointXYZI>::Ptr btc_cloud =
            ConvertSubmapCloud(
                cloud_S);

        result.input_points =
            btc_cloud->size();

        if (btc_cloud->empty())
        {
            return result;
        }

        // ====================================================================
        // V31.3 DIAGNOSTIC ONLY
        //
        // Exact same BTC Submap point set:
        //
        //   frame 0 = original cloud
        //   frame 1 = same cloud rotated +180 deg around local Z
        //
        // Independent BtcDescManager:
        //   does NOT modify manager_
        //   does NOT modify internal_to_submap_id_
        //   does NOT modify the real BTC database.
        //
        // This isolates pure yaw-rotation behaviour from FOV, occlusion,
        // sampling, frontend odometry, ICP and PoseGraph effects.
        // ====================================================================

        if (kEnableBtcSelfTests && submap_id == kBtcRotationSelfTestEndpointKf)
        {

            ConfigSetting test_config =
                config_;

            // Only two diagnostic frames exist (0 and 1), therefore the normal
            // FR-SLAM 30-frame temporal exclusion must be disabled locally.
            test_config.skip_near_num_ = 0;

            BtcDescManager test_manager(
                test_config);

            test_manager.print_debug_info_ =
                true;

            // --------------------------------------------------------------
            // Diagnostic frame 0: original cloud.
            // --------------------------------------------------------------
            std::vector<BTC> original_btcs;

            test_manager.GenerateBtcDescs(
                btc_cloud,
                0,
                original_btcs);

            const std::size_t original_binary_count =
                !test_manager.history_binary_list_.empty()
                    ? test_manager.history_binary_list_[0].size()
                    : 0;

            if (!original_btcs.empty())
            {
                test_manager.AddBtcDescs(
                    original_btcs);
            }

            // --------------------------------------------------------------
            // Diagnostic frame 1: same points, pure Rz(pi).
            //
            // x' = -x
            // y' = -y
            // z' =  z
            //
            // No translation and no point removal/addition.
            // --------------------------------------------------------------
            pcl::PointCloud<pcl::PointXYZI>::Ptr rotated_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());

            rotated_cloud->reserve(
                btc_cloud->size());

            for (const pcl::PointXYZI &point :
                 btc_cloud->points)
            {
                pcl::PointXYZI rotated_point =
                    point;

                rotated_point.x =
                    -point.x;

                rotated_point.y =
                    -point.y;

                rotated_point.z =
                    point.z;

                rotated_cloud->push_back(
                    rotated_point);
            }

            rotated_cloud->width =
                static_cast<std::uint32_t>(
                    rotated_cloud->size());

            rotated_cloud->height = 1;
            rotated_cloud->is_dense = true;

            std::vector<BTC> rotated_btcs;

            test_manager.GenerateBtcDescs(
                rotated_cloud,
                1,
                rotated_btcs);

            const std::size_t rotated_binary_count =
                test_manager.history_binary_list_.size() > 1
                    ? test_manager.history_binary_list_[1].size()
                    : 0;

            std::pair<int, double>
                test_search_result(-1, 0.0);

            std::pair<
                Eigen::Vector3d,
                Eigen::Matrix3d>
                test_loop_transform;

            test_loop_transform.first =
                Eigen::Vector3d::Zero();

            test_loop_transform.second =
                Eigen::Matrix3d::Identity();

            std::vector<
                std::pair<BTC, BTC>>
                test_matched_pairs;

            if (!rotated_btcs.empty() &&
                !original_btcs.empty())
            {
                test_manager.SearchLoop(
                    rotated_btcs,
                    test_search_result,
                    test_loop_transform,
                    test_matched_pairs);
            }

            double test_roll_deg = 0.0;
            double test_pitch_deg = 0.0;
            double test_yaw_deg = 0.0;

            RotationToRpyDeg(
                test_loop_transform.second,
                test_roll_deg,
                test_pitch_deg,
                test_yaw_deg);

            const double yaw_180_error_deg =
                std::isfinite(test_yaw_deg)
                    ? std::abs(
                          std::abs(test_yaw_deg) -
                          180.0)
                    : std::numeric_limits<double>::infinity();

            const double btc_count_ratio =
                original_btcs.empty()
                    ? 0.0
                    : static_cast<double>(
                          rotated_btcs.size()) /
                          static_cast<double>(
                              original_btcs.size());

            const double binary_count_ratio =
                original_binary_count == 0
                    ? 0.0
                    : static_cast<double>(
                          rotated_binary_count) /
                          static_cast<double>(
                              original_binary_count);


        }

        // ====================================================================
        // V31.5 DIAGNOSTIC ONLY
        //
        // Exact identity self-test:
        //
        //   frame 0 = original BTC cloud
        //   frame 1 = EXACT SAME original BTC cloud
        //
        // No rotation.
        // No translation.
        // No point removal.
        // No FOV change.
        //
        // Uses a completely independent BtcDescManager.
        // ====================================================================

        if (kEnableBtcSelfTests && submap_id == kBtcRotationSelfTestEndpointKf)
        {

            ConfigSetting identity_config =
                config_;

            identity_config.skip_near_num_ = 0;

            BtcDescManager identity_manager(
                identity_config);

            identity_manager.print_debug_info_ =
                true;

            // --------------------------------------------------------------
            // frame 0: reference
            // --------------------------------------------------------------

            std::vector<BTC> identity_reference_btcs;

            identity_manager.GenerateBtcDescs(
                btc_cloud,
                0,
                identity_reference_btcs);

            const std::size_t identity_reference_binary =
                !identity_manager.history_binary_list_.empty()
                    ? identity_manager.history_binary_list_[0].size()
                    : 0;

            if (!identity_reference_btcs.empty())
            {
                identity_manager.AddBtcDescs(
                    identity_reference_btcs);
            }

            // --------------------------------------------------------------
            // frame 1: exact same cloud
            // --------------------------------------------------------------

            std::vector<BTC> identity_query_btcs;

            identity_manager.GenerateBtcDescs(
                btc_cloud,
                1,
                identity_query_btcs);

            const std::size_t identity_query_binary =
                identity_manager.history_binary_list_.size() > 1
                    ? identity_manager.history_binary_list_[1].size()
                    : 0;

            std::pair<int, double>
                identity_search_result(-1, 0.0);

            std::pair<
                Eigen::Vector3d,
                Eigen::Matrix3d>
                identity_transform;

            identity_transform.first =
                Eigen::Vector3d::Zero();

            identity_transform.second =
                Eigen::Matrix3d::Identity();

            std::vector<
                std::pair<BTC, BTC>>
                identity_matched_pairs;

            if (!identity_reference_btcs.empty() &&
                !identity_query_btcs.empty())
            {
                identity_manager.SearchLoop(
                    identity_query_btcs,
                    identity_search_result,
                    identity_transform,
                    identity_matched_pairs);
            }

            double identity_roll_deg = 0.0;
            double identity_pitch_deg = 0.0;
            double identity_yaw_deg = 0.0;

            RotationToRpyDeg(
                identity_transform.second,
                identity_roll_deg,
                identity_pitch_deg,
                identity_yaw_deg);


        }

        // ====================================================================
        // V31.4 DIAGNOSTIC ONLY
        //
        // Synthetic 180-degree visibility self-test.
        //
        // Case A: SAME_HALF_ROT180
        //
        //   front half of the cloud
        //       vs
        //   exactly the same front-half points rotated by Rz(pi)
        //
        // This checks whether partial visibility alone breaks yaw invariance.
        //
        // Case B: OPPOSITE_HALF_ROT180
        //
        //   front half
        //       vs
        //   opposite/back half, expressed in a yaw-180 sensor frame
        //
        // This changes visible structure while keeping the same underlying
        // original Submap and therefore isolates visibility/FOV sensitivity.
        //
        // IMPORTANT:
        // Every test uses its own BtcDescManager. The real BTC database is not
        // touched.
        // ====================================================================

        if (kEnableBtcSelfTests && submap_id == kBtcRotationSelfTestEndpointKf)
        {

            pcl::PointCloud<pcl::PointXYZI>::Ptr front_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());

            pcl::PointCloud<pcl::PointXYZI>::Ptr front_rot180_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());

            pcl::PointCloud<pcl::PointXYZI>::Ptr back_rot180_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());

            front_cloud->reserve(
                btc_cloud->size());

            front_rot180_cloud->reserve(
                btc_cloud->size());

            back_rot180_cloud->reserve(
                btc_cloud->size());

            for (const pcl::PointXYZI &point :
                 btc_cloud->points)
            {
                if (point.x >= 0.0F)
                {
                    // Original forward-facing 180-degree half.
                    front_cloud->push_back(
                        point);

                    // Same exact visible points, only yaw-rotated.
                    pcl::PointXYZI rotated_point =
                        point;

                    rotated_point.x =
                        -point.x;

                    rotated_point.y =
                        -point.y;

                    rotated_point.z =
                        point.z;

                    front_rot180_cloud->push_back(
                        rotated_point);
                }
                else
                {
                    // Opposite physical half of the environment.
                    //
                    // Rotate it into the coordinate system of a sensor whose
                    // heading has changed by 180 degrees.
                    pcl::PointXYZI rotated_point =
                        point;

                    rotated_point.x =
                        -point.x;

                    rotated_point.y =
                        -point.y;

                    rotated_point.z =
                        point.z;

                    back_rot180_cloud->push_back(
                        rotated_point);
                }
            }

            const auto finalize_cloud =
                [](pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud)
            {
                cloud->width =
                    static_cast<std::uint32_t>(
                        cloud->size());

                cloud->height = 1;
                cloud->is_dense = true;
            };

            finalize_cloud(
                front_cloud);

            finalize_cloud(
                front_rot180_cloud);

            finalize_cloud(
                back_rot180_cloud);


            const auto run_fov_test =
                [&](const char *test_name,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &reference_cloud,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr &query_cloud)
            {
                ConfigSetting test_config =
                    config_;

                test_config.skip_near_num_ = 0;

                BtcDescManager test_manager(
                    test_config);

                test_manager.print_debug_info_ =
                    true;

                std::vector<BTC> reference_btcs;
                std::vector<BTC> query_btcs;

                test_manager.GenerateBtcDescs(
                    reference_cloud,
                    0,
                    reference_btcs);

                const std::size_t reference_binary_count =
                    !test_manager.history_binary_list_.empty()
                        ? test_manager.history_binary_list_[0].size()
                        : 0;

                if (!reference_btcs.empty())
                {
                    test_manager.AddBtcDescs(
                        reference_btcs);
                }

                test_manager.GenerateBtcDescs(
                    query_cloud,
                    1,
                    query_btcs);

                const std::size_t query_binary_count =
                    test_manager.history_binary_list_.size() > 1
                        ? test_manager.history_binary_list_[1].size()
                        : 0;

                std::pair<int, double>
                    test_search_result(-1, 0.0);

                std::pair<
                    Eigen::Vector3d,
                    Eigen::Matrix3d>
                    test_loop_transform;

                test_loop_transform.first =
                    Eigen::Vector3d::Zero();

                test_loop_transform.second =
                    Eigen::Matrix3d::Identity();

                std::vector<
                    std::pair<BTC, BTC>>
                    test_matched_pairs;

                if (!reference_btcs.empty() &&
                    !query_btcs.empty())
                {
                    test_manager.SearchLoop(
                        query_btcs,
                        test_search_result,
                        test_loop_transform,
                        test_matched_pairs);
                }

                double test_roll_deg = 0.0;
                double test_pitch_deg = 0.0;
                double test_yaw_deg = 0.0;

                RotationToRpyDeg(
                    test_loop_transform.second,
                    test_roll_deg,
                    test_pitch_deg,
                    test_yaw_deg);

                const double yaw_180_error_deg =
                    std::isfinite(test_yaw_deg)
                        ? std::abs(
                              std::abs(test_yaw_deg) -
                              180.0)
                        : std::numeric_limits<double>::infinity();

            };

            // --------------------------------------------------------------
            // Control:
            // same exact visible 180-degree subset, only yaw rotated.
            // --------------------------------------------------------------
            run_fov_test(
                "SAME_HALF_ROT180",
                front_cloud,
                front_rot180_cloud);

            // --------------------------------------------------------------
            // Actual visibility test:
            // opposite spatial half becomes visible after heading reversal.
            // --------------------------------------------------------------
            run_fov_test(
                "OPPOSITE_HALF_ROT180",
                front_cloud,
                back_rot180_cloud);

        }

        if (internal_to_submap_id_.size() >=
            static_cast<std::size_t>(
                std::numeric_limits<unsigned short>::max()))
        {
            throw std::runtime_error(
                "Official BTC frame_number_ limit exceeded.");
        }

        result.input_valid = true;

        const int internal_frame_id =
            static_cast<int>(
                internal_to_submap_id_.size());

        result.internal_frame_id =
            internal_frame_id;

        std::vector<BTC> btc_descriptors;

        const auto timing_adapter_begin =
            std::chrono::steady_clock::now();

        const auto timing_generate_begin =
            std::chrono::steady_clock::now();

        manager_->GenerateBtcDescs(
            btc_cloud,
            internal_frame_id,
            btc_descriptors);

        result.btc_descriptor_count =
            btc_descriptors.size();

        std::pair<int, double>
            search_result(-1, 0.0);

        std::pair<
            Eigen::Vector3d,
            Eigen::Matrix3d>
            loop_transform;

        loop_transform.first =
            Eigen::Vector3d::Zero();
        loop_transform.second =
            Eigen::Matrix3d::Identity();

        std::vector<
            std::pair<BTC, BTC>>
            matched_pairs;

        if (internal_frame_id >
                config_.skip_near_num_ &&
            !btc_descriptors.empty())
        {
            result.queried = true;

            manager_->SearchLoop(
                btc_descriptors,
                search_result,
                loop_transform,
                matched_pairs);
        }

        const bool full_database_window =
            submap_id >= btc_window_size_ - 1 &&
            ((submap_id - (btc_window_size_ - 1)) %
                 btc_window_stride_ ==
             0);

        // Preserve official ordering for database windows:
        // Generate -> Search -> Add.
        // Query-only windows intentionally stop after Search, but their generated
        // plane/binary history stays in the manager so official frame indexing
        // remains contiguous and candidate_verify() can safely index history.
        if (full_database_window)
        {
            manager_->AddBtcDescs(
                btc_descriptors);
        }

        internal_to_submap_id_.push_back(
            submap_id);

        result.newly_processed = true;
        result.score =
            search_result.second;
        result.matched_triangle_pairs =
            matched_pairs.size();

        if (search_result.first >= 0 &&
            static_cast<std::size_t>(search_result.first) <
                internal_to_submap_id_.size() - 1)
        {
            result.has_candidate = true;

            // Official BTC returns its internal frame id.  Map that generation
            // index back to the external database window endpoint KF id.
            result.historical_submap_id =
                internal_to_submap_id_[static_cast<std::size_t>(
                    search_result.first)];

            result.t_H_C =
                loop_transform.first;
            result.R_H_C =
                loop_transform.second;

            RotationToRpyDeg(
                result.R_H_C,
                result.roll_deg,
                result.pitch_deg,
                result.yaw_deg);
        }


        return result;
    }

} // namespace fr_slam_btc
