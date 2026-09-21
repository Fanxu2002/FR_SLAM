#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cmath>
#include <cstddef>


// ============================================================================
// LoopRetrievalSample
//
// IMPORTANT:
//
// This is NOT a new PoseGraph vertex.
//
// It is only a stable sampling layer for loop retrieval:
//
//     Mapping Keyframe
//           |
//           v
//     LoopRetrievalSampler
//           |
//           v
//     Loop Retrieval Sample
//           |
//           +---- Scan Context
//           |
//           +---- BTC
//
// Every Loop Retrieval Sample always remembers the original Mapping Keyframe
// ID. Therefore any accepted loop can still be converted back to the original
// Mapping Keyframe pair and inserted into the existing PoseGraph.
// ============================================================================
struct LoopRetrievalSample
{
    std::size_t sample_id = 0;

    std::size_t mapping_keyframe_id = 0;

    double timestamp = 0.0;

    Eigen::Isometry3d T_WL =
        Eigen::Isometry3d::Identity();
};


// ============================================================================
// LoopRetrievalSamplingDecision
//
// Diagnostic information for one Mapping Keyframe.
//
// accumulated_path_distance_m:
//     Sum of consecutive Mapping-Keyframe translation distances since the
//     latest accepted Loop Retrieval Sample.
//
// rotation_from_last_sample_deg:
//     Full SO(3) angle from the latest accepted Loop Retrieval Sample.
//
// First sample is always accepted.
// ============================================================================
struct LoopRetrievalSamplingDecision
{
    bool valid = false;

    bool accepted = false;

    bool first_sample = false;

    bool trigger_translation = false;

    bool trigger_rotation = false;

    double accumulated_path_distance_m = 0.0;

    double rotation_from_last_sample_deg = 0.0;
};


// ============================================================================
// LoopRetrievalSampler
//
// Outdoor V1 rule:
//
//     accumulated path distance >= 0.50 m
//
// OR
//
//     rotation from latest Loop Sample >= 10 deg
//
// IMPORTANT:
//
// Translation uses ACCUMULATED PATH LENGTH:
//
//     KF100 -> KF101 -> KF102 -> KF103
//
// rather than:
//
//     || p_KF103 - p_KF100 ||
//
// This prevents curved motion from being under-sampled.
//
// Mapping Keyframe generation remains completely unchanged.
// ============================================================================
class LoopRetrievalSampler
{
public:
    LoopRetrievalSampler(
        double translation_threshold_m,
        double rotation_threshold_deg)
        : translation_threshold_m_(
              std::max(
                  0.0,
                  translation_threshold_m)),
          rotation_threshold_rad_(
              std::max(
                  0.0,
                  rotation_threshold_deg) *
              kPi / 180.0)
    {
    }

    bool ProcessMappingKeyframe(
        std::size_t mapping_keyframe_id,
        double timestamp,
        const Eigen::Isometry3d &T_WL,
        LoopRetrievalSample &sample,
        LoopRetrievalSamplingDecision &decision)
    {
        decision =
            LoopRetrievalSamplingDecision();

        if (!std::isfinite(timestamp) ||
            !T_WL.matrix().allFinite())
        {
            return false;
        }

        decision.valid = true;

        // --------------------------------------------------------------------
        // First Mapping Keyframe always becomes the first Loop Sample.
        // --------------------------------------------------------------------
        if (!has_previous_mapping_keyframe_)
        {
            previous_mapping_pose_ =
                T_WL;

            has_previous_mapping_keyframe_ =
                true;

            last_sample_pose_ =
                T_WL;

            has_last_sample_ =
                true;

            accumulated_path_distance_m_ =
                0.0;

            decision.accepted = true;
            decision.first_sample = true;

            sample.sample_id =
                next_sample_id_++;

            sample.mapping_keyframe_id =
                mapping_keyframe_id;

            sample.timestamp =
                timestamp;

            sample.T_WL =
                T_WL;

            return true;
        }

        // --------------------------------------------------------------------
        // Accumulate traveled path distance between consecutive Mapping KFs.
        // --------------------------------------------------------------------
        const double segment_distance_m =
            (T_WL.translation() -
             previous_mapping_pose_.translation())
                .norm();

        if (std::isfinite(segment_distance_m))
        {
            accumulated_path_distance_m_ +=
                segment_distance_m;
        }

        previous_mapping_pose_ =
            T_WL;

        decision.accumulated_path_distance_m =
            accumulated_path_distance_m_;

        // --------------------------------------------------------------------
        // Rotation is measured relative to the latest accepted Loop Sample.
        // --------------------------------------------------------------------
        double rotation_rad = 0.0;

        if (has_last_sample_)
        {
            const Eigen::Matrix3d relative_rotation =
                last_sample_pose_.rotation().transpose() *
                T_WL.rotation();

            if (relative_rotation.allFinite())
            {
                const Eigen::AngleAxisd angle_axis(
                    relative_rotation);

                rotation_rad =
                    std::abs(
                        angle_axis.angle());
            }
        }

        decision.rotation_from_last_sample_deg =
            rotation_rad * 180.0 / kPi;

        decision.trigger_translation =
            accumulated_path_distance_m_ >=
            translation_threshold_m_;

        decision.trigger_rotation =
            rotation_rad >=
            rotation_threshold_rad_;

        decision.accepted =
            decision.trigger_translation ||
            decision.trigger_rotation;

        if (!decision.accepted)
        {
            return false;
        }

        // --------------------------------------------------------------------
        // Accepted Loop Retrieval Sample.
        //
        // Reset path accumulation from THIS accepted sample.
        // --------------------------------------------------------------------
        sample.sample_id =
            next_sample_id_++;

        sample.mapping_keyframe_id =
            mapping_keyframe_id;

        sample.timestamp =
            timestamp;

        sample.T_WL =
            T_WL;

        last_sample_pose_ =
            T_WL;

        has_last_sample_ =
            true;

        accumulated_path_distance_m_ =
            0.0;

        return true;
    }

    void Reset()
    {
        next_sample_id_ = 0;

        has_previous_mapping_keyframe_ = false;
        has_last_sample_ = false;

        accumulated_path_distance_m_ = 0.0;

        previous_mapping_pose_ =
            Eigen::Isometry3d::Identity();

        last_sample_pose_ =
            Eigen::Isometry3d::Identity();
    }

    std::size_t NextSampleId() const
    {
        return next_sample_id_;
    }

private:
    static constexpr double kPi =
        3.14159265358979323846;

    double translation_threshold_m_ =
        0.50;

    double rotation_threshold_rad_ =
        10.0 * kPi / 180.0;

    std::size_t next_sample_id_ = 0;

    bool has_previous_mapping_keyframe_ = false;

    bool has_last_sample_ = false;

    double accumulated_path_distance_m_ = 0.0;

    Eigen::Isometry3d previous_mapping_pose_ =
        Eigen::Isometry3d::Identity();

    Eigen::Isometry3d last_sample_pose_ =
        Eigen::Isometry3d::Identity();
};
