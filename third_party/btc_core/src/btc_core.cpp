#include "btc_core.hpp"

void load_config_setting(std::string &config_file,
                         ConfigSetting &config_setting)
{
  cv::FileStorage fSettings(config_file, cv::FileStorage::READ);
  if (!fSettings.isOpened())
  {
    std::cerr << "Failed to open settings file at: " << config_file
              << std::endl;
    exit(-1);
  }

  // for binary descriptor
  config_setting.useful_corner_num_ = fSettings["useful_corner_num"];
  config_setting.plane_merge_normal_thre_ =
      fSettings["plane_merge_normal_thre"];
  config_setting.plane_merge_dis_thre_ = fSettings["plane_merge_dis_thre"];
  config_setting.plane_detection_thre_ = fSettings["plane_detection_thre"];
  config_setting.voxel_size_ = fSettings["voxel_size"];
  config_setting.voxel_init_num_ = fSettings["voxel_init_num"];
  config_setting.proj_plane_num_ = fSettings["proj_plane_num"];
  config_setting.proj_image_resolution_ = fSettings["proj_image_resolution"];
  config_setting.proj_image_high_inc_ = fSettings["proj_image_high_inc"];
  config_setting.proj_dis_min_ = fSettings["proj_dis_min"];
  config_setting.proj_dis_max_ = fSettings["proj_dis_max"];
  config_setting.summary_min_thre_ = fSettings["summary_min_thre"];
  config_setting.line_filter_enable_ = fSettings["line_filter_enable"];

  // std descriptor
  config_setting.descriptor_near_num_ = fSettings["descriptor_near_num"];
  config_setting.descriptor_min_len_ = fSettings["descriptor_min_len"];
  config_setting.descriptor_max_len_ = fSettings["descriptor_max_len"];
  config_setting.non_max_suppression_radius_ = fSettings["max_constrait_dis"];
  config_setting.std_side_resolution_ = fSettings["triangle_resolution"];

  // candidate search
  config_setting.skip_near_num_ = fSettings["skip_near_num"];
  config_setting.candidate_num_ = fSettings["candidate_num"];
  config_setting.rough_dis_threshold_ = fSettings["rough_dis_threshold"];
  config_setting.similarity_threshold_ = fSettings["similarity_threshold"];
  config_setting.icp_threshold_ = fSettings["icp_threshold"];
  config_setting.normal_threshold_ = fSettings["normal_threshold"];
  config_setting.dis_threshold_ = fSettings["dis_threshold"];

}

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat,
                         double voxel_size)
{
  int intensity = rand() % 255;
  if (voxel_size < 0.01)
  {
    return;
  }
  std::unordered_map<VOXEL_LOC, M_POINT> voxel_map;
  uint plsize = pl_feat.size();

  for (uint i = 0; i < plsize; i++)
  {
    pcl::PointXYZI &p_c = pl_feat[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_c.data[j] / voxel_size;
      if (loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end())
    {
      iter->second.xyz[0] += p_c.x;
      iter->second.xyz[1] += p_c.y;
      iter->second.xyz[2] += p_c.z;
      iter->second.intensity += p_c.intensity;
      iter->second.count++;
    }
    else
    {
      M_POINT anp;
      anp.xyz[0] = p_c.x;
      anp.xyz[1] = p_c.y;
      anp.xyz[2] = p_c.z;
      anp.intensity = p_c.intensity;
      anp.count = 1;
      voxel_map[position] = anp;
    }
  }
  plsize = voxel_map.size();
  pl_feat.clear();
  pl_feat.resize(plsize);

  uint i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
  {
    pl_feat[i].x = iter->second.xyz[0] / iter->second.count;
    pl_feat[i].y = iter->second.xyz[1] / iter->second.count;
    pl_feat[i].z = iter->second.xyz[2] / iter->second.count;
    pl_feat[i].intensity = iter->second.intensity / iter->second.count;
    i++;
  }
}
double binary_similarity(const BinaryDescriptor &b1,
                         const BinaryDescriptor &b2)
{
  double dis = 0;
  for (size_t i = 0; i < b1.occupy_array_.size(); i++)
  {
    // to be debug hanming distance
    if (b1.occupy_array_[i] == true && b2.occupy_array_[i] == true)
    {
      dis += 1;
    }
  }
  return 2 * dis / (b1.summary_ + b2.summary_);
}

bool binary_greater_sort(BinaryDescriptor a, BinaryDescriptor b)
{
  return (a.summary_ > b.summary_);
}

bool plane_greater_sort(std::shared_ptr<Plane> plane1,
                        std::shared_ptr<Plane> plane2)
{
  return plane1->points_size_ > plane2->points_size_;
}

void OctoTree::init_octo_tree()
{
  if (voxel_points_.size() > config_setting_.voxel_init_num_)
  {
    init_plane();
  }
}

void OctoTree::init_plane()
{
  plane_ptr_->covariance_ = Eigen::Matrix3d::Zero();
  plane_ptr_->center_ = Eigen::Vector3d::Zero();
  plane_ptr_->normal_ = Eigen::Vector3d::Zero();
  plane_ptr_->points_size_ = voxel_points_.size();
  plane_ptr_->radius_ = 0;
  for (auto pi : voxel_points_)
  {
    plane_ptr_->covariance_ += pi * pi.transpose();
    plane_ptr_->center_ += pi;
  }
  plane_ptr_->center_ = plane_ptr_->center_ / plane_ptr_->points_size_;
  plane_ptr_->covariance_ =
      plane_ptr_->covariance_ / plane_ptr_->points_size_ -
      plane_ptr_->center_ * plane_ptr_->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane_ptr_->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3d::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMin) < config_setting_.plane_detection_thre_)
  {
    plane_ptr_->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
        evecs.real()(2, evalsMin);
    plane_ptr_->min_eigen_value_ = evalsReal(evalsMin);
    plane_ptr_->radius_ = sqrt(evalsReal(evalsMax));
    plane_ptr_->is_plane_ = true;

    plane_ptr_->d_ = -(plane_ptr_->normal_(0) * plane_ptr_->center_(0) +
                       plane_ptr_->normal_(1) * plane_ptr_->center_(1) +
                       plane_ptr_->normal_(2) * plane_ptr_->center_(2));
    plane_ptr_->p_center_.x = plane_ptr_->center_(0);
    plane_ptr_->p_center_.y = plane_ptr_->center_(1);
    plane_ptr_->p_center_.z = plane_ptr_->center_(2);
    plane_ptr_->p_center_.normal_x = plane_ptr_->normal_(0);
    plane_ptr_->p_center_.normal_y = plane_ptr_->normal_(1);
    plane_ptr_->p_center_.normal_z = plane_ptr_->normal_(2);
  }
  else
  {
    plane_ptr_->is_plane_ = false;
  }
}

double calc_triangle_dis(
    const std::vector<std::pair<BTC, BTC>> &match_std_list)
{
  double mean_triangle_dis = 0;
  for (auto var : match_std_list)
  {
    mean_triangle_dis += (var.first.triangle_ - var.second.triangle_).norm() /
                         var.first.triangle_.norm();
  }
  if (match_std_list.size() > 0)
  {
    mean_triangle_dis = mean_triangle_dis / match_std_list.size();
  }
  else
  {
    mean_triangle_dis = -1;
  }
  return mean_triangle_dis;
}

double calc_binary_similaity(
    const std::vector<std::pair<BTC, BTC>> &match_std_list)
{
  double mean_binary_similarity = 0;
  for (auto var : match_std_list)
  {
    mean_binary_similarity +=
        (binary_similarity(var.first.binary_A_, var.second.binary_A_) +
         binary_similarity(var.first.binary_B_, var.second.binary_B_) +
         binary_similarity(var.first.binary_C_, var.second.binary_C_)) /
        3;
  }
  if (match_std_list.size() > 0)
  {
    mean_binary_similarity = mean_binary_similarity / match_std_list.size();
  }
  else
  {
    mean_binary_similarity = -1;
  }
  return mean_binary_similarity;
}

void BtcDescManager::GenerateBtcDescs(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, const int frame_id,
    std::vector<BTC> &btcs_vec)
{ // step1, voxelization and plane dection

  // =======================================================================
  // FR-SLAM V28.5 FULL PIPELINE DIAGNOSTIC
  //
  // Diagnostic only. No algorithm state or threshold is changed.
  //
  // The checksum is intentionally order-independent:
  //   - quantize XYZ to 1 mm
  //   - per-point integer hash
  //   - aggregate with XOR and unsigned SUM
  //
  // If the same Submap gets a different hash on two runs, the BTC input
  // itself changed before descriptor extraction.
  // =======================================================================
  if (print_debug_info_)
  {
    double diag_min_x = 1e100;
    double diag_min_y = 1e100;
    double diag_min_z = 1e100;
    double diag_max_x = -1e100;
    double diag_max_y = -1e100;
    double diag_max_z = -1e100;

    double diag_sum_x = 0.0;
    double diag_sum_y = 0.0;
    double diag_sum_z = 0.0;

    std::size_t diag_finite_points = 0;

    unsigned long long diag_hash_xor = 0ULL;
    unsigned long long diag_hash_sum = 0ULL;

    for (const auto &diag_point : input_cloud->points)
    {
      if (!std::isfinite(diag_point.x) ||
          !std::isfinite(diag_point.y) ||
          !std::isfinite(diag_point.z))
      {
        continue;
      }

      ++diag_finite_points;

      diag_min_x = std::min(diag_min_x, static_cast<double>(diag_point.x));
      diag_min_y = std::min(diag_min_y, static_cast<double>(diag_point.y));
      diag_min_z = std::min(diag_min_z, static_cast<double>(diag_point.z));
      diag_max_x = std::max(diag_max_x, static_cast<double>(diag_point.x));
      diag_max_y = std::max(diag_max_y, static_cast<double>(diag_point.y));
      diag_max_z = std::max(diag_max_z, static_cast<double>(diag_point.z));

      diag_sum_x += static_cast<double>(diag_point.x);
      diag_sum_y += static_cast<double>(diag_point.y);
      diag_sum_z += static_cast<double>(diag_point.z);

      const long long qx =
          static_cast<long long>(std::llround(
              static_cast<double>(diag_point.x) * 1000.0));
      const long long qy =
          static_cast<long long>(std::llround(
              static_cast<double>(diag_point.y) * 1000.0));
      const long long qz =
          static_cast<long long>(std::llround(
              static_cast<double>(diag_point.z) * 1000.0));

      unsigned long long point_hash =
          static_cast<unsigned long long>(qx) *
          11400714819323198485ULL;

      point_hash ^=
          static_cast<unsigned long long>(qy) *
          14029467366897019727ULL;

      point_hash ^=
          static_cast<unsigned long long>(qz) *
          1609587929392839161ULL;

      diag_hash_xor ^= point_hash;
      diag_hash_sum += point_hash;
    }

    if (diag_finite_points == 0)
    {
      diag_min_x = diag_min_y = diag_min_z = 0.0;
      diag_max_x = diag_max_y = diag_max_z = 0.0;
    }

  }

  std::unordered_map<VOXEL_LOC, OctoTree *> voxel_map;
  init_voxel_map(input_cloud, voxel_map);

  if (print_debug_info_)
  {
    std::size_t diag_root_init_ready = 0;
    std::size_t diag_plane_voxels = 0;
    std::size_t diag_nonplane_voxels = 0;

    std::size_t diag_min_points_per_voxel =
        voxel_map.empty() ? 0 : static_cast<std::size_t>(-1);

    std::size_t diag_max_points_per_voxel = 0;
    std::size_t diag_total_voxel_points = 0;

    for (const auto &diag_entry : voxel_map)
    {
      const OctoTree *diag_tree =
          diag_entry.second;

      if (diag_tree == nullptr)
      {
        continue;
      }

      const std::size_t diag_points =
          diag_tree->voxel_points_.size();

      diag_total_voxel_points += diag_points;
      diag_min_points_per_voxel =
          std::min(diag_min_points_per_voxel, diag_points);
      diag_max_points_per_voxel =
          std::max(diag_max_points_per_voxel, diag_points);

      if (diag_points >
          static_cast<std::size_t>(
              config_setting_.voxel_init_num_))
      {
        ++diag_root_init_ready;
      }

      if (diag_tree->plane_ptr_ &&
          diag_tree->plane_ptr_->is_plane_)
      {
        ++diag_plane_voxels;
      }
      else
      {
        ++diag_nonplane_voxels;
      }
    }

    const double diag_mean_points_per_voxel =
        voxel_map.empty()
            ? 0.0
            : static_cast<double>(diag_total_voxel_points) /
                  static_cast<double>(voxel_map.size());

  }

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr plane_cloud(
      new pcl::PointCloud<pcl::PointXYZINormal>);
  get_plane(voxel_map, plane_cloud);
  if (print_debug_info_)
  {
  }

  plane_cloud_vec_.push_back(plane_cloud);

  // step3, extraction binary descriptors
  std::vector<std::shared_ptr<Plane>> proj_plane_list;
  std::vector<std::shared_ptr<Plane>> merge_plane_list;
  get_project_plane(voxel_map, proj_plane_list);
  if (proj_plane_list.size() == 0)
  {
    std::shared_ptr<Plane> single_plane(new Plane);
    single_plane->normal_ << 0, 0, 1;
    single_plane->center_ << input_cloud->points[0].x, input_cloud->points[0].y,
        input_cloud->points[0].z;
    merge_plane_list.push_back(single_plane);
  }
  else
  {
    sort(proj_plane_list.begin(), proj_plane_list.end(), plane_greater_sort);
    merge_plane(proj_plane_list, merge_plane_list);
    sort(merge_plane_list.begin(), merge_plane_list.end(), plane_greater_sort);
  }
  if (print_debug_info_)
  {
  }

  std::vector<BinaryDescriptor> binary_list;

  // =======================================================================
  constexpr bool kEnableBtcV286ProjectionDiagnostics = false;
  if (kEnableBtcV286ProjectionDiagnostics && print_debug_info_)
  {
  // FR-SLAM V28.6 PROJECTION CANDIDATE QUALITY DIAGNOSTIC
  //
  // Diagnostic only:
  //   * Every merged projection-plane candidate is evaluated independently.
  //   * The real binary_extractor() call below is NOT changed.
  //   * No thresholds, descriptor lists, database state, or retrieval state
  //     are modified.
  //
  // The purpose is to answer:
  //   "Does this Submap lack usable projected structure, or does the official
  //    first-N projection-plane selector simply choose weak planes?"
  // =======================================================================
  if (print_debug_info_)
  {
    struct DiagProjectionCandidate
    {
      std::size_t index = 0;
      std::size_t projected_points = 0;
      std::size_t raw_binary = 0;
      std::size_t per_plane_after_nms = 0;
      bool official_examined = false;
      bool selector_pass = false;
      bool official_selected = false;
      double angle_to_previous_selected_deg = -1.0;
      std::size_t support_points = 0;
      double radius = 0.0;
      Eigen::Vector3d normal = Eigen::Vector3d::Zero();
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      std::vector<BinaryDescriptor> raw_binary_list;
    };

    std::vector<DiagProjectionCandidate> diag_candidates;
    diag_candidates.reserve(merge_plane_list.size());

    Eigen::Vector3d diag_last_selected_normal(0.0, 0.0, 0.0);
    int diag_selected_count = 0;

    for (std::size_t diag_i = 0;
         diag_i < merge_plane_list.size();
         ++diag_i)
    {
      DiagProjectionCandidate diag_candidate;
      diag_candidate.index = diag_i;
      diag_candidate.center = merge_plane_list[diag_i]->center_;
      diag_candidate.normal = merge_plane_list[diag_i]->normal_;
      diag_candidate.support_points =
          static_cast<std::size_t>(
              merge_plane_list[diag_i]->points_size_);
      diag_candidate.radius =
          merge_plane_list[diag_i]->radius_;

      const double diag_normal_norm =
          diag_candidate.normal.norm();

      if (diag_normal_norm <= 1e-12)
      {
        continue;
      }

      diag_candidate.normal /= diag_normal_norm;

      // FR-SLAM V31.6 — rotation-equivariant normal orientation.

        // The BTC submap is expressed in the anchor-local frame.

        // Orient the plane normal toward the anchor origin (0,0,0).

        // This preserves the sign convention under rigid rotation.

        if (diag_candidate.normal.dot(diag_candidate.center) > 0.0)

        {

          diag_candidate.normal = -diag_candidate.normal;

        }



      // Reproduce the exact official selector decision, but without touching
      // the real selector state.
      if (diag_selected_count < config_setting_.proj_plane_num_)
      {
        diag_candidate.official_examined = true;

        if (diag_selected_count > 0)
        {
          double diag_dot =
              std::abs(
                  diag_candidate.normal.dot(
                      diag_last_selected_normal));

          if (diag_dot > 1.0)
          {
            diag_dot = 1.0;
          }

          diag_candidate.angle_to_previous_selected_deg =
              std::acos(diag_dot) *
              180.0 / 3.14159265358979323846;
        }

        diag_candidate.selector_pass =
            ((diag_candidate.normal -
              diag_last_selected_normal)
                     .norm() < 0.3 ||
             (diag_candidate.normal +
              diag_last_selected_normal)
                     .norm() > 0.3);

        if (diag_candidate.selector_pass)
        {
          diag_candidate.official_selected = true;
          diag_last_selected_normal =
              diag_candidate.normal;
          ++diag_selected_count;
        }
      }

      // Count exactly how many input points enter extract_binary()'s
      // signed-distance band for this projection plane.
      const double diag_A = diag_candidate.normal[0];
      const double diag_B = diag_candidate.normal[1];
      const double diag_C = diag_candidate.normal[2];
      const double diag_D =
          -(diag_A * diag_candidate.center[0] +
            diag_B * diag_candidate.center[1] +
            diag_C * diag_candidate.center[2]);

      for (const auto &diag_point : input_cloud->points)
      {
        const double diag_dis =
            static_cast<double>(diag_point.x) * diag_A +
            static_cast<double>(diag_point.y) * diag_B +
            static_cast<double>(diag_point.z) * diag_C +
            diag_D;

        if (diag_dis >= config_setting_.proj_dis_min_ &&
            diag_dis <= config_setting_.proj_dis_max_)
        {
          ++diag_candidate.projected_points;
        }
      }

      // Evaluate this candidate by itself. extract_binary() only writes to the
      // supplied local vector, so this does not alter the real BTC pipeline.
      extract_binary(
          diag_candidate.center,
          diag_candidate.normal,
          input_cloud,
          diag_candidate.raw_binary_list);

      diag_candidate.raw_binary =
          diag_candidate.raw_binary_list.size();

      std::vector<BinaryDescriptor> diag_single_nms =
          diag_candidate.raw_binary_list;

      if (!diag_single_nms.empty())
      {
        non_maxi_suppression(diag_single_nms);
      }

      diag_candidate.per_plane_after_nms =
          diag_single_nms.size();


      diag_candidates.push_back(
          std::move(diag_candidate));
    }

    // Reconstruct the descriptor pool produced by the candidates that the
    // official first-N selector would select.
    std::vector<BinaryDescriptor> diag_official_pool;

    for (const auto &diag_candidate : diag_candidates)
    {
      if (!diag_candidate.official_selected)
      {
        continue;
      }

      diag_official_pool.insert(
          diag_official_pool.end(),
          diag_candidate.raw_binary_list.begin(),
          diag_candidate.raw_binary_list.end());
    }

    const std::size_t diag_official_raw =
        diag_official_pool.size();

    if (!diag_official_pool.empty())
    {
      non_maxi_suppression(diag_official_pool);
    }

    const std::size_t diag_official_after_nms =
        diag_official_pool.size();

    const std::size_t diag_official_predicted_final =
        std::min(
            diag_official_after_nms,
            static_cast<std::size_t>(
                config_setting_.useful_corner_num_));

    // Rank all candidates by their independent Binary yield.
    std::vector<std::size_t> diag_rank_indices(
        diag_candidates.size());

    for (std::size_t diag_i = 0;
         diag_i < diag_rank_indices.size();
         ++diag_i)
    {
      diag_rank_indices[diag_i] = diag_i;
    }

    std::sort(
        diag_rank_indices.begin(),
        diag_rank_indices.end(),
        [&](const std::size_t diag_a,
            const std::size_t diag_b)
        {
          if (diag_candidates[diag_a].raw_binary !=
              diag_candidates[diag_b].raw_binary)
          {
            return diag_candidates[diag_a].raw_binary >
                   diag_candidates[diag_b].raw_binary;
          }

          if (diag_candidates[diag_a].per_plane_after_nms !=
              diag_candidates[diag_b].per_plane_after_nms)
          {
            return diag_candidates[diag_a].per_plane_after_nms >
                   diag_candidates[diag_b].per_plane_after_nms;
          }

          return diag_candidates[diag_a].support_points >
                 diag_candidates[diag_b].support_points;
        });

    const std::size_t diag_top_k =
        std::min(
            diag_rank_indices.size(),
            static_cast<std::size_t>(
                std::max(0, config_setting_.proj_plane_num_)));

    std::vector<BinaryDescriptor> diag_best_pool;

    for (std::size_t diag_rank = 0;
         diag_rank < diag_rank_indices.size();
         ++diag_rank)
    {
      const auto &diag_candidate =
          diag_candidates[diag_rank_indices[diag_rank]];

      if (diag_rank < 5)
      {
      }

      if (diag_rank < diag_top_k)
      {
        diag_best_pool.insert(
            diag_best_pool.end(),
            diag_candidate.raw_binary_list.begin(),
            diag_candidate.raw_binary_list.end());
      }
    }

    const std::size_t diag_best_raw =
        diag_best_pool.size();

    if (!diag_best_pool.empty())
    {
      non_maxi_suppression(diag_best_pool);
    }

    const std::size_t diag_best_after_nms =
        diag_best_pool.size();

    const std::size_t diag_best_predicted_final =
        std::min(
            diag_best_after_nms,
            static_cast<std::size_t>(
                config_setting_.useful_corner_num_));

    const long long diag_gain =
        static_cast<long long>(
            diag_best_predicted_final) -
        static_cast<long long>(
            diag_official_predicted_final);

  }

  }

  binary_extractor(merge_plane_list, input_cloud, binary_list);
  history_binary_list_.push_back(binary_list);
  // corner_cloud_vec_.push_back(corner_points);
  if (print_debug_info_)
  {
  }

  // step4, generate stable triangle descriptors
  btcs_vec.clear();
  generate_btc(binary_list, frame_id, btcs_vec);

  if (print_debug_info_)
  {
    std::unordered_map<BTC_LOC, bool> diag_btc_buckets;

    double diag_min_triangle_norm =
        btcs_vec.empty() ? 0.0 : 1e100;

    double diag_max_triangle_norm = 0.0;
    double diag_sum_triangle_norm = 0.0;

    for (const auto &diag_btc : btcs_vec)
    {
      BTC_LOC diag_position;

      diag_position.x =
          static_cast<int>(diag_btc.triangle_[0] + 0.5);
      diag_position.y =
          static_cast<int>(diag_btc.triangle_[1] + 0.5);
      diag_position.z =
          static_cast<int>(diag_btc.triangle_[2] + 0.5);

      diag_btc_buckets[diag_position] = true;

      const double diag_norm =
          diag_btc.triangle_.norm();

      diag_min_triangle_norm =
          std::min(diag_min_triangle_norm, diag_norm);
      diag_max_triangle_norm =
          std::max(diag_max_triangle_norm, diag_norm);
      diag_sum_triangle_norm += diag_norm;
    }

    const double diag_mean_triangle_norm =
        btcs_vec.empty()
            ? 0.0
            : diag_sum_triangle_norm /
                  static_cast<double>(btcs_vec.size());


  }
  // step5, clear memory
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    delete (iter->second);
  }
  return;
}

void BtcDescManager::SearchLoop(
    const std::vector<BTC> &btcs_vec, std::pair<int, double> &loop_result,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
    std::vector<std::pair<BTC, BTC>> &loop_std_pair)
{
  if (btcs_vec.size() == 0)
  {
    std::cerr << "[Official BTC] No BTC descriptors." << std::endl;

    if (print_debug_info_)
    {
    }

    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
  // step1, select candidates, default number 50
  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<BTCMatchList> candidate_matcher_vec;
  candidate_selector(btcs_vec, candidate_matcher_vec);

  if (print_debug_info_)
  {
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  // step2, select best candidates from rough candidates
  double best_score = 0;
  int best_candidate_id = -1;
  int triggle_candidate = -1;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> best_transform;
  std::vector<std::pair<BTC, BTC>> best_sucess_match_vec;
  for (size_t i = 0; i < candidate_matcher_vec.size(); i++)
  {
    double verify_score = -1;
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> relative_pose;
    std::vector<std::pair<BTC, BTC>> sucess_match_vec;
    candidate_verify(candidate_matcher_vec[i], verify_score, relative_pose,
                     sucess_match_vec);
    if (print_debug_info_)
    {
    }

    if (verify_score > best_score)
    {
      best_score = verify_score;
      best_candidate_id = candidate_matcher_vec[i].match_id_.second;
      best_transform = relative_pose;
      best_sucess_match_vec = sucess_match_vec;
      triggle_candidate = i;
    }
  }
  auto t3 = std::chrono::high_resolution_clock::now();

  //           << " ms, candidate verify: " << time_inc(t3, t2) << "ms"
  //           << std::endl;
  if (print_debug_info_)
  {
  }

  if (print_debug_info_)
  {
  }

  if (best_score > config_setting_.icp_threshold_)
  {
    loop_result = std::pair<int, double>(best_candidate_id, best_score);
    loop_transform = best_transform;
    loop_std_pair = best_sucess_match_vec;
    return;
  }
  else
  {
    loop_result = std::pair<int, double>(-1, 0);
    return;
  }
}

void BtcDescManager::AddBtcDescs(const std::vector<BTC> &btcs_vec)
{
  // update frame id
  for (auto single_std : btcs_vec)
  {
    // calculate the position of single std
    BTC_LOC position;
    position.x = (int)(single_std.triangle_[0] + 0.5);
    position.y = (int)(single_std.triangle_[1] + 0.5);
    position.z = (int)(single_std.triangle_[2] + 0.5);
    auto iter = data_base_.find(position);
    if (iter != data_base_.end())
    {
      data_base_[position].push_back(single_std);
    }
    else
    {
      std::vector<BTC> descriptor_vec;
      descriptor_vec.push_back(single_std);
      data_base_[position] = descriptor_vec;
    }
  }
  return;
}

void BtcDescManager::PlaneGeomrtricIcp(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform)
{
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++)
  {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }
  kd_tree->setInputCloud(input_cloud);
  ceres::LocalParameterization *quaternion_local_parameterization = new ceres::EigenQuaternionParameterization();
  ceres::Problem problem;
  ceres::LossFunction *loss_function = nullptr;
  Eigen::Matrix3d rot = transform.second;
  Eigen::Quaterniond q(rot);
  Eigen::Vector3d t = transform.first;
  double para_q[4] = {q.x(), q.y(), q.z(), q.w()};
  double para_t[3] = {t(0), t(1), t(2)};
  problem.AddParameterBlock(para_q, 4, quaternion_local_parameterization);
  problem.AddParameterBlock(para_t, 3);
  Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
  Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  int useful_match = 0;
  for (size_t i = 0; i < source_cloud->size(); i++)
  {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    pcl::PointXYZ use_search_point;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0)
    {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_point_dis = (pi - tpi).norm();
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < config_setting_.normal_threshold_ ||
           normal_add.norm() < config_setting_.normal_threshold_) &&
          point_to_plane < config_setting_.dis_threshold_ &&
          point_to_point_dis < 3)
      {
        useful_match++;
        ceres::CostFunction *cost_function;
        Eigen::Vector3d curr_point(source_cloud->points[i].x,
                                   source_cloud->points[i].y,
                                   source_cloud->points[i].z);
        Eigen::Vector3d curr_normal(source_cloud->points[i].normal_x,
                                    source_cloud->points[i].normal_y,
                                    source_cloud->points[i].normal_z);

        cost_function = PlaneSolver::Create(curr_point, curr_normal, tpi, tni);
        problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
      }
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  Eigen::Quaterniond q_opt(para_q[3], para_q[0], para_q[1], para_q[2]);
  rot = q_opt.toRotationMatrix();
  t << t_last_curr(0), t_last_curr(1), t_last_curr(2);
  transform.first = t;
  transform.second = rot;
}

void BtcDescManager::init_voxel_map(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map)
{
  uint plsize = input_cloud->size();
  for (uint i = 0; i < plsize; i++)
  {
    Eigen::Vector3d p_c(input_cloud->points[i].x, input_cloud->points[i].y,
                        input_cloud->points[i].z);
    double loc_xyz[3];
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = p_c[j] / config_setting_.voxel_size_;
      if (loc_xyz[j] < 0)
      {
        loc_xyz[j] -= 1.0;
      }
    }
    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                       (int64_t)loc_xyz[2]);
    auto iter = voxel_map.find(position);
    if (iter != voxel_map.end())
    {
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
    else
    {
      OctoTree *octo_tree = new OctoTree(config_setting_);
      voxel_map[position] = octo_tree;
      voxel_map[position]->voxel_points_.push_back(p_c);
    }
  }
  std::vector<std::unordered_map<VOXEL_LOC, OctoTree *>::iterator> iter_list;
  std::vector<size_t> index;
  size_t i = 0;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); ++iter)
  {
    index.push_back(i);
    i++;
    iter_list.push_back(iter);
    // iter->second->init_octo_tree();
  }
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i)
      { iter_list[i]->second->init_octo_tree(); });
}

void BtcDescManager::get_plane(
    const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud)
{
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    if (iter->second->plane_ptr_->is_plane_)
    {
      pcl::PointXYZINormal pi;
      pi.x = iter->second->plane_ptr_->center_[0];
      pi.y = iter->second->plane_ptr_->center_[1];
      pi.z = iter->second->plane_ptr_->center_[2];
      pi.normal_x = iter->second->plane_ptr_->normal_[0];
      pi.normal_y = iter->second->plane_ptr_->normal_[1];
      pi.normal_z = iter->second->plane_ptr_->normal_[2];
      plane_cloud->push_back(pi);
    }
  }
}

void BtcDescManager::get_project_plane(
    std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
    std::vector<std::shared_ptr<Plane>> &project_plane_list)
{
  std::vector<std::shared_ptr<Plane>> origin_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++)
  {
    if (iter->second->plane_ptr_->is_plane_)
    {
      origin_list.push_back(iter->second->plane_ptr_);
    }
  }
  // =====================================================================
  // FR-SLAM BTC_PROJECT_DSU_V319
  //
  // Build true connected components of mutually compatible voxel planes.
  //
  // The previous implementation assigned integer IDs while traversing an
  // unordered_map-derived vector. If two already-created groups were later
  // connected by another compatible pair, the two IDs were never unioned.
  // Therefore the result depended on traversal order.
  //
  // DSU / Union-Find removes that order-dependent grouping defect while
  // preserving the original pairwise compatibility criteria.
  // =====================================================================

  const std::size_t project_plane_count = origin_list.size();

  std::vector<int> project_parent(project_plane_count);
  std::vector<int> project_rank(project_plane_count, 0);

  for (std::size_t i = 0;
       i < project_plane_count;
       ++i)
  {
    project_parent[i] = static_cast<int>(i);
    origin_list[i]->id_ = 0;
  }

  auto project_find_root =
      [&project_parent](int node)
      {
        while (project_parent[node] != node)
        {
          project_parent[node] =
              project_parent[project_parent[node]];

          node = project_parent[node];
        }

        return node;
      };

  auto project_union =
      [&project_parent,
       &project_rank,
       &project_find_root](int lhs, int rhs)
      {
        int root_lhs = project_find_root(lhs);
        int root_rhs = project_find_root(rhs);

        if (root_lhs == root_rhs)
        {
          return;
        }

        if (project_rank[root_lhs] <
            project_rank[root_rhs])
        {
          project_parent[root_lhs] = root_rhs;
        }
        else if (project_rank[root_lhs] >
                 project_rank[root_rhs])
        {
          project_parent[root_rhs] = root_lhs;
        }
        else
        {
          project_parent[root_rhs] = root_lhs;
          ++project_rank[root_lhs];
        }
      };

  for (std::size_t i = 0;
       i < project_plane_count;
       ++i)
  {
    for (std::size_t j = i + 1;
         j < project_plane_count;
         ++j)
    {
      const Eigen::Vector3d normal_diff =
          origin_list[i]->normal_ -
          origin_list[j]->normal_;

      const Eigen::Vector3d normal_add =
          origin_list[i]->normal_ +
          origin_list[j]->normal_;

      const double dis1 =
          std::fabs(
              origin_list[i]->normal_(0) *
                  origin_list[j]->center_(0) +
              origin_list[i]->normal_(1) *
                  origin_list[j]->center_(1) +
              origin_list[i]->normal_(2) *
                  origin_list[j]->center_(2) +
              origin_list[i]->d_);

      const double dis2 =
          std::fabs(
              origin_list[j]->normal_(0) *
                  origin_list[i]->center_(0) +
              origin_list[j]->normal_(1) *
                  origin_list[i]->center_(1) +
              origin_list[j]->normal_(2) *
                  origin_list[i]->center_(2) +
              origin_list[j]->d_);

      const bool normal_compatible =
          normal_diff.norm() <
              config_setting_.plane_merge_normal_thre_ ||
          normal_add.norm() <
              config_setting_.plane_merge_normal_thre_;

      const bool distance_compatible =
          dis1 < config_setting_.plane_merge_dis_thre_ &&
          dis2 < config_setting_.plane_merge_dis_thre_;

      if (normal_compatible &&
          distance_compatible)
      {
        project_union(
            static_cast<int>(i),
            static_cast<int>(j));
      }
    }
  }

  std::vector<int> project_component_size(
      project_plane_count,
      0);

  for (std::size_t i = 0;
       i < project_plane_count;
       ++i)
  {
    const int root =
        project_find_root(static_cast<int>(i));

    ++project_component_size[root];
  }

  std::vector<int> project_root_to_id(
      project_plane_count,
      0);

  int current_id = 1;

  for (std::size_t i = 0;
       i < project_plane_count;
       ++i)
  {
    const int root =
        project_find_root(static_cast<int>(i));

    // Preserve the original get_project_plane() behavior:
    // singleton planes keep id=0 and are not emitted here.
    if (project_component_size[root] < 2)
    {
      origin_list[i]->id_ = 0;
      continue;
    }

    if (project_root_to_id[root] == 0)
    {
      project_root_to_id[root] = current_id;
      ++current_id;
    }

    origin_list[i]->id_ =
        project_root_to_id[root];
  }

  std::vector<std::shared_ptr<Plane>> merge_list;
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++)
  {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end())
      continue;
    if (origin_list[i]->id_ == 0)
    {
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++)
    {
      if (i == j)
        continue;
      if (origin_list[j]->id_ == origin_list[i]->id_)
      {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_++;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge)
    {
      merge_flag.push_back(merge_plane->id_);
      merge_list.push_back(merge_plane);
    }
  }
  project_plane_list = merge_list;
}

void BtcDescManager::merge_plane(
    std::vector<std::shared_ptr<Plane>> &origin_list,
    std::vector<std::shared_ptr<Plane>> &merge_plane_list)
{
  if (origin_list.size() == 1)
  {
    merge_plane_list = origin_list;
    return;
  }
  for (size_t i = 0; i < origin_list.size(); i++)
    origin_list[i]->id_ = 0;
  int current_id = 1;
  for (auto iter = origin_list.end() - 1; iter != origin_list.begin(); iter--)
  {
    for (auto iter2 = origin_list.begin(); iter2 != iter; iter2++)
    {
      Eigen::Vector3d normal_diff = (*iter)->normal_ - (*iter2)->normal_;
      Eigen::Vector3d normal_add = (*iter)->normal_ + (*iter2)->normal_;
      double dis1 =
          fabs((*iter)->normal_(0) * (*iter2)->center_(0) +
               (*iter)->normal_(1) * (*iter2)->center_(1) +
               (*iter)->normal_(2) * (*iter2)->center_(2) + (*iter)->d_);
      double dis2 =
          fabs((*iter2)->normal_(0) * (*iter)->center_(0) +
               (*iter2)->normal_(1) * (*iter)->center_(1) +
               (*iter2)->normal_(2) * (*iter)->center_(2) + (*iter2)->d_);
      if (normal_diff.norm() < config_setting_.plane_merge_normal_thre_ ||
          normal_add.norm() < config_setting_.plane_merge_normal_thre_)
        if (dis1 < config_setting_.plane_merge_dis_thre_ &&
            dis2 < config_setting_.plane_merge_dis_thre_)
        {
          if ((*iter)->id_ == 0 && (*iter2)->id_ == 0)
          {
            (*iter)->id_ = current_id;
            (*iter2)->id_ = current_id;
            current_id++;
          }
          else if ((*iter)->id_ == 0 && (*iter2)->id_ != 0)
            (*iter)->id_ = (*iter2)->id_;
          else if ((*iter)->id_ != 0 && (*iter2)->id_ == 0)
            (*iter2)->id_ = (*iter)->id_;
        }
    }
  }
  std::vector<int> merge_flag;

  for (size_t i = 0; i < origin_list.size(); i++)
  {
    auto it =
        std::find(merge_flag.begin(), merge_flag.end(), origin_list[i]->id_);
    if (it != merge_flag.end())
      continue;
    if (origin_list[i]->id_ == 0)
    {
      merge_plane_list.push_back(origin_list[i]);
      continue;
    }
    std::shared_ptr<Plane> merge_plane(new Plane);
    (*merge_plane) = (*origin_list[i]);
    bool is_merge = false;
    for (size_t j = 0; j < origin_list.size(); j++)
    {
      if (i == j)
        continue;
      if (origin_list[j]->id_ == origin_list[i]->id_)
      {
        is_merge = true;
        Eigen::Matrix3d P_PT1 =
            (merge_plane->covariance_ +
             merge_plane->center_ * merge_plane->center_.transpose()) *
            merge_plane->points_size_;
        Eigen::Matrix3d P_PT2 =
            (origin_list[j]->covariance_ +
             origin_list[j]->center_ * origin_list[j]->center_.transpose()) *
            origin_list[j]->points_size_;
        Eigen::Vector3d merge_center =
            (merge_plane->center_ * merge_plane->points_size_ +
             origin_list[j]->center_ * origin_list[j]->points_size_) /
            (merge_plane->points_size_ + origin_list[j]->points_size_);
        Eigen::Matrix3d merge_covariance =
            (P_PT1 + P_PT2) /
                (merge_plane->points_size_ + origin_list[j]->points_size_) -
            merge_center * merge_center.transpose();
        merge_plane->covariance_ = merge_covariance;
        merge_plane->center_ = merge_center;
        merge_plane->points_size_ =
            merge_plane->points_size_ + origin_list[j]->points_size_;
        merge_plane->sub_plane_num_ += origin_list[j]->sub_plane_num_;
        // for (size_t k = 0; k < origin_list[j]->cloud.size(); k++) {
        //   merge_plane->cloud.points.push_back(origin_list[j]->cloud.points[k]);
        // }
        Eigen::EigenSolver<Eigen::Matrix3d> es(merge_plane->covariance_);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        merge_plane->normal_ << evecs.real()(0, evalsMin),
            evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
        merge_plane->radius_ = sqrt(evalsReal(evalsMax));
        merge_plane->d_ = -(merge_plane->normal_(0) * merge_plane->center_(0) +
                            merge_plane->normal_(1) * merge_plane->center_(1) +
                            merge_plane->normal_(2) * merge_plane->center_(2));
        merge_plane->p_center_.x = merge_plane->center_(0);
        merge_plane->p_center_.y = merge_plane->center_(1);
        merge_plane->p_center_.z = merge_plane->center_(2);
        merge_plane->p_center_.normal_x = merge_plane->normal_(0);
        merge_plane->p_center_.normal_y = merge_plane->normal_(1);
        merge_plane->p_center_.normal_z = merge_plane->normal_(2);
      }
    }
    if (is_merge)
    {
      merge_flag.push_back(merge_plane->id_);
      merge_plane_list.push_back(merge_plane);
    }
  }
}

void BtcDescManager::binary_extractor(
    const std::vector<std::shared_ptr<Plane>> proj_plane_list,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_descriptor_list)
{
  // =======================================================================
  // FR-SLAM V28.7 — Projection Selector Fix
  //
  // Goal:
  //   1) Do not simply consume the first proj_plane_num_ planes.
  //   2) Prefer planes that actually produce stable Binary descriptors.
  //   3) Avoid spending multiple slots on nearly parallel projection planes.
  //   4) Select subsequent planes by the REAL incremental gain after global
  //      non-max suppression, not by raw Binary count alone.
  //
  // IMPORTANT:
  //   - No BTC retrieval threshold is changed here.
  //   - proj_plane_num_ is still respected.
  //   - useful_corner_num_ and the existing NMS remain unchanged.
  // =======================================================================

  binary_descriptor_list.clear();

  if (proj_plane_list.empty() ||
      config_setting_.proj_plane_num_ <= 0)
  {
    return;
  }

  struct SelectorCandidate
  {
    std::size_t index = 0;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    std::size_t support_points = 0;
    std::vector<BinaryDescriptor> raw_binary;
    std::size_t per_plane_after_nms = 0;
  };

  std::vector<SelectorCandidate> candidates;
  candidates.reserve(proj_plane_list.size());

  std::size_t max_support_points = 0;

  // Evaluate every projection candidate once for the ACTIVE selector.
  for (std::size_t i = 0; i < proj_plane_list.size(); ++i)
  {
    if (!proj_plane_list[i])
    {
      continue;
    }

    SelectorCandidate candidate;
    candidate.index = i;
    candidate.center = proj_plane_list[i]->center_;
    candidate.normal = proj_plane_list[i]->normal_;

    // ===================================================================
    //
    // FR-SLAM V31.6 — BTC rotation-equivariant plane-normal orientation

    //
    // A geometric plane has two equivalent normal representations:
    //     (n, d) == (-n, -d)
    //
    // Official BTC's extract_binary() is NOT sign invariant because:
    //   1) it uses a signed projection-distance gate [proj_dis_min, max];
    //   2) y_axis = normal.cross(x_axis), so n -> -n mirrors one image axis.
    //
    // The previous rule `normal.z() >= 0` is stable for horizontal planes,
    // but not for near-vertical walls where z is close to zero and can change
    // sign from tiny fitting noise. Canonicalize by the DOMINANT component
    // instead: whichever of |x|, |y|, |z| is largest is forced positive.
    // This maps n and -n to exactly the same representative while preserving
    // the physical plane center/position.
    // ===================================================================
    const double normal_norm = candidate.normal.norm();
    if (normal_norm <= 1e-12)
    {
      continue;
    }

    candidate.normal /= normal_norm;

    // FR-SLAM V31.6 — rotation-equivariant normal orientation.

        //

        // The BTC submap is expressed in the anchor-local frame.

        // Orient every projection-plane normal toward the anchor origin.

        //

        // For c' = R*c and n' = R*n:

        //     (R*n)^T (R*c) = n^T c

        // so the sign decision is invariant to rigid rotation.

        // This is important because BTC uses the asymmetric signed-distance

        // projection gate [proj_dis_min, proj_dis_max] = [-1, +4].

        if (candidate.normal.dot(candidate.center) > 0.0)

        {

          candidate.normal = -candidate.normal;

        }



    const double support_value =
        static_cast<double>(proj_plane_list[i]->points_size_);

    if (support_value > 0.0)
    {
      candidate.support_points =
          static_cast<std::size_t>(support_value);
    }

    max_support_points =
        std::max(max_support_points,
                 candidate.support_points);

    extract_binary(candidate.center,
                   candidate.normal,
                   input_cloud,
                   candidate.raw_binary);

    std::vector<BinaryDescriptor> per_plane_nms =
        candidate.raw_binary;

    if (!per_plane_nms.empty())
    {
      non_maxi_suppression(per_plane_nms);
    }

    candidate.per_plane_after_nms =
        per_plane_nms.size();

    // FR-SLAM diagnostic only.
    //
    // skip_near_num_ == 0 is used only by the isolated BTC self-test
    // managers. The real FR-SLAM BTC manager uses skip_near_num_=30,
    // therefore this does not flood the normal runtime log.
    if (config_setting_.skip_near_num_ == 0)
    {
    }

    candidates.push_back(std::move(candidate));
  }

  if (candidates.empty())
  {
    return;
  }

  // Reliability guard:
  // Very tiny planes can sometimes generate many Binary points, but those
  // descriptors are more likely to be unstable between revisits. Keep planes
  // with at least 10% of the strongest plane support, and at least 30 points.
  //
  // If this leaves no usable candidate, the selector automatically falls back
  // to all non-empty candidates instead of failing.
  constexpr double kSupportRatio = 0.10;
  constexpr std::size_t kMinSupportPoints = 30;

  const std::size_t support_gate =
      std::max(
          kMinSupportPoints,
          static_cast<std::size_t>(
              std::ceil(
                  kSupportRatio *
                  static_cast<double>(max_support_points))));

  // Treat n and -n as the same plane orientation.
  constexpr double kMinNormalSeparationDeg = 15.0;
  constexpr double kPi =
      3.14159265358979323846;

  std::vector<bool> selected(candidates.size(), false);
  std::vector<std::size_t> selected_indices;
  std::vector<BinaryDescriptor> selected_raw_pool;

  const std::size_t target_plane_num =
      std::min(
          static_cast<std::size_t>(
              config_setting_.proj_plane_num_),
          candidates.size());

  auto candidate_has_reliable_support =
      [&](const SelectorCandidate &candidate) -> bool
  {
    return candidate.support_points >= support_gate &&
           candidate.per_plane_after_nms > 0;
  };

  auto min_angle_to_selected_deg =
      [&](const SelectorCandidate &candidate) -> double
  {
    if (selected_indices.empty())
    {
      return 180.0;
    }

    double min_angle_deg = 180.0;

    for (const std::size_t selected_index :
         selected_indices)
    {
      double dot =
          std::abs(
              candidate.normal.dot(
                  candidates[selected_index].normal));

      dot = std::max(-1.0, std::min(1.0, dot));

      const double angle_deg =
          std::acos(dot) *
          180.0 / kPi;

      min_angle_deg =
          std::min(min_angle_deg, angle_deg);
    }

    return min_angle_deg;
  };

  auto pool_after_nms_size =
      [&](const std::vector<BinaryDescriptor> &raw_pool)
      -> std::size_t
  {
    std::vector<BinaryDescriptor> nms_pool = raw_pool;

    if (!nms_pool.empty())
    {
      non_maxi_suppression(nms_pool);
    }

    return nms_pool.size();
  };

  // -----------------------------------------------------------------------
  // Step 1:
  // Pick the strongest reliable plane by its OWN post-NMS Binary yield.
  // This directly addresses frames such as 12/16/25 where the first plane in
  // support order was not the most useful Binary plane.
  // -----------------------------------------------------------------------
  int first_choice = -1;

  auto choose_first =
      [&](const bool enforce_support_gate) -> int
  {
    int best = -1;

    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
      const auto &candidate = candidates[i];

      if (candidate.per_plane_after_nms == 0)
      {
        continue;
      }

      if (enforce_support_gate &&
          !candidate_has_reliable_support(candidate))
      {
        continue;
      }

      if (best < 0 ||
          candidate.per_plane_after_nms >
              candidates[best].per_plane_after_nms ||
          (candidate.per_plane_after_nms ==
               candidates[best].per_plane_after_nms &&
           candidate.support_points >
               candidates[best].support_points))
      {
        best = static_cast<int>(i);
      }
    }

    return best;
  };

  first_choice = choose_first(true);

  bool first_support_fallback = false;

  if (first_choice < 0)
  {
    first_choice = choose_first(false);
    first_support_fallback = true;
  }

  if (first_choice < 0)
  {
    return;
  }

  selected[first_choice] = true;
  selected_indices.push_back(
      static_cast<std::size_t>(first_choice));

  selected_raw_pool.insert(
      selected_raw_pool.end(),
      candidates[first_choice].raw_binary.begin(),
      candidates[first_choice].raw_binary.end());

  std::size_t current_after_nms =
      pool_after_nms_size(selected_raw_pool);

  if (print_debug_info_)
  {
  }

  // -----------------------------------------------------------------------
  // Step 2..N:
  // Greedily add the candidate that contributes the most NEW Binary points
  // after global NMS, while first enforcing:
  //   * plane support gate
  //   * >= 15 degree normal separation from ALL selected planes
  //
  // If no candidate survives the diversity gate, relax only diversity.
  // If no candidate survives support either, relax support as the final
  // fallback.  This prevents sparse scenes from returning no descriptor.
  // -----------------------------------------------------------------------
  while (selected_indices.size() < target_plane_num)
  {
    int best_choice = -1;
    std::size_t best_gain = 0;
    std::size_t best_total_after_nms =
        current_after_nms;
    double best_min_angle_deg = -1.0;

    bool used_diversity_fallback = false;
    bool used_support_fallback = false;

    auto search_best =
        [&](const bool enforce_support_gate,
            const bool enforce_diversity)
    {
      int local_best = -1;
      std::size_t local_best_gain = 0;
      std::size_t local_best_total =
          current_after_nms;
      double local_best_angle = -1.0;

      for (std::size_t i = 0;
           i < candidates.size();
           ++i)
      {
        if (selected[i])
        {
          continue;
        }

        const auto &candidate = candidates[i];

        if (candidate.per_plane_after_nms == 0)
        {
          continue;
        }

        if (enforce_support_gate &&
            !candidate_has_reliable_support(candidate))
        {
          continue;
        }

        const double min_angle_deg =
            min_angle_to_selected_deg(candidate);

        if (enforce_diversity &&
            min_angle_deg <
                kMinNormalSeparationDeg)
        {
          continue;
        }

        std::vector<BinaryDescriptor> trial_pool =
            selected_raw_pool;

        trial_pool.insert(
            trial_pool.end(),
            candidate.raw_binary.begin(),
            candidate.raw_binary.end());

        const std::size_t trial_after_nms =
            pool_after_nms_size(trial_pool);

        const std::size_t gain =
            trial_after_nms >
                    current_after_nms
                ? trial_after_nms -
                      current_after_nms
                : 0;

        const bool better =
            local_best < 0 ||
            gain > local_best_gain ||
            (gain == local_best_gain &&
             candidate.per_plane_after_nms >
                 candidates[local_best]
                     .per_plane_after_nms) ||
            (gain == local_best_gain &&
             candidate.per_plane_after_nms ==
                 candidates[local_best]
                     .per_plane_after_nms &&
             min_angle_deg >
                 local_best_angle) ||
            (gain == local_best_gain &&
             candidate.per_plane_after_nms ==
                 candidates[local_best]
                     .per_plane_after_nms &&
             std::abs(
                 min_angle_deg -
                 local_best_angle) < 1e-9 &&
             candidate.support_points >
                 candidates[local_best]
                     .support_points);

        if (better)
        {
          local_best =
              static_cast<int>(i);
          local_best_gain = gain;
          local_best_total =
              trial_after_nms;
          local_best_angle =
              min_angle_deg;
        }
      }

      best_choice = local_best;
      best_gain = local_best_gain;
      best_total_after_nms =
          local_best_total;
      best_min_angle_deg =
          local_best_angle;
    };

    // Preferred search: reliable support + diverse normal.
    search_best(true, true);

    // If all remaining reliable planes are directionally redundant,
    // relax only the diversity condition.
    if (best_choice < 0)
    {
      used_diversity_fallback = true;
      search_best(true, false);
    }

    // Sparse-scene fallback: allow lower-support planes, but again prefer
    // diversity before finally relaxing it.
    if (best_choice < 0)
    {
      used_support_fallback = true;
      used_diversity_fallback = false;
      search_best(false, true);
    }

    if (best_choice < 0)
    {
      used_support_fallback = true;
      used_diversity_fallback = true;
      search_best(false, false);
    }

    // A plane that adds zero post-NMS features provides no useful new
    // descriptor evidence, so do not spend a projection slot on it.
    if (best_choice < 0 || best_gain == 0)
    {
      break;
    }

    selected[best_choice] = true;
    selected_indices.push_back(
        static_cast<std::size_t>(best_choice));

    selected_raw_pool.insert(
        selected_raw_pool.end(),
        candidates[best_choice].raw_binary.begin(),
        candidates[best_choice].raw_binary.end());

    current_after_nms =
        best_total_after_nms;

    if (print_debug_info_)
    {
    }
  }

  // -----------------------------------------------------------------------
  // FR-SLAM diagnostic only:
  // Show exactly which candidate planes survived the V28.7 selector.
  // -----------------------------------------------------------------------
  if (config_setting_.skip_near_num_ == 0)
  {

    for (std::size_t slot = 0;
         slot < selected_indices.size();
         ++slot)
    {
      const std::size_t selected_index =
          selected_indices[slot];

      const SelectorCandidate &selected_candidate =
          candidates[selected_index];

    }
  }

  // Keep the old human-readable reference-plane messages for compatibility.
  for (const std::size_t selected_index :
       selected_indices)
  {
  }

  std::vector<BinaryDescriptor> temp_binary_list =
      selected_raw_pool;

  const std::size_t raw_binary_sum =
      temp_binary_list.size();

  if (!temp_binary_list.empty())
  {
    non_maxi_suppression(temp_binary_list);
  }

  const std::size_t after_nms =
      temp_binary_list.size();

  if (config_setting_.useful_corner_num_ >
      static_cast<int>(temp_binary_list.size()))
  {
    binary_descriptor_list =
        temp_binary_list;
  }
  else
  {
    std::sort(temp_binary_list.begin(),
              temp_binary_list.end(),
              binary_greater_sort);

    for (std::size_t i = 0;
         i <
         static_cast<std::size_t>(
             config_setting_.useful_corner_num_);
         ++i)
    {
      binary_descriptor_list.push_back(
          temp_binary_list[i]);
    }
  }

  if (print_debug_info_)
  {

    // Preserve the V28.5 summary marker, now reporting the ACTIVE V28.7
    // selector result rather than the old first-N selector.
  }

  return;
}

void BtcDescManager::extract_binary(
    const Eigen::Vector3d &project_center,
    const Eigen::Vector3d &project_normal,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
    std::vector<BinaryDescriptor> &binary_list)
{
  binary_list.clear();
  double binary_min_dis = config_setting_.summary_min_thre_;
  double resolution = config_setting_.proj_image_resolution_;
  double dis_threshold_min = config_setting_.proj_dis_min_;
  double dis_threshold_max = config_setting_.proj_dis_max_;
  double high_inc = config_setting_.proj_image_high_inc_;
  bool line_filter_enable = config_setting_.line_filter_enable_;
  double A = project_normal[0];
  double B = project_normal[1];
  double C = project_normal[2];
  double D =
      -(A * project_center[0] + B * project_center[1] + C * project_center[2]);
  std::vector<Eigen::Vector3d> projection_points;

  // ===================================================================
  // FR-SLAM V31.8 — rotation/translation-equivariant projection basis.
  //
  // Do NOT construct the tangent basis from a fixed global +X axis.
  // Instead use intrinsic scene geometry:
  //
  //     tangent_hint = mean(gated_points) - project_center
  //
  // and remove its normal component.
  //
  // Under a rigid transform:
  //     p' = R p + t
  //     c' = R c + t
  //     n' = R n
  //
  // therefore:
  //     mean(p') - c' = R (mean(p) - c)
  //
  // and the tangent basis rotates together with the geometry.
  //
  // If the centroid direction degenerates, use the farthest gated point
  // from the plane center. A fixed-axis fallback is used only as a final
  // numerical safety fallback.
  // ===================================================================

  const double basis_normal_sq = project_normal.squaredNorm();

  Eigen::Vector3d tangent_hint = Eigen::Vector3d::Zero();
  std::size_t tangent_count = 0;

  for (std::size_t basis_i = 0;
       basis_i < input_cloud->size();
       ++basis_i)
  {
    const double basis_px = input_cloud->points[basis_i].x;
    const double basis_py = input_cloud->points[basis_i].y;
    const double basis_pz = input_cloud->points[basis_i].z;

    const double basis_dis =
        basis_px * A +
        basis_py * B +
        basis_pz * C +
        D;

    if (basis_dis < dis_threshold_min ||
        basis_dis > dis_threshold_max)
    {
      continue;
    }

    tangent_hint +=
        Eigen::Vector3d(
            basis_px,
            basis_py,
            basis_pz) -
        project_center;

    ++tangent_count;
  }

  if (tangent_count > 0)
  {
    tangent_hint /=
        static_cast<double>(tangent_count);
  }

  if (basis_normal_sq > 1e-12)
  {
    tangent_hint -=
        project_normal *
        (project_normal.dot(tangent_hint) /
         basis_normal_sq);
  }

  const double centroid_tangent_norm =
      tangent_hint.norm();

  Eigen::Vector3d x_axis =
      Eigen::Vector3d::Zero();

  const char *basis_source = "CENTROID";

  if (centroid_tangent_norm > 1e-6)
  {
    x_axis = tangent_hint / centroid_tangent_norm;
  }
  else
  {
    basis_source = "FARTHEST_POINT";

    double best_tangent_sq = -1.0;

    for (std::size_t basis_i = 0;
         basis_i < input_cloud->size();
         ++basis_i)
    {
      const double basis_px = input_cloud->points[basis_i].x;
      const double basis_py = input_cloud->points[basis_i].y;
      const double basis_pz = input_cloud->points[basis_i].z;

      const double basis_dis =
          basis_px * A +
          basis_py * B +
          basis_pz * C +
          D;

      if (basis_dis < dis_threshold_min ||
          basis_dis > dis_threshold_max)
      {
        continue;
      }

      Eigen::Vector3d candidate_tangent(
          basis_px - project_center[0],
          basis_py - project_center[1],
          basis_pz - project_center[2]);

      if (basis_normal_sq > 1e-12)
      {
        candidate_tangent -=
            project_normal *
            (project_normal.dot(candidate_tangent) /
             basis_normal_sq);
      }

      const double candidate_tangent_sq =
          candidate_tangent.squaredNorm();

      if (candidate_tangent_sq > best_tangent_sq)
      {
        best_tangent_sq = candidate_tangent_sq;
        x_axis = candidate_tangent;
      }
    }

    if (x_axis.norm() > 1e-6)
    {
      x_axis.normalize();
    }
    else
    {
      basis_source = "GLOBAL_FALLBACK";

      Eigen::Vector3d reference_axis =
          Eigen::Vector3d::UnitX();

      double reference_alignment =
          std::abs(project_normal.dot(reference_axis));

      const double y_alignment =
          std::abs(
              project_normal.dot(
                  Eigen::Vector3d::UnitY()));

      if (y_alignment < reference_alignment)
      {
        reference_axis = Eigen::Vector3d::UnitY();
        reference_alignment = y_alignment;
      }

      const double z_alignment =
          std::abs(
              project_normal.dot(
                  Eigen::Vector3d::UnitZ()));

      if (z_alignment < reference_alignment)
      {
        reference_axis = Eigen::Vector3d::UnitZ();
      }

      x_axis =
          reference_axis -
          project_normal *
          (project_normal.dot(reference_axis) /
           basis_normal_sq);

      x_axis.normalize();
    }
  }

  Eigen::Vector3d y_axis =
      project_normal.cross(x_axis);

  y_axis.normalize();

  if (print_debug_info_ &&
      config_setting_.skip_near_num_ == 0)
  {
  }
  double ax = x_axis[0];
  double bx = x_axis[1];
  double cx = x_axis[2];
  double dx = -(ax * project_center[0] + bx * project_center[1] +
                cx * project_center[2]);
  double ay = y_axis[0];
  double by = y_axis[1];
  double cy = y_axis[2];
  double dy = -(ay * project_center[0] + by * project_center[1] +
                cy * project_center[2]);
  std::vector<Eigen::Vector2d> point_list_2d;
  pcl::PointCloud<pcl::PointXYZ> point_list_3d;
  std::vector<double> dis_list_2d;

  // FR-SLAM diagnostic only.
  std::size_t diag_gate_in = 0;
  std::size_t diag_below_min = 0;
  std::size_t diag_above_max = 0;
  std::size_t diag_negative = 0;
  std::size_t diag_nonnegative = 0;

  double diag_dis_min = 1e100;
  double diag_dis_max = -1e100;

  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    double x = input_cloud->points[i].x;
    double y = input_cloud->points[i].y;
    double z = input_cloud->points[i].z;
    double dis = x * A + y * B + z * C + D;

    diag_dis_min =
        std::min(diag_dis_min, dis);

    diag_dis_max =
        std::max(diag_dis_max, dis);

    if (dis < 0.0)
    {
      ++diag_negative;
    }
    else
    {
      ++diag_nonnegative;
    }

    if (dis < dis_threshold_min)
    {
      ++diag_below_min;
    }
    else if (dis > dis_threshold_max)
    {
      ++diag_above_max;
    }
    else
    {
      ++diag_gate_in;
    }

    pcl::PointXYZ pi;
    if (dis < dis_threshold_min || dis > dis_threshold_max)
    {
      continue;
    }
    else
    {
      if (dis > dis_threshold_min && dis <= dis_threshold_max)
      {
        pi.x = x;
        pi.y = y;
        pi.z = z;
      }
    }
    Eigen::Vector3d cur_project;

    cur_project[0] = (-A * (B * y + C * z + D) + x * (B * B + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[1] = (-B * (A * x + C * z + D) + y * (A * A + C * C)) /
                     (A * A + B * B + C * C);
    cur_project[2] = (-C * (A * x + B * y + D) + z * (A * A + B * B)) /
                     (A * A + B * B + C * C);
    pcl::PointXYZ p;
    p.x = cur_project[0];
    p.y = cur_project[1];
    p.z = cur_project[2];
    double project_x =
        cur_project[0] * ay + cur_project[1] * by + cur_project[2] * cy + dy;
    double project_y =
        cur_project[0] * ax + cur_project[1] * bx + cur_project[2] * cx + dx;
    Eigen::Vector2d p_2d(project_x, project_y);
    point_list_2d.push_back(p_2d);
    dis_list_2d.push_back(dis);
    point_list_3d.points.push_back(pi);
  }

  if (print_debug_info_ &&
      config_setting_.skip_near_num_ == 0)
  {
  }

  double min_x = 10;
  double max_x = -10;
  double min_y = 10;
  double max_y = -10;
  if (point_list_2d.size() <= 5)
  {
    return;
  }
  for (auto pi : point_list_2d)
  {
    if (pi[0] < min_x)
    {
      min_x = pi[0];
    }
    if (pi[0] > max_x)
    {
      max_x = pi[0];
    }
    if (pi[1] < min_y)
    {
      min_y = pi[1];
    }
    if (pi[1] > max_y)
    {
      max_y = pi[1];
    }
  }
  // segment project cloud
  int segmen_base_num = 5;
  double segmen_len = segmen_base_num * resolution;
  int x_segment_num = (max_x - min_x) / segmen_len + 1;
  int y_segment_num = (max_y - min_y) / segmen_len + 1;
  int x_axis_len = (int)((max_x - min_x) / resolution + segmen_base_num);
  int y_axis_len = (int)((max_y - min_y) / resolution + segmen_base_num);

  std::vector<double> **dis_container = new std::vector<double> *[x_axis_len];
  BinaryDescriptor **binary_container = new BinaryDescriptor *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++)
  {
    dis_container[i] = new std::vector<double>[y_axis_len];
    binary_container[i] = new BinaryDescriptor[y_axis_len];
  }
  double **img_count = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++)
  {
    img_count[i] = new double[y_axis_len];
  }
  double **dis_array = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++)
  {
    dis_array[i] = new double[y_axis_len];
  }
  double **mean_x_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++)
  {
    mean_x_list[i] = new double[y_axis_len];
  }
  double **mean_y_list = new double *[x_axis_len];
  for (int i = 0; i < x_axis_len; i++)
  {
    mean_y_list[i] = new double[y_axis_len];
  }
  for (int x = 0; x < x_axis_len; x++)
  {
    for (int y = 0; y < y_axis_len; y++)
    {
      img_count[x][y] = 0;
      mean_x_list[x][y] = 0;
      mean_y_list[x][y] = 0;
      dis_array[x][y] = 0;
      std::vector<double> single_dis_container;
      dis_container[x][y] = single_dis_container;
    }
  }

  for (size_t i = 0; i < point_list_2d.size(); i++)
  {
    int x_index = (int)((point_list_2d[i][0] - min_x) / resolution);
    int y_index = (int)((point_list_2d[i][1] - min_y) / resolution);
    mean_x_list[x_index][y_index] += point_list_2d[i][0];
    mean_y_list[x_index][y_index] += point_list_2d[i][1];
    img_count[x_index][y_index]++;
    dis_container[x_index][y_index].push_back(dis_list_2d[i]);
  }

  for (int x = 0; x < x_axis_len; x++)
  {
    for (int y = 0; y < y_axis_len; y++)
    {
      // calc segment dis array
      if (img_count[x][y] > 0)
      {
        int cut_num = (dis_threshold_max - dis_threshold_min) / high_inc;
        std::vector<bool> occup_list;
        std::vector<double> cnt_list;
        BinaryDescriptor single_binary;
        for (size_t i = 0; i < cut_num; i++)
        {
          cnt_list.push_back(0);
          occup_list.push_back(false);
        }
        for (size_t j = 0; j < dis_container[x][y].size(); j++)
        {
          int cnt_index =
              (dis_container[x][y][j] - dis_threshold_min) / high_inc;
          cnt_list[cnt_index]++;
        }
        double segmnt_dis = 0;
        for (size_t i = 0; i < cut_num; i++)
        {
          if (cnt_list[i] >= 1)
          {
            segmnt_dis++;
            occup_list[i] = true;
          }
        }
        dis_array[x][y] = segmnt_dis;
        single_binary.occupy_array_ = occup_list;
        single_binary.summary_ = segmnt_dis;
        binary_container[x][y] = single_binary;
      }
    }
  }

  // filter by distance
  std::vector<double> max_dis_list;
  std::vector<int> max_dis_x_index_list;
  std::vector<int> max_dis_y_index_list;

  for (int x_segment_index = 0; x_segment_index < x_segment_num;
       x_segment_index++)
  {
    for (int y_segment_index = 0; y_segment_index < y_segment_num;
         y_segment_index++)
    {
      double max_dis = 0;
      int max_dis_x_index = -10;
      int max_dis_y_index = -10;
      for (int x_index = x_segment_index * segmen_base_num;
           x_index < (x_segment_index + 1) * segmen_base_num; x_index++)
      {
        for (int y_index = y_segment_index * segmen_base_num;
             y_index < (y_segment_index + 1) * segmen_base_num; y_index++)
        {
          if (dis_array[x_index][y_index] > max_dis)
          {
            max_dis = dis_array[x_index][y_index];
            max_dis_x_index = x_index;
            max_dis_y_index = y_index;
          }
        }
      }
      if (max_dis >= binary_min_dis)
      {
        max_dis_list.push_back(max_dis);
        max_dis_x_index_list.push_back(max_dis_x_index);
        max_dis_y_index_list.push_back(max_dis_y_index);
      }
    }
  }
  // calc line or not
  std::vector<Eigen::Vector2i> direction_list;
  Eigen::Vector2i d(0, 1);
  direction_list.push_back(d);
  d << 1, 0;
  direction_list.push_back(d);
  d << 1, 1;
  direction_list.push_back(d);
  d << 1, -1;
  direction_list.push_back(d);
  for (size_t i = 0; i < max_dis_list.size(); i++)
  {
    Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
    if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
        p[1] >= y_axis_len - 1)
    {
      continue;
    }
    bool is_add = true;

    if (line_filter_enable)
    {
      for (int j = 0; j < 4; j++)
      {
        Eigen::Vector2i p(max_dis_x_index_list[i], max_dis_y_index_list[i]);
        if (p[0] <= 0 || p[0] >= x_axis_len - 1 || p[1] <= 0 ||
            p[1] >= y_axis_len - 1)
        {
          continue;
        }
        Eigen::Vector2i p1 = p + direction_list[j];
        Eigen::Vector2i p2 = p - direction_list[j];
        double threshold = dis_array[p[0]][p[1]] - 3;
        if (dis_array[p1[0]][p1[1]] >= threshold)
        {
          if (dis_array[p2[0]][p2[1]] >= 0.5 * dis_array[p[0]][p[1]])
          {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold)
        {
          if (dis_array[p1[0]][p1[1]] >= 0.5 * dis_array[p[0]][p[1]])
          {
            is_add = false;
          }
        }
        if (dis_array[p1[0]][p1[1]] >= threshold)
        {
          if (dis_array[p2[0]][p2[1]] >= threshold)
          {
            is_add = false;
          }
        }
        if (dis_array[p2[0]][p2[1]] >= threshold)
        {
          if (dis_array[p1[0]][p1[1]] >= threshold)
          {
            is_add = false;
          }
        }
      }
    }
    if (is_add)
    {
      double px =
          mean_x_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      double py =
          mean_y_list[max_dis_x_index_list[i]][max_dis_y_index_list[i]] /
          img_count[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      Eigen::Vector3d coord = py * x_axis + px * y_axis + project_center;
      pcl::PointXYZ pi;
      pi.x = coord[0];
      pi.y = coord[1];
      pi.z = coord[2];
      BinaryDescriptor single_binary =
          binary_container[max_dis_x_index_list[i]][max_dis_y_index_list[i]];
      single_binary.location_ = coord;
      binary_list.push_back(single_binary);
    }
  }
  for (int i = 0; i < x_axis_len; i++)
  {
    delete[] binary_container[i];
    delete[] dis_container[i];
    delete[] img_count[i];
    delete[] dis_array[i];
    delete[] mean_x_list[i];
    delete[] mean_y_list[i];
  }
  delete[] binary_container;
  delete[] dis_container;
  delete[] img_count;
  delete[] dis_array;
  delete[] mean_x_list;
  delete[] mean_y_list;
}

void BtcDescManager::non_maxi_suppression(
    std::vector<BinaryDescriptor> &binary_list)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr prepare_key_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  std::vector<int> pre_count_list;
  std::vector<bool> is_add_list;
  for (auto var : binary_list)
  {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    prepare_key_cloud->push_back(pi);
    pre_count_list.push_back(var.summary_);
    is_add_list.push_back(true);
  }
  kd_tree.setInputCloud(prepare_key_cloud);
  std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  double radius = config_setting_.non_max_suppression_radius_;
  for (size_t i = 0; i < prepare_key_cloud->size(); i++)
  {
    pcl::PointXYZ searchPoint = prepare_key_cloud->points[i];
    if (kd_tree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch,
                             pointRadiusSquaredDistance) > 0)
    {
      Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
      for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j)
      {
        Eigen::Vector3d pj(
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].x,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].y,
            prepare_key_cloud->points[pointIdxRadiusSearch[j]].z);
        if (pointIdxRadiusSearch[j] == i)
        {
          continue;
        }
        if (pre_count_list[i] <= pre_count_list[pointIdxRadiusSearch[j]])
        {
          is_add_list[i] = false;
        }
      }
    }
  }
  std::vector<BinaryDescriptor> pass_binary_list;
  for (size_t i = 0; i < is_add_list.size(); i++)
  {
    if (is_add_list[i])
    {
      pass_binary_list.push_back(binary_list[i]);
    }
  }
  binary_list.clear();
  for (auto var : pass_binary_list)
  {
    binary_list.push_back(var);
  }
  return;
}

void BtcDescManager::generate_btc(
    const std::vector<BinaryDescriptor> &binary_list, const int &frame_id,
    std::vector<BTC> &btc_list)
{
  double scale = 1.0 / config_setting_.std_side_resolution_;
  std::unordered_map<VOXEL_LOC, bool> feat_map;
  pcl::PointCloud<pcl::PointXYZ> key_cloud;
  for (auto var : binary_list)
  {
    pcl::PointXYZ pi;
    pi.x = var.location_[0];
    pi.y = var.location_[1];
    pi.z = var.location_[2];
    key_cloud.push_back(pi);
  }
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  kd_tree->setInputCloud(key_cloud.makeShared());
  int K = config_setting_.descriptor_near_num_;
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);
  for (size_t i = 0; i < key_cloud.size(); i++)
  {
    pcl::PointXYZ searchPoint = key_cloud.points[i];
    if (kd_tree->nearestKSearch(searchPoint, K, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0)
    {
      for (int m = 1; m < K - 1; m++)
      {
        for (int n = m + 1; n < K; n++)
        {
          pcl::PointXYZ p1 = searchPoint;
          pcl::PointXYZ p2 = key_cloud.points[pointIdxNKNSearch[m]];
          pcl::PointXYZ p3 = key_cloud.points[pointIdxNKNSearch[n]];
          double a = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) +
                          pow(p1.z - p2.z, 2));
          double b = sqrt(pow(p1.x - p3.x, 2) + pow(p1.y - p3.y, 2) +
                          pow(p1.z - p3.z, 2));
          double c = sqrt(pow(p3.x - p2.x, 2) + pow(p3.y - p2.y, 2) +
                          pow(p3.z - p2.z, 2));
          if (a > config_setting_.descriptor_max_len_ ||
              b > config_setting_.descriptor_max_len_ ||
              c > config_setting_.descriptor_max_len_ ||
              a < config_setting_.descriptor_min_len_ ||
              b < config_setting_.descriptor_min_len_ ||
              c < config_setting_.descriptor_min_len_)
          {
            continue;
          }
          double temp;
          Eigen::Vector3d A, B, C;
          Eigen::Vector3i l1, l2, l3;
          Eigen::Vector3i l_temp;
          l1 << 1, 2, 0;
          l2 << 1, 0, 3;
          l3 << 0, 2, 3;
          if (a > b)
          {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (b > c)
          {
            temp = b;
            b = c;
            c = temp;
            l_temp = l2;
            l2 = l3;
            l3 = l_temp;
          }
          if (a > b)
          {
            temp = a;
            a = b;
            b = temp;
            l_temp = l1;
            l1 = l2;
            l2 = l_temp;
          }
          if (fabs(c - (a + b)) < 0.2)
          {
            continue;
          }

          pcl::PointXYZ d_p;
          d_p.x = a * 1000;
          d_p.y = b * 1000;
          d_p.z = c * 1000;
          VOXEL_LOC position((int64_t)d_p.x, (int64_t)d_p.y, (int64_t)d_p.z);
          auto iter = feat_map.find(position);
          Eigen::Vector3d normal_1, normal_2, normal_3;
          BinaryDescriptor binary_A;
          BinaryDescriptor binary_B;
          BinaryDescriptor binary_C;
          if (iter == feat_map.end())
          {
            if (l1[0] == l2[0])
            {
              A << p1.x, p1.y, p1.z;
              binary_A = binary_list[i];
            }
            else if (l1[1] == l2[1])
            {
              A << p2.x, p2.y, p2.z;
              binary_A = binary_list[pointIdxNKNSearch[m]];
            }
            else
            {
              A << p3.x, p3.y, p3.z;
              binary_A = binary_list[pointIdxNKNSearch[n]];
            }
            if (l1[0] == l3[0])
            {
              B << p1.x, p1.y, p1.z;
              binary_B = binary_list[i];
            }
            else if (l1[1] == l3[1])
            {
              B << p2.x, p2.y, p2.z;
              binary_B = binary_list[pointIdxNKNSearch[m]];
            }
            else
            {
              B << p3.x, p3.y, p3.z;
              binary_B = binary_list[pointIdxNKNSearch[n]];
            }
            if (l2[0] == l3[0])
            {
              C << p1.x, p1.y, p1.z;
              binary_C = binary_list[i];
            }
            else if (l2[1] == l3[1])
            {
              C << p2.x, p2.y, p2.z;
              binary_C = binary_list[pointIdxNKNSearch[m]];
            }
            else
            {
              C << p3.x, p3.y, p3.z;
              binary_C = binary_list[pointIdxNKNSearch[n]];
            }
            BTC single_descriptor;
            single_descriptor.binary_A_ = binary_A;
            single_descriptor.binary_B_ = binary_B;
            single_descriptor.binary_C_ = binary_C;
            single_descriptor.center_ = (A + B + C) / 3;
            single_descriptor.triangle_ << scale * a, scale * b, scale * c;
            single_descriptor.angle_[0] = fabs(5 * normal_1.dot(normal_2));
            single_descriptor.angle_[1] = fabs(5 * normal_1.dot(normal_3));
            single_descriptor.angle_[2] = fabs(5 * normal_3.dot(normal_2));
            // single_descriptor.angle << 0, 0, 0;
            single_descriptor.frame_number_ = frame_id;
            // single_descriptor.score_frame_.push_back(frame_number);
            Eigen::Matrix3d triangle_positon;
            triangle_positon.block<3, 1>(0, 0) = A;
            triangle_positon.block<3, 1>(0, 1) = B;
            triangle_positon.block<3, 1>(0, 2) = C;
            // single_descriptor.position_list_.push_back(triangle_positon);
            // single_descriptor.triangle_scale_ = scale;
            feat_map[position] = true;
            btc_list.push_back(single_descriptor);
          }
        }
      }
    }
  }
}

void BtcDescManager::candidate_selector(
    const std::vector<BTC> &current_STD_list,
    std::vector<BTCMatchList> &candidate_matcher_vec)
{
  int current_frame_id = current_STD_list[0].frame_number_;
  int outlier = 0;
  double max_dis = 50;
  double match_array[20000] = {0};
  std::vector<std::pair<BTC, BTC>> match_list;
  std::vector<int> match_list_index;
  std::vector<Eigen::Vector3i> voxel_round;
  for (int x = -1; x <= 1; x++)
  {
    for (int y = -1; y <= 1; y++)
    {
      for (int z = -1; z <= 1; z++)
      {
        Eigen::Vector3i voxel_inc(x, y, z);
        voxel_round.push_back(voxel_inc);
      }
    }
  }
  std::vector<bool> useful_match(current_STD_list.size());
  std::vector<std::vector<size_t>> useful_match_index(current_STD_list.size());
  std::vector<std::vector<BTC_LOC>> useful_match_position(
      current_STD_list.size());
  std::vector<size_t> index(current_STD_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_match[i] = false;
  }
  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();

  int query_num = 0;
  int pass_num = 0;
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i)
      {
        BTC descriptor = current_STD_list[i];
        BTC_LOC position;
        int best_index = 0;
        BTC_LOC best_position;
        double dis_threshold =
            descriptor.triangle_.norm() *
            config_setting_.rough_dis_threshold_; // old 0.005
        for (auto voxel_inc : voxel_round)
        {
          position.x = (int)(descriptor.triangle_[0] + voxel_inc[0]);
          position.y = (int)(descriptor.triangle_[1] + voxel_inc[1]);
          position.z = (int)(descriptor.triangle_[2] + voxel_inc[2]);
          Eigen::Vector3d voxel_center((double)position.x + 0.5,
                                       (double)position.y + 0.5,
                                       (double)position.z + 0.5);
          if ((descriptor.triangle_ - voxel_center).norm() < 1.5)
          {
            auto iter = data_base_.find(position);
            if (iter != data_base_.end())
            {
              bool is_push_position = false;
              for (size_t j = 0; j < data_base_[position].size(); j++)
              {
                if ((descriptor.frame_number_ -
                     data_base_[position][j].frame_number_) >
                    config_setting_.skip_near_num_)
                {
                  double dis =
                      (descriptor.triangle_ - data_base_[position][j].triangle_)
                          .norm();
                  if (dis < dis_threshold)
                  {
                    double similarity =
                        (binary_similarity(descriptor.binary_A_,
                                           data_base_[position][j].binary_A_) +
                         binary_similarity(descriptor.binary_B_,
                                           data_base_[position][j].binary_B_) +
                         binary_similarity(descriptor.binary_C_,
                                           data_base_[position][j].binary_C_)) /
                        3;
                    if (similarity > config_setting_.similarity_threshold_)
                    {
                      useful_match[i] = true;
                      useful_match_position[i].push_back(position);
                      useful_match_index[i].push_back(j);
                    }
                  }
                }
              }
            }
          }
        }
      });
  std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i>>
      index_recorder;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < useful_match.size(); i++)
  {
    if (useful_match[i])
    {
      for (size_t j = 0; j < useful_match_index[i].size(); j++)
      {
        match_array[data_base_[useful_match_position[i][j]]
                              [useful_match_index[i][j]]
                                  .frame_number_] += 1;
        Eigen::Vector2i match_index(i, j);
        index_recorder.push_back(match_index);
        // match_list.push_back(single_match_pair);
        match_list_index.push_back(
            data_base_[useful_match_position[i][j]][useful_match_index[i][j]]
                .frame_number_);
      }
    }
  }
  bool multi_thread_en = false;
  if (multi_thread_en)
  {
    std::for_each(
        std::execution::par_unseq, index.begin(), index.end(),
        [&](const size_t &i)
        {
          if (useful_match[i])
          {
            std::pair<BTC, BTC> single_match_pair;
            single_match_pair.first = current_STD_list[i];
            for (size_t j = 0; j < useful_match_index[i].size(); j++)
            {
              single_match_pair.second = data_base_[useful_match_position[i][j]]
                                                   [useful_match_index[i][j]];
              mylock.lock();
              match_array[single_match_pair.second.frame_number_] += 1;
              match_list.push_back(single_match_pair);
              match_list_index.push_back(
                  single_match_pair.second.frame_number_);
              mylock.unlock();
            }
          }
        });
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  //           << std::endl;
  // use index recorder

  // -----------------------------------------------------------------------
  constexpr bool kEnableBtcV284MatchStageDiagnostics = false;
  if (kEnableBtcV284MatchStageDiagnostics && print_debug_info_)
  {
  // FR-SLAM V28.4 match-stage diagnostic only.
  //
  // This is a separate SERIAL second pass over the same query BTCs and DB.
  // It does NOT modify match_array, candidate_matcher_vec, or data_base_.
  //
  // Per historical frame:
  //   bucket_hits      = neighboring triangle hash bucket hit after time gate
  //   rough_pass       = triangle distance passes rough_dis_threshold_
  //   similarity_pass  = binary similarity passes similarity_threshold_
  // -----------------------------------------------------------------------
  if (print_debug_info_)
  {
    std::vector<int> diag_bucket_hits(20000, 0);
    std::vector<int> diag_rough_pass(20000, 0);
    std::vector<int> diag_similarity_pass(20000, 0);

    std::vector<double> diag_best_rel_dis(20000, 1e9);
    std::vector<double> diag_best_similarity(20000, -1.0);

    std::size_t diag_bucket_hits_before_time_gate = 0;
    std::size_t diag_skip_near_rejects = 0;

    for (size_t diag_i = 0;
         diag_i < current_STD_list.size();
         ++diag_i)
    {
      const BTC &diag_descriptor =
          current_STD_list[diag_i];

      const double diag_triangle_norm =
          diag_descriptor.triangle_.norm();

      const double diag_dis_threshold =
          diag_triangle_norm *
          config_setting_.rough_dis_threshold_;

      for (const auto &diag_voxel_inc :
           voxel_round)
      {
        BTC_LOC diag_position;

        diag_position.x =
            static_cast<int>(
                diag_descriptor.triangle_[0] +
                diag_voxel_inc[0]);
        diag_position.y =
            static_cast<int>(
                diag_descriptor.triangle_[1] +
                diag_voxel_inc[1]);
        diag_position.z =
            static_cast<int>(
                diag_descriptor.triangle_[2] +
                diag_voxel_inc[2]);

        const Eigen::Vector3d diag_voxel_center(
            static_cast<double>(diag_position.x) + 0.5,
            static_cast<double>(diag_position.y) + 0.5,
            static_cast<double>(diag_position.z) + 0.5);

        if ((diag_descriptor.triangle_ -
             diag_voxel_center)
                .norm() >= 1.5)
        {
          continue;
        }

        const auto diag_iter =
            data_base_.find(diag_position);

        if (diag_iter == data_base_.end())
        {
          continue;
        }

        const auto &diag_history_bucket =
            diag_iter->second;

        for (size_t diag_j = 0;
             diag_j < diag_history_bucket.size();
             ++diag_j)
        {
          const BTC &diag_history =
              diag_history_bucket[diag_j];

          const int diag_historical_frame =
              diag_history.frame_number_;

          if (diag_historical_frame < 0 ||
              diag_historical_frame >= 20000)
          {
            continue;
          }

          ++diag_bucket_hits_before_time_gate;

          if ((diag_descriptor.frame_number_ -
               diag_historical_frame) <=
              config_setting_.skip_near_num_)
          {
            ++diag_skip_near_rejects;
            continue;
          }

          ++diag_bucket_hits[diag_historical_frame];

          const double diag_dis =
              (diag_descriptor.triangle_ -
               diag_history.triangle_)
                  .norm();

          double diag_rel_dis = 1e9;

          if (diag_triangle_norm > 1e-12)
          {
            diag_rel_dis =
                diag_dis /
                diag_triangle_norm;
          }

          if (diag_rel_dis <
              diag_best_rel_dis[diag_historical_frame])
          {
            diag_best_rel_dis[diag_historical_frame] =
                diag_rel_dis;
          }

          if (diag_dis >=
              diag_dis_threshold)
          {
            continue;
          }

          ++diag_rough_pass[diag_historical_frame];

          const double diag_similarity =
              (binary_similarity(
                   diag_descriptor.binary_A_,
                   diag_history.binary_A_) +
               binary_similarity(
                   diag_descriptor.binary_B_,
                   diag_history.binary_B_) +
               binary_similarity(
                   diag_descriptor.binary_C_,
                   diag_history.binary_C_)) /
              3.0;

          if (diag_similarity >
              diag_best_similarity[diag_historical_frame])
          {
            diag_best_similarity[diag_historical_frame] =
                diag_similarity;
          }

          if (diag_similarity >
              config_setting_.similarity_threshold_)
          {
            ++diag_similarity_pass[diag_historical_frame];
          }
        }
      }
    }

    int diag_bucket_history_count = 0;
    int diag_rough_history_count = 0;
    int diag_similarity_history_count = 0;

    for (int diag_frame = 0;
         diag_frame < 20000;
         ++diag_frame)
    {
      if (diag_bucket_hits[diag_frame] <= 0)
      {
        continue;
      }

      ++diag_bucket_history_count;

      if (diag_rough_pass[diag_frame] > 0)
      {
        ++diag_rough_history_count;
      }

      if (diag_similarity_pass[diag_frame] > 0)
      {
        ++diag_similarity_history_count;
      }

    }

  }

  // -----------------------------------------------------------------------
  }

  // FR-SLAM V28.3 diagnostic only.
  //
  // Print the strongest historical-frame BTC rough vote counts BEFORE the
  // official max_vote >= 4 gate below. This does not modify match_array.
  // -----------------------------------------------------------------------
  if (print_debug_info_)
  {
    constexpr int kDebugTopVoteCount = 5;

    double debug_vote[kDebugTopVoteCount] = {0, 0, 0, 0, 0};
    int debug_frame[kDebugTopVoteCount] = {-1, -1, -1, -1, -1};

    for (int frame_id = 0; frame_id < 20000; ++frame_id)
    {
      const double vote = match_array[frame_id];

      if (vote <= 0)
      {
        continue;
      }

      for (int rank = 0; rank < kDebugTopVoteCount; ++rank)
      {
        if (vote > debug_vote[rank])
        {
          for (int move = kDebugTopVoteCount - 1; move > rank; --move)
          {
            debug_vote[move] = debug_vote[move - 1];
            debug_frame[move] = debug_frame[move - 1];
          }

          debug_vote[rank] = vote;
          debug_frame[rank] = frame_id;
          break;
        }
      }
    }

    bool printed_any_vote = false;

    for (int rank = 0; rank < kDebugTopVoteCount; ++rank)
    {
      if (debug_frame[rank] < 0)
      {
        continue;
      }

      printed_any_vote = true;

    }

    if (!printed_any_vote)
    {
    }
  }
  // =====================================================================
  // FR-SLAM BTC V28.9 BORDERLINE DIAGNOSTIC ONLY
  //
  // Purpose:
  //   Inspect the expected true revisit:
  //       current Submap 30 -> historical Submap 26
  //
  // IMPORTANT:
  //   - DOES NOT change official selector threshold (vote >= 5).
  //   - DOES NOT push this candidate into candidate_matcher_vec.
  //   - DOES NOT affect loop_result / PGO.
  //   - Only runs candidate_verify() once for diagnosis.
  // =====================================================================
  if (print_debug_info_)
  {
    constexpr int kDiagCurrentFrame = 30;
    constexpr int kDiagHistoricalFrame = 26;

    if (current_frame_id == kDiagCurrentFrame)
    {
      const double diag_selector_vote =
          match_array[kDiagHistoricalFrame];


      if (diag_selector_vote > 0.0)
      {
        BTCMatchList diag_match_triangle_list;

        diag_match_triangle_list.match_frame_ =
            kDiagHistoricalFrame;

        diag_match_triangle_list.match_id_.first =
            current_frame_id;

        diag_match_triangle_list.match_id_.second =
            kDiagHistoricalFrame;

        // Reconstruct EXACTLY the same triangle-pair list that the
        // official selector would build if this historical frame
        // passed vote >= 5.
        for (size_t diag_i = 0;
             diag_i < index_recorder.size();
             ++diag_i)
        {
          if (match_list_index[diag_i] !=
              kDiagHistoricalFrame)
          {
            continue;
          }

          const int query_index =
              index_recorder[diag_i][0];

          const int match_index =
              index_recorder[diag_i][1];

          std::pair<BTC, BTC> diag_match_pair;

          diag_match_pair.first =
              current_STD_list[query_index];

          diag_match_pair.second =
              data_base_[useful_match_position[query_index]
                                              [match_index]]
                        [useful_match_index[query_index]
                                           [match_index]];

          diag_match_triangle_list.match_list_.push_back(
              diag_match_pair);
        }


        if (!diag_match_triangle_list.match_list_.empty())
        {
          double diag_verify_score = -1.0;

          std::pair<Eigen::Vector3d, Eigen::Matrix3d>
              diag_relative_pose;

          diag_relative_pose.first.setZero();
          diag_relative_pose.second.setIdentity();

          std::vector<std::pair<BTC, BTC>>
              diag_success_match_vec;

          candidate_verify(
              diag_match_triangle_list,
              diag_verify_score,
              diag_relative_pose,
              diag_success_match_vec);

          const bool diag_pose_valid =
              diag_verify_score >= 0.0;

          double diag_rotation_deg = 0.0;

          if (diag_pose_valid)
          {
            Eigen::AngleAxisd diag_angle_axis(
                diag_relative_pose.second);

            diag_rotation_deg =
                std::abs(diag_angle_axis.angle()) *
                57.29577951308232;
          }

        }
      }
    }
  }

  // ============================================================
  // FR-SLAM BTC_PENDING_RESCUE_V31_11
  //
  // Strict global discovery:
  //     rough vote >= 4
  //
  // Pending-track continuation:
  //     ONLY historical frames inside the pending range
  //     may enter candidate_verify with rough vote >= 3.
  //
  // IMPORTANT:
  //     candidate_verify() keeps its original >=4 SE(3)
  //     consensus gate.
  // ============================================================
  for (int cnt = 0;
       cnt < config_setting_.candidate_num_;
       ++cnt)
  {
    double max_vote = -1.0;
    int max_vote_index = -1;
    bool selected_by_pending_rescue = false;

    for (int i = 0; i < 20000; ++i)
    {
      const bool in_pending_range =
          pending_rescue_enabled_ &&
          i >= pending_rescue_frame_min_ &&
          i <= pending_rescue_frame_max_;

      const double required_vote =
          in_pending_range
              ? 3.0
              : 4.0;

      if (match_array[i] >= required_vote &&
          match_array[i] > max_vote)
      {
        max_vote = match_array[i];
        max_vote_index = i;
        selected_by_pending_rescue =
            in_pending_range;
      }
    }

    if (max_vote_index < 0)
    {
      break;
    }

    BTCMatchList match_triangle_list;

    match_array[max_vote_index] = 0;

    match_triangle_list.match_frame_ =
        max_vote_index;

    match_triangle_list.match_id_.first =
        current_frame_id;

    match_triangle_list.match_id_.second =
        max_vote_index;

    for (size_t i = 0;
         i < index_recorder.size();
         ++i)
    {
      if (match_list_index[i] !=
          max_vote_index)
      {
        continue;
      }

      std::pair<BTC, BTC>
          single_match_pair;

      single_match_pair.first =
          current_STD_list[
              index_recorder[i][0]];

      single_match_pair.second =
          data_base_[
              useful_match_position[
                  index_recorder[i][0]]
                                   [
                  index_recorder[i][1]]]
                    [
              useful_match_index[
                  index_recorder[i][0]]
                                [
                  index_recorder[i][1]]];

      match_triangle_list
          .match_list_
          .push_back(
              single_match_pair);
    }

    if (selected_by_pending_rescue &&
        max_vote < 4.0)
    {
    }

    candidate_matcher_vec.push_back(
        match_triangle_list);
  }
}

void BtcDescManager::candidate_verify(
    const BTCMatchList &candidate_matcher, double &verify_score,
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
    std::vector<std::pair<BTC, BTC>> &sucess_match_list)
{
  sucess_match_list.clear();

  // -----------------------------------------------------------------------
  // FR-SLAM V28.8 Candidate Verify Consistency Diagnostic
  //
  // Diagnostic only:
  //   * official 3.0 m vertex threshold is unchanged
  //   * official max_vote >= 4 gate is unchanged
  //   * plane_geometric_verify() is unchanged
  //
  // The extra serial replay below answers:
  //   1) how many SE(3)-consistent triangle pairs each hypothesis gets
  //   2) whether the best rejected case is "almost 4 votes"
  //   3) how each triangle pair fails under the best SE(3)
  //   4) what would happen at 3.25 / 3.5 / 4.0 m, WITHOUT changing behavior
  // -----------------------------------------------------------------------

  double dis_threshold = 3;
  std::time_t solve_time = 0;
  std::time_t verify_time = 0;

  const int current_frame =
      candidate_matcher.match_list_.empty()
          ? candidate_matcher.match_id_.first
          : candidate_matcher.match_list_.front().first.frame_number_;

  const int historical_frame =
      candidate_matcher.match_id_.second;

  int skip_len =
      (int)(candidate_matcher.match_list_.size() / 50) + 1;

  int use_size =
      candidate_matcher.match_list_.size() / skip_len;

  std::vector<size_t> index(use_size);
  std::vector<int> vote_list(use_size);

  for (size_t i = 0; i < index.size(); i++)
  {
    index[i] = i;
  }

  if (print_debug_info_)
  {
  }

  std::mutex mylock;
  auto t0 = std::chrono::high_resolution_clock::now();

  // -----------------------------------------------------------------------
  // ORIGINAL official vote calculation -- unchanged.
  // -----------------------------------------------------------------------
  std::for_each(
      std::execution::par_unseq, index.begin(), index.end(),
      [&](const size_t &i)
      {
        auto single_pair =
            candidate_matcher.match_list_[i * skip_len];

        int vote = 0;
        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;

        triangle_solver(single_pair, test_t, test_rot);

        for (size_t j = 0;
             j < candidate_matcher.match_list_.size();
             j++)
        {
          auto verify_pair =
              candidate_matcher.match_list_[j];

          Eigen::Vector3d A =
              verify_pair.first.binary_A_.location_;
          Eigen::Vector3d A_transform =
              test_rot * A + test_t;

          Eigen::Vector3d B =
              verify_pair.first.binary_B_.location_;
          Eigen::Vector3d B_transform =
              test_rot * B + test_t;

          Eigen::Vector3d C =
              verify_pair.first.binary_C_.location_;
          Eigen::Vector3d C_transform =
              test_rot * C + test_t;

          double dis_A =
              (A_transform -
               verify_pair.second.binary_A_.location_)
                  .norm();

          double dis_B =
              (B_transform -
               verify_pair.second.binary_B_.location_)
                  .norm();

          double dis_C =
              (C_transform -
               verify_pair.second.binary_C_.location_)
                  .norm();

          if (dis_A < dis_threshold &&
              dis_B < dis_threshold &&
              dis_C < dis_threshold)
          {
            vote++;
          }
        }

        mylock.lock();
        vote_list[i] = vote;
        mylock.unlock();
      });

  int max_vote_index = 0;
  int max_vote = 0;

  for (size_t i = 0; i < vote_list.size(); i++)
  {
    if (max_vote < vote_list[i])
    {
      max_vote_index = i;
      max_vote = vote_list[i];
    }
  }

  // =====================================================================
  //
  // Keep Official BTC consensus voting as the primary criterion.
  //
  // Problem:
  //   With many highly repetitive / duplicate triangle correspondences,
  //   two hypotheses may have almost identical vote counts. A physically
  //   wrong pose can win by only one accidental vote.
  //
  // Solution:
  //   Only inside the top 99% vote plateau, use mean triangle residual
  //   as a tie-breaker.
  //
  // Examples:
  //   1410 vs 1409 -> considered near-tie
  //   100  vs 99   -> considered near-tie
  //   50   vs 49   -> NOT considered near-tie
  //   5    vs 4    -> NOT considered near-tie
  //
  // Therefore low-vote Official BTC behavior is essentially preserved.
  // =====================================================================

  const int raw_max_vote = max_vote;
  const int raw_max_vote_index = max_vote_index;
  const double near_tie_vote_ratio = 0.99;

  int near_tie_count = 0;
  double selected_residual_mean = 1e100;

  auto hypothesis_residual_mean =
      [&](const size_t hypothesis_index)
      {
        const size_t pair_index =
            hypothesis_index *
            static_cast<size_t>(skip_len);

        auto hypothesis_pair =
            candidate_matcher.match_list_[pair_index];

        Eigen::Matrix3d test_rot;
        Eigen::Vector3d test_t;

        triangle_solver(
            hypothesis_pair,
            test_t,
            test_rot);

        double residual_sum = 0.0;

        for (size_t j = 0;
             j < candidate_matcher.match_list_.size();
             ++j)
        {
          const auto &verify_pair =
              candidate_matcher.match_list_[j];

          const Eigen::Vector3d A_transform =
              test_rot *
                  verify_pair.first.binary_A_.location_ +
              test_t;

          const Eigen::Vector3d B_transform =
              test_rot *
                  verify_pair.first.binary_B_.location_ +
              test_t;

          const Eigen::Vector3d C_transform =
              test_rot *
                  verify_pair.first.binary_C_.location_ +
              test_t;

          const double dis_A =
              (A_transform -
               verify_pair.second.binary_A_.location_)
                  .norm();

          const double dis_B =
              (B_transform -
               verify_pair.second.binary_B_.location_)
                  .norm();

          const double dis_C =
              (C_transform -
               verify_pair.second.binary_C_.location_)
                  .norm();

          residual_sum +=
              std::max(
                  dis_A,
                  std::max(dis_B, dis_C));
        }

        if (candidate_matcher.match_list_.empty())
        {
          return 1e100;
        }

        return residual_sum /
               static_cast<double>(
                   candidate_matcher.match_list_.size());
      };

  if (!vote_list.empty() &&
      raw_max_vote > 0)
  {
    selected_residual_mean =
        hypothesis_residual_mean(
            static_cast<size_t>(raw_max_vote_index));

    for (size_t i = 0;
         i < vote_list.size();
         ++i)
    {
      const double vote_ratio =
          static_cast<double>(vote_list[i]) /
          static_cast<double>(raw_max_vote);

      if (vote_ratio < near_tie_vote_ratio)
      {
        continue;
      }

      ++near_tie_count;

      const double residual_mean =
          hypothesis_residual_mean(i);

      // Keep the higher-vote Official winner unless another hypothesis
      // inside the near-tie plateau has clearly smaller residual.
      if (residual_mean + 1e-9 <
          selected_residual_mean)
      {
        selected_residual_mean =
            residual_mean;

        max_vote_index =
            static_cast<int>(i);
      }
    }

    // From here onward, all existing verification code uses the selected
    // hypothesis and its own actual vote count.
    max_vote =
        vote_list[
            static_cast<size_t>(max_vote_index)];
  }

  if (print_debug_info_)
  {
  }

  // -----------------------------------------------------------------------
  // V28.8 SERIAL diagnostic replay.
  // No production state is modified here.
  // -----------------------------------------------------------------------
  constexpr bool kEnableBtcV288HeavyDiagnostics = false;
  if (kEnableBtcV288HeavyDiagnostics && print_debug_info_)
  {
    const double sweep_thresholds[4] =
        {3.0, 3.25, 3.5, 4.0};

    int sweep_best_votes[4] =
        {0, 0, 0, 0};

    int sweep_best_hypothesis[4] =
        {-1, -1, -1, -1};

    for (size_t diag_i = 0;
         diag_i < index.size();
         ++diag_i)
    {
      const size_t pair_index =
          diag_i * skip_len;

      auto hypothesis_pair =
          candidate_matcher.match_list_[pair_index];

      Eigen::Matrix3d diag_rot;
      Eigen::Vector3d diag_t;

      triangle_solver(
          hypothesis_pair,
          diag_t,
          diag_rot);

      int diag_votes[4] =
          {0, 0, 0, 0};

      double residual_sum = 0.0;
      double residual_max = 0.0;

      for (size_t diag_j = 0;
           diag_j < candidate_matcher.match_list_.size();
           ++diag_j)
      {
        const auto &verify_pair =
            candidate_matcher.match_list_[diag_j];

        const Eigen::Vector3d A_transform =
            diag_rot *
                verify_pair.first.binary_A_.location_ +
            diag_t;

        const Eigen::Vector3d B_transform =
            diag_rot *
                verify_pair.first.binary_B_.location_ +
            diag_t;

        const Eigen::Vector3d C_transform =
            diag_rot *
                verify_pair.first.binary_C_.location_ +
            diag_t;

        const double dis_A =
            (A_transform -
             verify_pair.second.binary_A_.location_)
                .norm();

        const double dis_B =
            (B_transform -
             verify_pair.second.binary_B_.location_)
                .norm();

        const double dis_C =
            (C_transform -
             verify_pair.second.binary_C_.location_)
                .norm();

        const double pair_max =
            std::max(dis_A,
                     std::max(dis_B, dis_C));

        residual_sum += pair_max;
        residual_max =
            std::max(residual_max, pair_max);

        for (int sweep_i = 0;
             sweep_i < 4;
             ++sweep_i)
        {
          const double threshold =
              sweep_thresholds[sweep_i];

          if (dis_A < threshold &&
              dis_B < threshold &&
              dis_C < threshold)
          {
            ++diag_votes[sweep_i];
          }
        }
      }

      for (int sweep_i = 0;
           sweep_i < 4;
           ++sweep_i)
      {
        if (diag_votes[sweep_i] >
            sweep_best_votes[sweep_i])
        {
          sweep_best_votes[sweep_i] =
              diag_votes[sweep_i];

          sweep_best_hypothesis[sweep_i] =
              static_cast<int>(diag_i);
        }
      }

      double rotation_cos =
          (diag_rot.trace() - 1.0) * 0.5;

      if (rotation_cos > 1.0)
      {
        rotation_cos = 1.0;
      }

      if (rotation_cos < -1.0)
      {
        rotation_cos = -1.0;
      }

      const double rotation_deg =
          std::acos(rotation_cos) *
          180.0 / 3.14159265358979323846;

      const double residual_mean =
          candidate_matcher.match_list_.empty()
              ? 0.0
              : residual_sum /
                    static_cast<double>(
                        candidate_matcher.match_list_.size());

    }



    // Even if official max_vote < 4, inspect every rough pair under
    // the best official 3.0 m hypothesis.
    if (!vote_list.empty() &&
        !candidate_matcher.match_list_.empty())
    {
      const size_t best_pair_index =
          static_cast<size_t>(max_vote_index) *
          static_cast<size_t>(skip_len);

      auto best_diag_pair =
          candidate_matcher.match_list_[best_pair_index];

      Eigen::Matrix3d best_diag_rot;
      Eigen::Vector3d best_diag_t;

      triangle_solver(
          best_diag_pair,
          best_diag_t,
          best_diag_rot);

      int pass_3p0 = 0;
      int pass_3p25 = 0;
      int pass_3p5 = 0;
      int pass_4p0 = 0;

      for (size_t diag_j = 0;
           diag_j < candidate_matcher.match_list_.size();
           ++diag_j)
      {
        const auto &verify_pair =
            candidate_matcher.match_list_[diag_j];

        const Eigen::Vector3d A_transform =
            best_diag_rot *
                verify_pair.first.binary_A_.location_ +
            best_diag_t;

        const Eigen::Vector3d B_transform =
            best_diag_rot *
                verify_pair.first.binary_B_.location_ +
            best_diag_t;

        const Eigen::Vector3d C_transform =
            best_diag_rot *
                verify_pair.first.binary_C_.location_ +
            best_diag_t;

        const double dis_A =
            (A_transform -
             verify_pair.second.binary_A_.location_)
                .norm();

        const double dis_B =
            (B_transform -
             verify_pair.second.binary_B_.location_)
                .norm();

        const double dis_C =
            (C_transform -
             verify_pair.second.binary_C_.location_)
                .norm();

        const double pair_max =
            std::max(dis_A,
                     std::max(dis_B, dis_C));

        const bool pair_pass_3p0 =
            dis_A < 3.0 &&
            dis_B < 3.0 &&
            dis_C < 3.0;

        const bool pair_pass_3p25 =
            dis_A < 3.25 &&
            dis_B < 3.25 &&
            dis_C < 3.25;

        const bool pair_pass_3p5 =
            dis_A < 3.5 &&
            dis_B < 3.5 &&
            dis_C < 3.5;

        const bool pair_pass_4p0 =
            dis_A < 4.0 &&
            dis_B < 4.0 &&
            dis_C < 4.0;

        pass_3p0 += pair_pass_3p0 ? 1 : 0;
        pass_3p25 += pair_pass_3p25 ? 1 : 0;
        pass_3p5 += pair_pass_3p5 ? 1 : 0;
        pass_4p0 += pair_pass_4p0 ? 1 : 0;

      }

    }
  }

  // -----------------------------------------------------------------------
  // ORIGINAL official gate -- unchanged.
  // -----------------------------------------------------------------------
  if (max_vote >= 4)
  {
    auto best_pair =
        candidate_matcher
            .match_list_[max_vote_index * skip_len];

    int vote = 0;
    Eigen::Matrix3d best_rot;
    Eigen::Vector3d best_t;

    triangle_solver(
        best_pair,
        best_t,
        best_rot);

    relative_pose.first = best_t;
    relative_pose.second = best_rot;

    for (size_t j = 0;
         j < candidate_matcher.match_list_.size();
         j++)
    {
      auto verify_pair =
          candidate_matcher.match_list_[j];

      Eigen::Vector3d A =
          verify_pair.first.binary_A_.location_;
      Eigen::Vector3d A_transform =
          best_rot * A + best_t;

      Eigen::Vector3d B =
          verify_pair.first.binary_B_.location_;
      Eigen::Vector3d B_transform =
          best_rot * B + best_t;

      Eigen::Vector3d C =
          verify_pair.first.binary_C_.location_;
      Eigen::Vector3d C_transform =
          best_rot * C + best_t;

      double dis_A =
          (A_transform -
           verify_pair.second.binary_A_.location_)
              .norm();

      double dis_B =
          (B_transform -
           verify_pair.second.binary_B_.location_)
              .norm();

      double dis_C =
          (C_transform -
           verify_pair.second.binary_C_.location_)
              .norm();

      if (dis_A < dis_threshold &&
          dis_B < dis_threshold &&
          dis_C < dis_threshold)
      {
        sucess_match_list.push_back(verify_pair);
      }
    }

    verify_score =
        plane_geometric_verify(
            plane_cloud_vec_.back(),
            plane_cloud_vec_[candidate_matcher.match_id_.second],
            relative_pose);

    if (print_debug_info_)
    {
    }
  }
  else
  {
    verify_score = -1;

    if (print_debug_info_)
    {
    }
  }

  return;
}

void BtcDescManager::triangle_solver(std::pair<BTC, BTC> &std_pair,
                                     Eigen::Vector3d &t, Eigen::Matrix3d &rot)
{
  Eigen::Matrix3d src = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d ref = Eigen::Matrix3d::Zero();
  src.col(0) = std_pair.first.binary_A_.location_ - std_pair.first.center_;
  src.col(1) = std_pair.first.binary_B_.location_ - std_pair.first.center_;
  src.col(2) = std_pair.first.binary_C_.location_ - std_pair.first.center_;
  ref.col(0) = std_pair.second.binary_A_.location_ - std_pair.second.center_;
  ref.col(1) = std_pair.second.binary_B_.location_ - std_pair.second.center_;
  ref.col(2) = std_pair.second.binary_C_.location_ - std_pair.second.center_;
  Eigen::Matrix3d covariance = src * ref.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      covariance, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::Matrix3d V = svd.matrixV();
  Eigen::Matrix3d U = svd.matrixU();
  rot = V * U.transpose();
  if (rot.determinant() < 0)
  {
    Eigen::Matrix3d K;
    K << 1, 0, 0, 0, 1, 0, 0, 0, -1;
    rot = V * K * U.transpose();
  }
  t = -rot * std_pair.first.center_ + std_pair.second.center_;
}

double BtcDescManager::plane_geometric_verify(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform)
{
  Eigen::Vector3d t = transform.first;
  Eigen::Matrix3d rot = transform.second;
  pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(
      new pcl::KdTreeFLANN<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  for (size_t i = 0; i < target_cloud->size(); i++)
  {
    pcl::PointXYZ pi;
    pi.x = target_cloud->points[i].x;
    pi.y = target_cloud->points[i].y;
    pi.z = target_cloud->points[i].z;
    input_cloud->push_back(pi);
  }

  kd_tree->setInputCloud(input_cloud);
  // 创建两个向量，分别存放近邻的索引值、近邻的中心距
  std::vector<int> pointIdxNKNSearch(1);
  std::vector<float> pointNKNSquaredDistance(1);
  double useful_match = 0;
  double normal_threshold = config_setting_.normal_threshold_;
  double dis_threshold = config_setting_.dis_threshold_;
  for (size_t i = 0; i < source_cloud->size(); i++)
  {
    pcl::PointXYZINormal searchPoint = source_cloud->points[i];
    pcl::PointXYZ use_search_point;
    use_search_point.x = searchPoint.x;
    use_search_point.y = searchPoint.y;
    use_search_point.z = searchPoint.z;
    Eigen::Vector3d pi(searchPoint.x, searchPoint.y, searchPoint.z);
    pi = rot * pi + t;
    use_search_point.x = pi[0];
    use_search_point.y = pi[1];
    use_search_point.z = pi[2];
    Eigen::Vector3d ni(searchPoint.normal_x, searchPoint.normal_y,
                       searchPoint.normal_z);
    ni = rot * ni;
    if (kd_tree->nearestKSearch(use_search_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0)
    {
      pcl::PointXYZINormal nearstPoint =
          target_cloud->points[pointIdxNKNSearch[0]];
      Eigen::Vector3d tpi(nearstPoint.x, nearstPoint.y, nearstPoint.z);
      Eigen::Vector3d tni(nearstPoint.normal_x, nearstPoint.normal_y,
                          nearstPoint.normal_z);
      Eigen::Vector3d normal_inc = ni - tni;
      Eigen::Vector3d normal_add = ni + tni;
      double point_to_plane = fabs(tni.transpose() * (pi - tpi));
      if ((normal_inc.norm() < normal_threshold ||
           normal_add.norm() < normal_threshold) &&
          point_to_plane < dis_threshold)
      {
        useful_match++;
      }
    }
  }
  return useful_match / source_cloud->size();
}
