// Connect the hardware model and compute joint targets from end-effector poses.
// Resolve wrist orientation continuously and apply joint and per-step limits.

#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <time.h>
#include <Eigen/Dense>

#include "hardware.hpp"
#include "kinematics.hpp"
#include "kinematics_utils.hpp"
#include "link.hpp"
#include "rt_manipulators_ik.hpp"
#include "x7_controller.hpp"

namespace x7 {

bool Controller::connect(const std::string& port,
                         int baudrate,
                         const std::string& hw_cfg,
                         const std::string& link_cfg, bool record_feedback) {
  port_ = port;

  links_ = kinematics_utils::parse_link_config_file(link_cfg);
  std::cout << "[DBG] links_.size() = " << links_.size() << std::endl;

  hardware_ = std::make_unique<rt_manipulators_cpp::Hardware>(port_);
  if (record_feedback) hardware_->enable_recording_feedback();
  if (!hardware_->connect(baudrate)) return false;
  if (!hardware_->load_config_file(hw_cfg)) return false;

  for (auto link_id : kinematics_utils::find_route(links_, 8)) {
    if (links_[link_id].dxl_id > 0) {
      hardware_->get_max_position_limit(links_[link_id].dxl_id, links_[link_id].max_q);
      hardware_->get_min_position_limit(links_[link_id].dxl_id, links_[link_id].min_q);
    }
  }

  const bool arm_profile_acceleration_ok = hardware_->write_max_acceleration_to_group("arm", M_PI);
  if (record_feedback) initialization_status_["arm_profile_acceleration"] = arm_profile_acceleration_ok;
  const bool arm_profile_velocity_ok = hardware_->write_max_velocity_to_group("arm", M_PI);
  if (record_feedback) initialization_status_["arm_profile_velocity"] = arm_profile_velocity_ok;
  const bool arm_position_pid_ok = hardware_->write_position_pid_gain_to_group("arm", 800, 0, 0);
  if (record_feedback) initialization_status_["arm_position_pid"] = arm_position_pid_ok;
  const bool arm_torque_on_ok = hardware_->torque_on("arm");
  if (record_feedback) initialization_status_["arm_torque_on"] = arm_torque_on_ok;
  const bool hand_profile_acceleration_ok = hardware_->write_max_acceleration_to_group("hand", M_PI);
  if (record_feedback) initialization_status_["hand_profile_acceleration"] = hand_profile_acceleration_ok;
  const bool hand_profile_velocity_ok = hardware_->write_max_velocity_to_group("hand", M_PI);
  if (record_feedback) initialization_status_["hand_profile_velocity"] = hand_profile_velocity_ok;
  const bool hand_position_pid_ok = hardware_->write_position_pid_gain_to_group("hand", 800, 0, 0);
  if (record_feedback) initialization_status_["hand_position_pid"] = hand_position_pid_ok;
  const bool hand_torque_on_ok = hardware_->torque_on("hand");
  if (record_feedback) initialization_status_["hand_torque_on"] = hand_torque_on_ok;

  const bool thread_started = hardware_->start_thread({"arm", "hand"}, std::chrono::milliseconds(10));
  if (record_feedback) initialization_status_["communication_thread_started"] = thread_started;
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  std::vector<double> q_cur;
  if (!hardware_->get_positions("arm", q_cur) || q_cur.size() != 7) {
    std::cerr << "[ERR] failed to read arm positions\n";
    return false;
  }

  for (int j = 0; j < 7; ++j) {
    int link_idx = 2 + j;
    if (link_idx < static_cast<int>(links_.size())) {
      links_[link_idx].q = q_cur[j];
    }
  }
  cached_q_ = q_cur;

  kinematics::forward_kinematics(links_, 1);
  return true;
}

std::vector<double> Controller::solve_ik(const Eigen::Matrix4d& T_ee) {
  return solve_ik(T_ee, nullptr);
}

std::vector<double> Controller::solve_ik(const Eigen::Matrix4d& T_ee, IKDiagnostics* diag) {
  auto monotonic_ns = []() -> int64_t {
    timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts);
    return int64_t(ts.tv_sec) * 1000000000LL + ts.tv_nsec;
  };
  if (diag) { *diag = IKDiagnostics{}; diag->started_ns = monotonic_ns(); }
  struct Finish {
    IKDiagnostics* diag;
    decltype(monotonic_ns)& clock;
    ~Finish() { if (diag) diag->ended_ns = clock(); }
  } finish{diag, monotonic_ns};

  const Eigen::Vector3d target_p = T_ee.block<3, 1>(0, 3);
  const Eigen::Matrix3d target_R = T_ee.block<3, 3>(0, 0);

  if (cached_q_.size() == 7) {
    for (int j = 0; j < 7; ++j) links_[2 + j].q = cached_q_[j];
  }
  kinematics::forward_kinematics(links_, 1);

  kinematics_utils::q_list_t q_list;
  bool ok = samples03::x7_3dof_inverse_kinematics(links_, target_p, q_list);

  if (!ok) {
    if (diag) diag->zero_seed_retry = true;
    std::cerr << "[WARN] Position IK failed with cached seed, retrying from zero." << std::endl;
    for (int j = 0; j < 7; ++j) links_[2 + j].q = 0.0;
    kinematics::forward_kinematics(links_, 1);
    q_list.clear();
    ok = samples03::x7_3dof_inverse_kinematics(links_, target_p, q_list);
  }

  if (!ok) {
    std::cerr << "[WARN] Analytical position IK failed. Holding current position." << std::endl;
    if (diag) {
      diag->status = "position_ik_failed";
      diag->hold_reason = "position_ik_failed";
      diag->held.fill(cached_q_.size() == 7);
      diag->q_final = cached_q_;
    }
    if (cached_q_.size() == 7) return cached_q_;
    throw std::runtime_error("IK failed and no cached position available.");
  }

  q_list[4] = 0.0;

  kinematics_utils::set_q_list(links_, q_list);
  links_[6].q = 0.0;
  links_[7].q = 0.0;
  links_[8].q = 0.0;
  kinematics::forward_kinematics(links_, 1);

  const Eigen::Matrix3d wrist_R = links_[8].R;
  const Eigen::Matrix3d diff_R  = wrist_R.transpose() * target_R;

  const double R00 = diff_R(0, 0);
  const double R10 = diff_R(1, 0);
  const double R02 = diff_R(0, 2);
  const double R12 = diff_R(1, 2);
  const double R20 = diff_R(2, 0);
  const double R21 = diff_R(2, 1);
  const double R22 = diff_R(2, 2);

  auto wrap_to_pi = [](double x) {
  while (x >  M_PI) x -= 2.0 * M_PI;
  while (x < -M_PI) x += 2.0 * M_PI;
  return x;
  };

  double theta_j5 = 0.0, theta_j6 = 0.0, theta_j7 = 0.0;

  const double s = std::sqrt(R02*R02 + R12*R12);
  const double eps = 1e-8;
  if (diag) { diag->singular_s = s; diag->r22 = R22; diag->status = "ok"; }

  struct Cand { double t5, t6, t7; };

  auto in_range = [&](const Cand& c) {
    kinematics_utils::q_list_t wrist_q;
    wrist_q[6] = c.t5;
    wrist_q[7] = c.t6;
    wrist_q[8] = c.t7;
    return samples03::q_list_are_in_range(links_, wrist_q);
  };

  auto dist_to_cached = [&](const Cand& c) {
    if (cached_q_.size() != 7) return 0.0;

    const double d5 = c.t5 - cached_q_[4];
    const double d6 = c.t6 - cached_q_[5];
    const double d7 = c.t7 - cached_q_[6];
    return d5*d5 + d6*d6 + d7*d7;
  };

  auto closest_singular_pair = [&](double phi, Cand& result) {
    if (cached_q_.size() != 7) return false;

    const double prev5 = cached_q_[4];
    const double prev7 = cached_q_[6];
    const double min5 = links_[6].min_q;
    const double max5 = links_[6].max_q;
    const double min7 = links_[8].min_q;
    const double max7 = links_[8].max_q;

    bool found = false;
    double best_cost = 1e100;
    for (int k = -2; k <= 2; ++k) {
      const double desired_sum = phi + 2.0 * M_PI * static_cast<double>(k);
      const double feasible_low = std::max(min5, desired_sum - max7);
      const double feasible_high = std::min(max5, desired_sum - min7);
      if (feasible_low > feasible_high) continue;

      const double unconstrained =
        0.5 * (desired_sum + prev5 - prev7);
      const double t5 = std::max(feasible_low,
        std::min(feasible_high, unconstrained));
      const double t7 = desired_sum - t5;
      const double cost =
        (t5-prev5)*(t5-prev5) + (t7-prev7)*(t7-prev7);
      if (cost < best_cost) {
        best_cost = cost;
        result = {t5, 0.0, t7};
        found = true;
      }
    }
    return found;
  };

  if (s < eps) {
    if (diag) { diag->status = "exact_singularity"; diag->branch = "singular"; }
    const double b = (R22 >= 0.0) ? 0.0 : M_PI;
    const double phi = std::atan2(R10, R00);
    Cand singular{};
    if (b == 0.0 && closest_singular_pair(phi, singular)) {
      if (diag) { diag->singular_continuity_applied = true; diag->continuity_weight = 1.0; }
      theta_j5 = singular.t5;
      theta_j6 = 0.0;
      theta_j7 = singular.t7;
    } else if (cached_q_.size() == 7) {
      if (diag) {
        diag->status = "wrist_held";
        diag->hold_reason = "singular_pair_unavailable_or_pi_out_of_range";
        diag->held[4] = diag->held[5] = diag->held[6] = true;
      }
      theta_j5 = cached_q_[4];
      theta_j6 = cached_q_[5];
      theta_j7 = cached_q_[6];
    } else {
      theta_j5 = phi;
      theta_j6 = -b;
      theta_j7 = 0.0;
    }
  } else {
    const double b1 = std::atan2(s, R22);
    const double a1 = std::atan2(R12, R02);
    const double c1 = std::atan2(R21, -R20);

    const double b2 = std::atan2(-s, R22);
    const double a2 = wrap_to_pi(a1 + M_PI);
    const double c2 = wrap_to_pi(c1 + M_PI);

    Cand cand[2] = {
      { a1, -b1, c1 },
      { a2, -b2, c2 }
    };

    int best = -1;
    double best_cost = 1e100;
    for (int i=0;i<2;i++) {
      if (!in_range(cand[i])) continue;
      const double cost = dist_to_cached(cand[i]);
      if (cost < best_cost) { best_cost = cost; best = i; }
    }
    if (best < 0) {
      if (diag) {
        diag->status = "wrist_held";
        diag->branch = "no_feasible_wrist_branch";
        diag->hold_reason = cached_q_.size() == 7 ? "wrist_branches_out_of_range" : "wrist_zero_fallback_no_cache";
        diag->held[4] = diag->held[5] = diag->held[6] = cached_q_.size() == 7;
      }
      if (cached_q_.size() == 7) {
        theta_j5 = cached_q_[4];
        theta_j6 = cached_q_[5];
        theta_j7 = cached_q_[6];
      } else {
        theta_j5 = 0.0;
        theta_j6 = 0.0;
        theta_j7 = 0.0;
      }
    } else {
      Cand selected = cand[best];
      if (diag) {
        diag->selected_wrist_branch = best;
        diag->branch = best == 0 ? "euler_primary" : "euler_alternate";
        diag->q_raw = std::vector<double>{q_list.at(2), q_list.at(3), 0.0,
          q_list.at(5), selected.t5, selected.t6, selected.t7};
      }

      constexpr double singular_s_threshold = 0.17;
      if (R22 > 0.0 && s < singular_s_threshold && cached_q_.size() == 7) {
        const double phi = std::atan2(R10, R00);
        Cand regularized{};
        if (closest_singular_pair(phi, regularized)) {
          double blend = s / singular_s_threshold;
          blend = blend * blend * (3.0 - 2.0 * blend);
          if (diag) { diag->singular_continuity_applied = true; diag->continuity_weight = 1.0 - blend; }
          selected.t5 = regularized.t5 + blend * (selected.t5 - regularized.t5);
          selected.t7 = regularized.t7 + blend * (selected.t7 - regularized.t7);
        }
      }

      theta_j5 = selected.t5;
      theta_j6 = selected.t6;
      theta_j7 = selected.t7;
    }
  }

  std::vector<double> q(7, 0.0);
  q[0] = q_list.count(2) ? q_list.at(2) : 0.0;
  q[1] = q_list.count(3) ? q_list.at(3) : 0.0;
  q[2] = 0.0;
  q[3] = q_list.count(5) ? q_list.at(5) : 0.0;
  q[4] = theta_j5;
  q[5] = theta_j6;
  q[6] = theta_j7;
  if (diag) diag->q_processed = q;

  if (cached_q_.size() == 7) {
    constexpr double max_position_step = 0.08;
    constexpr double max_wrist_step = 0.06;
    for (int j = 0; j < 7; ++j) {
      const double max_step = (j >= 4) ? max_wrist_step : max_position_step;
      const double delta = q[j] - cached_q_[j];
      if (diag) diag->step_limited[j] = delta > max_step || delta < -max_step;
      if (delta > max_step) q[j] = cached_q_[j] + max_step;
      if (delta < -max_step) q[j] = cached_q_[j] - max_step;
      const int link_id = 2 + j;
      if (diag) diag->joint_limited[j] = q[j] < links_[link_id].min_q || q[j] > links_[link_id].max_q;
      q[j] = std::max(links_[link_id].min_q,
        std::min(links_[link_id].max_q, q[j]));
    }
  }

  cached_q_ = q;
  for (int j = 0; j < 7; ++j) links_[2 + j].q = q[j];

  if (diag) diag->q_final = q;
  return q;
}

bool Controller::set_positions(const std::vector<double>& q) {
  if (!hardware_ || q.size() != 7) return false;
  std::vector<double> q_mutable = q;
  if (!hardware_->set_positions("arm", q_mutable)) return false;
  cached_q_ = q;
  for (int j = 0; j < 7; ++j) links_[2 + j].q = q[j];
  return true;
}

bool Controller::write_gripper(int id, double angle_deg) {
  if (!hardware_) return false;
  return hardware_->set_position(id, angle_deg * M_PI / 180.0);
}

rt_manipulators_cpp::PositionFeedback Controller::feedback_snapshot(const std::string& group) const {
  return hardware_ ? hardware_->recording_position_snapshot(group)
                   : rt_manipulators_cpp::PositionFeedback{};
}
std::vector<double> Controller::gripper_position_limits() const {
  double minimum, maximum;
  if (!hardware_ || !hardware_->get_min_position_limit(9,minimum) ||
      !hardware_->get_max_position_limit(9,maximum)) return {};
  return {minimum,maximum};
}
std::vector<double> Controller::model_position_limits(bool maximum) const {
  std::vector<double> result;
  for (int j = 2; j <= 8 && j < static_cast<int>(links_.size()); ++j)
    result.push_back(maximum ? links_[j].max_q : links_[j].min_q);
  return result;
}
}
