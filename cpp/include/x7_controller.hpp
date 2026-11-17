// Declare the robot controller interface and its cached model state.

#pragma once
#include <Eigen/Dense>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "hardware.hpp"
#include "kinematics_utils.hpp"
#include "ik_diagnostics.hpp"

namespace x7 {
class Controller {
public:
  bool connect(const std::string& port, int baudrate,
               const std::string& hw_cfg, const std::string& link_cfg,
               bool record_feedback = false);
  std::vector<double> solve_ik(const Eigen::Matrix4d& T_ee);
  std::vector<double> solve_ik(const Eigen::Matrix4d& T_ee, IKDiagnostics* diagnostics);
  rt_manipulators_cpp::PositionFeedback feedback_snapshot(const std::string& group) const;
  std::vector<double> model_position_limits(bool maximum) const;
  std::vector<double> gripper_position_limits() const;
  bool set_positions(const std::vector<double>& q);

  bool write_gripper(int id, double angle_deg);
  const std::map<std::string, bool>& initialization_status() const { return initialization_status_; }

private:
  friend struct OfflineControllerAccess;
  std::map<std::string, bool> initialization_status_;
  std::string port_;
  std::unique_ptr<rt_manipulators_cpp::Hardware> hardware_;
  kinematics_utils::links_t links_;
  std::vector<double> cached_q_;
};

}
