// Define the intermediate and final values reported by an IK solve.
// Optional values remain empty when the corresponding stage was not computed.

#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace x7 {
struct IKDiagnostics {
  std::string status = "not_solved", branch = "none", hold_reason;
  std::optional<std::vector<double>> q_raw, q_processed;
  std::vector<double> q_final;
  std::array<bool, 7> step_limited{}, joint_limited{}, held{};
  bool zero_seed_retry = false;
  bool singular_continuity_enabled = true, singular_continuity_applied = false;
  double continuity_weight = 0.0;
  std::optional<double> singular_s, r22;
  int selected_wrist_branch = -1;
  int64_t started_ns = 0, ended_ns = 0;
};
}
