// Declare the RT-derived position IK solver and joint-range checks.
// See cpp/NOTICE and the repository NOTICE for attribution.

#ifndef SAMPLES_SAMPLES03_INCLUDE_RT_MANIPULATORS_IK_HPP_
#define SAMPLES_SAMPLES03_INCLUDE_RT_MANIPULATORS_IK_HPP_

#include "kinematics_utils.hpp"

namespace samples03 {
bool q_list_are_in_range(const kinematics_utils::links_t & links, const kinematics_utils::q_list_t & q_list);

bool x7_3dof_inverse_kinematics(const kinematics_utils::links_t & links,
                                const Eigen::Vector3d & target_p,
                                kinematics_utils::q_list_t & q_list);

}

#endif
