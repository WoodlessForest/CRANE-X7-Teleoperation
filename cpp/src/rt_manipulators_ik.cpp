// Implement the RT-derived analytical position IK for the CRANE-X7.
// Check reachability and joint ranges before accepting the computed angles.

#include <iostream>
#include "rt_manipulators_ik.hpp"
#include <cmath>

namespace samples03 {

bool q_list_are_in_range(
  const kinematics_utils::links_t & links, const kinematics_utils::q_list_t & q_list) {
  for (const auto & [target_id, q_value] : q_list) {
    if (std::isnan(q_value)) {
      std::cerr << "リンクID:" << target_id << " の目標角度が異常値です"  << std::endl;
      return false;
    }

    if (q_value < links[target_id].min_q || q_value > links[target_id].max_q) {
      std::cerr << "リンクID:" << target_id << " の目標角度:" << q_value;
      std::cerr << " が可動範囲外です" << std::endl;
      return false;
    }
  }
  return true;
}

bool x7_3dof_inverse_kinematics(const kinematics_utils::links_t & links,
                                const Eigen::Vector3d & target_p,
                                kinematics_utils::q_list_t & q_list) {

  const double L1 = links[2].b.z() + links[3].b.z();
  const double L2 = links[4].b.z() + links[5].b.z();
  const double L3 = links[6].b.z() + links[7].b.z();
  const double L2_2 = std::pow(L2, 2);
  const double L3_2 = std::pow(L3, 2);


  const double PX = target_p.x();
  const double PY = target_p.y();
  const double PZ = target_p.z() - L1;
  const double PX_2 = std::pow(PX, 2);
  const double PY_2 = std::pow(PY, 2);
  const double PZ_2 = std::pow(PZ, 2);
  const double PXYZ_2 = PX_2 + PY_2 + PZ_2;

  if (std::pow(L2 - L3, 2) > PXYZ_2 || std::pow(L2 + L3, 2) < PXYZ_2) {
    std::cerr << "目標位置が可動範囲外です" << std::endl;
    return false;
  }
  double theta3 = std::acos((PXYZ_2 - L2_2 - L3_2) / (2 * L2 * L3));
  double theta1 = std::atan2(PY, PX);

  const double C3 = std::cos(theta3);
  const double S3 = std::sin(theta3);
  const double L2_L3C3 = L2 + L3 * C3;
  const double L3S3 =  L3 * S3;
  const double SQRT_PXY = std::sqrt(PX_2 + PY_2);

  double theta2 = std::atan2(
    +L2_L3C3 * SQRT_PXY - L3S3 * PZ,
    +L3S3 * SQRT_PXY + L2_L3C3 * PZ);

  theta2 *= -1.0;
  theta3 *= -1.0;

  q_list[2] = theta1;
  q_list[3] = theta2;
  q_list[5] = theta3;

  if (q_list_are_in_range(links, q_list) == false) {
    return false;
  }

  return true;
}

}
