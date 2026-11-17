// Expose the X7 controller, feedback, and IK diagnostics to Python through pybind11.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "x7_controller.hpp"
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>
#include "config_file_parser.hpp"

namespace py = pybind11;
using x7::Controller;

PYBIND11_MODULE(py_x7, m) {
  m.def("inspect_hardware_config", [](const std::string& path) {
    hardware_joints::Joints joints;
    if (!config_file_parser::parse(path, joints))
      throw std::runtime_error("Invalid robot configuration: " + path);
    auto yaml = YAML::LoadFile(path);
    py::dict result, groups;
    for (const auto& [name, group] : joints.groups()) {
      py::dict definition;
      definition["joints"] = group->joint_names();
      std::vector<std::string> reads, writes;
      if (group->sync_read_position_enabled()) reads.push_back("position");
      if (group->sync_read_velocity_enabled()) reads.push_back("velocity");
      if (group->sync_read_current_enabled()) reads.push_back("current");
      if (group->sync_read_voltage_enabled()) reads.push_back("voltage");
      if (group->sync_read_temperature_enabled()) reads.push_back("temperature");
      if (group->sync_write_position_enabled()) writes.push_back("position");
      if (group->sync_write_velocity_enabled()) writes.push_back("velocity");
      if (group->sync_write_current_enabled()) writes.push_back("current");
      definition["sync_read"] = reads; definition["sync_write"] = writes;
      groups[py::str(name)] = definition;
      for (const auto& joint_name : group->joint_names()) {
        const auto joint = joints.joint(joint_name);
        py::dict config;
        config["id"] = int(joint->id()); config["operating_mode"] = int(joint->operating_mode());
        config["dynamixel"] = yaml[joint_name]["dynamixel"] ? yaml[joint_name]["dynamixel"].as<std::string>() : "default";
        result[py::str(joint_name)] = config;
      }
    }
    result["joint_groups"] = groups;
    return result;
  });
  py::class_<Controller>(m, "Controller")
    .def(py::init<>())
    .def("connect",
         [](Controller& self, const std::string& port,
            int baudrate, const std::string& hw, const std::string& links, bool record_feedback){
           return self.connect(port, baudrate, hw, links, record_feedback);
         },
         py::arg("port"),
         py::arg("baudrate") = 3000000,
         py::arg("hw_cfg") = "../config/crane-x7.yaml",
         py::arg("links_cfg") = "../config/crane-x7_links.csv",
         py::arg("record_feedback") = false)
    .def("solve_ik",
         [](Controller& self,
            py::array_t<double,
                       py::array::c_style | py::array::forcecast> T_ee) {
           if (T_ee.ndim() != 2 || T_ee.shape(0) != 4 || T_ee.shape(1) != 4) {
             throw std::runtime_error("T_ee must be a 4x4 matrix");
           }
           auto buf = T_ee.unchecked<2>();
           Eigen::Matrix4d T;
           for (ssize_t i = 0; i < 4; ++i) {
             for (ssize_t j = 0; j < 4; ++j) {
               T(i, j) = buf(i, j);
             }
           }
           return self.solve_ik(T);
         },
         py::arg("T_ee"))
    .def("solve_ik_with_diagnostics", [](Controller& self,
          py::array_t<double, py::array::c_style | py::array::forcecast> input) {
      if (input.ndim() != 2 || input.shape(0) != 4 || input.shape(1) != 4)
        throw std::runtime_error("T_ee must be a 4x4 matrix");
      Eigen::Matrix4d T;
      auto a = input.unchecked<2>();
      for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) T(i,j) = a(i,j);
      x7::IKDiagnostics d;
      auto q = self.solve_ik(T, &d);
      py::dict out;
      out["status"] = d.status; out["branch"] = d.branch;
      out["q_ik_raw"] = py::cast(d.q_raw); out["q_ik_processed"] = py::cast(d.q_processed);
      out["q_final"] = d.q_final; out["step_limited"] = d.step_limited;
      out["joint_limited"] = d.joint_limited; out["held"] = d.held;
      out["hold_reason"] = d.hold_reason; out["zero_seed_retry"] = d.zero_seed_retry;
      out["singular_continuity_enabled"] = d.singular_continuity_enabled;
      out["singular_continuity_applied"] = d.singular_continuity_applied;
      out["continuity_weight"] = d.continuity_weight;
      out["singular_s"] = py::cast(d.singular_s); out["r22"] = py::cast(d.r22);
      out["selected_wrist_branch"] = d.selected_wrist_branch;
      out["started_ns"] = d.started_ns; out["ended_ns"] = d.ended_ns;
      return py::make_tuple(q, out);
    })
    .def("feedback_snapshot", [](const Controller& self, const std::string& group) {
      auto f = self.feedback_snapshot(group);
      py::dict out;
      out["enabled"] = f.enabled; out["updated"] = f.updated;
      out["last_read_ok"] = f.last_read_ok;
      out["update_ns"] = f.update_ns; out["attempt_ns"] = f.attempt_ns;
      out["sequence"] = f.sequence; out["positions"] = f.positions;
      return out;
    })
    .def("gripper_position_limits", &Controller::gripper_position_limits)
    .def("initialization_status", &Controller::initialization_status)
    .def("model_position_limits", &Controller::model_position_limits)
    .def("set_positions", &Controller::set_positions, py::arg("q"))
    .def("write_gripper", &Controller::write_gripper,
         py::arg("id") = 9, py::arg("angle_deg") = 0.0);
}
