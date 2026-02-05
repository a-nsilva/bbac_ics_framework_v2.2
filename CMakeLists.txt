cmake_minimum_required(VERSION 3.8)
project(bbac_framework)

# Compiler settings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AccessRequest.msg"
  "msg/AccessDecision.msg"
  "msg/LayerDecisionDetail.msg"
  "msg/EmergencyAlert.msg"
  "srv/ReconfigureLayers.srv"
  DEPENDENCIES std_msgs
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  src/ros/bbac_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install experiment scripts
install(PROGRAMS
  experiments/ablation_study.py
  experiments/baseline_comparison.py
  experiments/adaptive_eval.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install models
install(PROGRAMS
  scripts/train_models.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install utility scripts
install(PROGRAMS
  util/runner.py
  DESTINATION lib/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_ingestion tests/test_ingestion.py)
  ament_add_pytest_test(test_modeling tests/test_modeling.py)
  ament_add_pytest_test(test_decision tests/test_decision.py)
endif()

ament_package()
