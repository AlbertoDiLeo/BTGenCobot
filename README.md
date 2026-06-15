# BTGenCobot

Natural language to BehaviorTree XML generation for ROS2 mobile manipulators.

## Overview

BTGenCobot takes commands like "pick up the red cup and place it on the table" and generates valid BehaviorTree.CPP XML for execution on Nav2. Uses a fine-tuned Llama 3.2-1B model with grammar-constrained decoding to ensure syntactically correct output.

## Requirements

- Docker & Docker Compose
- NVIDIA GPU (optional, for faster inference)

## Usage

```bash
docker-compose up --build

# In the container
colcon build --symlink-install
source install/setup.bash

# Start simulation
ros2 launch turtlebot3_manipulation_description gazebo.launch.py

# Start Nav2
ros2 launch bt_bringup nav2_bringup.launch.py

# Start inference server (on host or in container)
cd inference_server && uv run serve

# Send commands
ros2 topic pub /btgen_nl_command std_msgs/String "data: 'pick up the red cup'"
```

## Saved-Map Navigation Runtime

For the frontend thesis workflow, room-based navigation should use a saved
metric map, not online SLAM as the operational assumption.

Preliminary mapping phase:

```bash
ros2 launch bt_bringup robot_bt_mapping.launch.py \
  world:=aws_small_house \
  use_rviz:=true \
  headless:=false

ros2 run nav2_map_server map_saver_cli -f /workspace/maps/aws_small_house
```

`robot_bt_mapping.launch.py` deliberately excludes Nav2 task execution,
inference, Florence-2 and manipulation services. This keeps scan acquisition
and SLAM more responsive during the dedicated mapping session. The historical
`robot_bt_frontend.launch.py` remains available for compatibility, but it is
not the preferred map-construction runtime.

### Laser validation before mapping

The simulated lidar publishes `/scan` in `base_scan` with a declared range of
`0.12-10.0 m`. SLAM Toolbox intentionally uses observations up to `5.0 m`.
The Gazebo GPU lidar is configured with an angular interval of `-pi..+pi`.
This is required for a real 360-degree scan in Gazebo Harmonic; the previous
`0..2*pi` interval was capped by the renderer. The validated message contains
360 samples spanning approximately `6.283 rad`.
Before producing a final saved map, validate the sensor in RViz:

1. launch the preliminary mapping runtime with RViz;
2. keep the RViz fixed frame on `map` and enable the `/scan` LaserScan display;
3. place the robot near walls at known distances and use the RViz Measure tool;
4. compare measured wall distances with the visible scan endpoints;
5. verify that scans remain aligned with walls while the robot moves and turns;
6. only then explore the complete environment and save the final map.

Useful runtime checks:

```bash
ros2 topic info /scan --verbose
ros2 topic echo /scan sensor_msgs/msg/LaserScan --once --field range_min
ros2 topic echo /scan sensor_msgs/msg/LaserScan --once --field range_max
ros2 topic hz /scan
ros2 run tf2_ros tf2_echo base_footprint base_scan
```

Do not increase the SLAM range solely to fill unknown areas. Missing walls must
first be investigated as an exploration, occlusion, TF, or scale problem.

The current saved AWS Small House map was produced with the dedicated mapping
launch and repeated observations from the living room, kitchen, western
passage and bedroom. It contains `375 x 224` cells at `0.05 m/cell`, with
origin `[-9.736, -5.601, 0]`.

Operational phase:

```bash
ros2 launch bt_bringup robot_bt_localization.launch.py \
  world:=aws_small_house \
  map_file:=/workspace/maps/aws_small_house.yaml
```

In the operational phase `map_server` publishes `/map`, AMCL localizes the
robot on that saved map, and Nav2 receives goals in the `map` frame. The
frontend semantic map and topology map should be aligned to this saved metric
map.

`robot_bt_localization.launch.py` starts Nav2 only after map server and AMCL
have established the `map -> odom -> base_footprint` transform chain. The
configured Living room, Kitchen and Bedroom topology goals have been checked
against the occupancy grid and fall in free cells. After replacing a saved
map, repeat the Nav2 room-navigation scenarios as the final acceptance check.

The Docker compose setup mounts `BTGenCobot/maps/` as `/workspace/maps/`, so
saved maps persist across container recreation.

## Architecture

```
Natural Language Command
         │
         ▼
┌─────────────────────────┐
│   Inference Server      │
│   (FastAPI + Llama 3.2) │
│   + EBNF Grammar CFG    │
└───────────┬─────────────┘
            │
            ▼
      BT XML Output
            │
            ▼
┌─────────────────────────┐
│  bt_text_interface      │
│  (ROS2 Action Server)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Nav2 bt_navigator      │
│  + Custom BT Plugins    │
└───────────┬─────────────┘
            │
            ▼
      Robot Execution
```

## Project Structure

```
BTGenCobot/
├── inference_server/           # LLM inference (FastAPI)
│   ├── api/                    # REST endpoints
│   ├── core/                   # Model loading, generation
│   ├── validation/             # XML validation, post-processing
│   └── prompts/                # System prompts
│
├── src/
│   ├── bt_text_interface/      # ROS2 action server for BT generation
│   ├── bt_nav2_plugins/        # Custom Nav2 BT nodes (DetectObject, Pick, Place, Spin)
│   ├── bt_bringup/             # Launch files and Nav2 config
│   ├── manipulator_control/    # Arm IK and control service
│   ├── vision_services/        # Florence-2 object detection
│   └── btgencobot_interfaces/  # Custom ROS2 messages/services
│
└── robot_description/          # TurtleBot3 + OpenManipulator-X URDF
```

## Custom BT Nodes

| Node | Description |
|------|-------------|
| `DetectObject` | Open-vocabulary object detection via Florence-2 |
| `PickObject` | Approach and grasp detected object |
| `PlaceObject` | Place held object at detected location |
| `SpinLeft/SpinRight` | Rotate in place |

## License

MIT
