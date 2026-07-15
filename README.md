# BTGenCobot

Natural-language to BehaviorTree.CPP XML generation and execution for ROS 2
mobile manipulators.

The project integrates:

- a FastAPI inference server for Behavior Tree generation;
- ROS 2 nodes for Behavior Tree validation and execution;
- Nav2 navigation on saved metric maps;
- Gazebo simulation with TurtleBot3 Waffle Pi and OpenManipulator-X;
- environment profiles for AWS Small House and AWS Hospital.

## Requirements

- Docker and Docker Compose
- NVIDIA GPU optional, but recommended for faster model inference
- AWS world assets available under `worlds/` when running the simulations

The Docker compose setup mounts the local project folders into the container:

- `src/` -> `/workspace/src`
- `robot_description/` -> `/workspace/robot_description`
- `worlds/` -> `/workspace/worlds`
- `maps/` -> `/workspace/maps`

Saved maps and world files are therefore loaded from the local repository copy.

## Quick Start

Start the container:

```bash
cd BTGenCobot
docker compose up -d --build
```

Open a shell inside the container:

```bash
docker exec -it btgencobot bash
```

Build the ROS 2 workspace:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Start the inference server in a separate terminal, either on the host or in the
container:

```bash
cd inference_server
uv run serve
```

Start the full saved-map runtime for AWS Small House:

```bash
ros2 launch bt_bringup robot_bt_localization.launch.py \
  environment:=aws_small_house
```

Start the full saved-map runtime for AWS Hospital:

```bash
ros2 launch bt_bringup robot_bt_localization.launch.py \
  environment:=aws_hospital
```

The launch starts Gazebo, map server, AMCL, Nav2, `bt_interface_node`,
perception/manipulation services, Foxglove Bridge, and the environment
publisher used by the frontend.

## Environment Profiles

Simulation environments are selected through centralized profiles in:

```text
src/bt_bringup/config/environments.json
```

Each profile binds:

- the Gazebo world file;
- the saved metric map;
- the robot spawn pose;
- the AMCL initial pose;
- startup delays for heavier environments.

Available profiles:

| Profile | Purpose |
| --- | --- |
| `aws_small_house` | Saved-map runtime for the AWS Small House environment |
| `aws_hospital` | Saved-map runtime for the AWS Hospital environment |

The active profile is published on:

```text
/btgen/environment
```

The frontend uses this topic to know which environment is active and to load
the matching semantic/topological data.

## Saved Maps

The repository contains the saved metric maps used by Nav2:

```text
maps/aws_small_house.yaml
maps/aws_small_house.pgm
maps/aws_hospital_final.yaml
maps/aws_hospital_final.pgm
```

The Docker compose file mounts `maps/` as `/workspace/maps`, so maps persist
across container recreation.

Runtime behavior:

- `map_server` publishes `/map`;
- AMCL estimates the robot pose on the saved map;
- Nav2 receives goals in the `map` frame;
- the frontend visualizes the metric map, robot pose, goal, and planned path.

## Mapping Runtime

Use the mapping launch only when a new saved map must be produced. It starts
Gazebo and SLAM, but deliberately avoids the full task-execution pipeline.

AWS Small House mapping:

```bash
ros2 launch bt_bringup robot_bt_mapping.launch.py \
  environment:=aws_small_house \
  use_rviz:=true \
  headless:=false
```

Save the map:

```bash
ros2 run nav2_map_server map_saver_cli \
  -f /workspace/maps/aws_small_house
```

AWS Hospital mapping:

```bash
ros2 launch bt_bringup robot_bt_mapping.launch.py \
  environment:=aws_hospital \
  use_rviz:=true \
  headless:=false
```

Drive the robot from a second sourced terminal:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Save the map after covering corridors, rooms, side walls, and endpoints from
multiple orientations:

```bash
ros2 run nav2_map_server map_saver_cli \
  -f /workspace/maps/aws_hospital
```


## Architecture

```text
Natural-language command
          |
          v
Inference server
FastAPI + Llama 3.2 + grammar-constrained generation
          |
          v
Behavior Tree XML
          |
          v
bt_interface_node
ROS 2 action lifecycle, validation, saving, execution orchestration
          |
          v
Nav2 + BehaviorTree.CPP
Navigation and Behavior Tree execution
          |
          v
Gazebo robot execution
TurtleBot3 Waffle Pi + OpenManipulator-X
```

## Project Structure

```text
BTGenCobot/
├── inference_server/           # LLM inference server
│   ├── api/                    # REST endpoints
│   ├── core/                   # Model loading and generation
│   ├── validation/             # XML validation and post-processing
│   └── prompts/                # Prompt templates
│
├── maps/                       # Saved metric maps for Nav2
├── worlds/                     # Gazebo world assets
├── robot_description/          # TurtleBot3 + OpenManipulator-X URDF
│
└── src/
    ├── bt_bringup/             # Launch files, Nav2 config, environment profiles
    ├── bt_text_interface/      # ROS 2 node/action for BT generation and execution
    ├── bt_nav2_plugins/        # Custom BT nodes
    ├── manipulator_control/    # Manipulator service
    ├── vision_services/        # Object detection service
    └── btgencobot_interfaces/  # Custom ROS 2 messages/actions
```

## Custom BT Nodes

| Node | Description |
| --- | --- |
| `DetectObject` | Open-vocabulary object detection |
| `ComputePathToPose` | Nav2 path computation toward a pose |
| `FollowPath` | Nav2 path following |
| `PickObject` | Approach and grasp detected object |
| `PlaceObject` | Place held object at detected location |
| `SpinLeft` / `SpinRight` | Rotate the base in place |

## Frontend Integration

The companion frontend connects to this backend through:

- HTTP requests to the inference server for command submission;
- Foxglove Bridge on port `8765` for ROS 2 topic exchange.

The frontend expects:

- `/btgen/environment` to identify the active profile;
- `/map`, `/amcl_pose`, `/odom`, and planning topics for map visualization;
- `/camera` for the robot camera stream;
- `/generated_behavior_tree`, `/bt_execution_feedback`, and
  `/behavior_tree_log` for Behavior Tree supervision.

## License

MIT
