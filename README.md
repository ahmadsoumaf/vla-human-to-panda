### Real-to-Simulation Robot Manipulation

![NVIDIA Isaac Sim](https://img.shields.io/badge/NVIDIA-Isaac%20Sim-76B900?logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Perception-5C3EE8?logo=opencv&logoColor=white)
![SmolVLA](https://img.shields.io/badge/VLA-SmolVLA-orange)
![OpenUSD](https://img.shields.io/badge/OpenUSD-Simulation-lightgrey)

<img width="960" height="270" alt="real_vs_isaac_sim" src="https://github.com/user-attachments/assets/44e93ace-ddba-472b-8b7b-ae117fbd7f55" />


A real-to-simulation robot learning pipeline that converts simple phone-recorded human manipulation demonstrations into Panda robot demonstrations in NVIDIA Isaac Sim and exports them as a LeRobot dataset for Vision-Language-Action training.

The project explores how a small amount of real human demonstration data can be transferred to a different robotic embodiment using perception, task-phase detection, trajectory retargeting, simulation and VLA-compatible dataset generation.

---

## Project Goal

The objective is to use real manipulation data recorded with a phone to drive a robotic manipulator in simulation.

The task used in this project is:

> **Pick up the blue triangular block and place it on the target.**

The complete pipeline is:

```text
Phone Video
    ↓
Hand + Object Perception
    ↓
Grasp / Carry / Release Detection
    ↓
Demo Quality Validation
    ↓
Human-to-Panda Retargeting
    ↓
NVIDIA Isaac Sim
    ↓
Physical Panda Grasp + Placement
    ↓
Robot Camera + State + Action Recording
    ↓
LeRobot Dataset
    ↓
SmolVLA Fine-Tuning
```

---

## Main Contribution

Rather than directly teleoperating the robot or manually specifying trajectories, the project starts from **human manipulation recorded using a standard phone camera**.

The pipeline automatically:

1. detects the human hand and manipulated object;
2. identifies the manipulation phases;
3. rejects demonstrations with unreliable grasp or release perception;
4. converts the accepted human motion into a Panda-compatible trajectory;
5. executes a physical grasp and placement inside Isaac Sim;
6. records robot observations, states and actions;
7. exports the resulting demonstrations in LeRobot format for VLA training.

This allows a small number of real human demonstrations to be transformed into structured robot-learning data.

---

## Current Results

Three human demonstrations passed the complete quality-validation pipeline.

| Demo | Pick observed | Release observed | Panda placement error | VLA dataset |
|---|---:|---:|---:|---:|
| `demo_001_best` | Yes | Yes | 5.5 mm | Accepted |
| `demo_002` | Yes | Yes | 6.0 mm | Accepted |
| `demo_004` | Yes | Yes | 5.7 mm | Accepted |

Demonstrations with inaccurate grasp detection were rejected even when the Panda replay happened to reach the target.

This keeps **robot replay success separate from demonstration quality**.

The final LeRobot dataset currently contains:

```text
Episodes:       3
Frames:         1,257
Frame rate:     30 FPS
Camera:         640 × 360 RGB
Robot state:    8 dimensions
Action:         8 dimensions
Task:
"Pick up the blue triangle and place it on the target."
```

The 8-D state contains:

```text
7 Panda arm joints
+
gripper opening
```
---

## Perception

The phone-video processing pipeline uses:

- MediaPipe hand landmarks
- OpenCV
- HSV object segmentation
- phase-aware validation
- grasp/release detection
- post-release target estimation
---

## Human-to-Panda Retargeting

Human motion is converted into Panda motion using the task geometry of the simulated table.

The retargeting stage creates a sequence containing:

```text
Approach
Descend to grasp
Close gripper
Lift
Transport
Descend to target
Open gripper
Lift after placement
```

The robot then executes this trajectory inside NVIDIA Isaac Sim.

The Panda controller handles:

- inverse kinematics;
- joint commands;
- gripper commands;
- grasp validation;
- placement success checking;
- trajectory execution logging.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/vla-human-to-panda.git
cd vla-human-to-panda
```

### 2. Set up the perception environment

```bash
./setup_perception_env.sh
```

### 3. Build the Isaac Sim scene

```bash
./run_scene.sh
```

---

## Processing a Human Demonstration

Process a phone video:

```bash
./run_process_demo.sh data/real/videos/demo_001_best.mp4
```

This performs:

```text
video
→ hand tracking
→ object tracking
→ phase detection
→ grasp/release detection
→ validation
→ processed demonstration
```

---

## Replay Human Motion with Panda

After processing:

```bash
./run_human_replay.sh demo_001_best
```

The system retargets the accepted human trajectory and executes it with the Panda.

The resulting report records whether the demonstration is suitable for the VLA dataset.

---

## Record Accepted Robot Episodes

```bash
./run_record_episodes.sh
```

This records:

- front-camera RGB observations;
- robot state;
- robot actions;
- episode metadata.

Only accepted demonstrations are used.

---

## Build the LeRobot Dataset

```bash
./run_build_lerobot_dataset.sh --overwrite
```

Example output:

```text
3 episodes
1257 frames
30 fps
image: (3, 360, 640)
state: (8,)
action: (8,)
```

---

## SmolVLA

The exported dataset can be used for SmolVLA fine-tuning through LeRobot.

Training entry point:

```bash
./run_train_smolvla.sh
```

The current three-episode dataset is primarily intended to validate the complete:

```text
real demonstration
→ robot demonstration
→ VLA dataset
→ VLA training
```

pipeline.

A larger and more spatially diverse dataset is required for strong generalisation.

---

## Pipeline Freeze

To ensure that dataset generation remains reproducible, the final perception, retargeting and controller pipeline is fingerprinted using SHA-256 hashes.

Check that the frozen pipeline has not changed:

```bash
python3 tools/freeze_manifest.py --check
```

If one of the frozen files changes, the check fails.

---

## Tests

Run the validation tests with:

```bash
python3 -m pytest tests/ -q
```
---

## Dataset Export

Accepted demonstrations are replayed in Isaac Sim while recording:

```text
observation.images.front
observation.state
action
task
```

They are then exported into LeRobot v3.0 format.

The final dataset is located at:

```text
data/lerobot/fls_panda_pick_place/
```

---

## Repository Structure

```text
vla_challenge/
│
├── README.md
├── FROZEN_PIPELINE.json
│
├── perception/
│   ├── process_demo.py
│   ├── hand_tracker.py
│   ├── object_tracker.py
│   ├── target_tracker.py
│   ├── synthetic_demo.py
│   └── default_config.json
│
├── retargeting/
│   ├── human_to_panda.py
│   └── replay_human_demo.py
│
├── isaac/
│   ├── build_fls_pick_place_scene.py
│   ├── panda_common.py
│   ├── task_geometry.py
│   ├── fls_pick_place_scene.usd
│   └── fls_pick_place_scene.anchors.json
│
├── export/
│   ├── record_sim_episodes.py
│   └── build_lerobot_dataset.py
│
├── assets/
│   ├── fls/
│   └── models/
│
├── data/
│   ├── real/
│   │   ├── videos/
│   │   └── processed/
│   └── lerobot/
│       └── fls_panda_pick_place/
│
├── tests/
│   ├── test_phase_validation.py
│   └── test_retargeting.py
│
├── tools/
│   └── freeze_manifest.py
│
├── setup_perception_env.sh
├── run_scene.sh
├── run_process_demo.sh
├── run_human_replay.sh
├── run_record_episodes.sh
├── run_build_lerobot_dataset.sh
└── run_train_smolvla.sh
```

---



## Current Limitations

The present dataset is intentionally small.

Current limitations include:

- only three accepted human demonstrations;
- limited variation in object and target positions;
- 2D phone-video perception;
- a relatively small object in the simulated camera view;
- limited viewpoint diversity;
- the current dataset is better suited to pipeline validation than robust VLA generalisation.

These limitations provide clear directions for future work.

---

## Future Work

Planned extensions include:

- closed-loop SmolVLA control in Isaac Sim;
- randomised block positions;
- randomised target positions;
- more real human demonstrations;
- simulation-based demonstration augmentation;
- evaluation on unseen layouts;
- success-rate comparison between small-data and augmented-data policies;
- improved camera viewpoints;
- stronger visual target representation.

A longer-term pipeline could become:

```text
Small Real Dataset
       ↓
Human-to-Robot Retargeting
       ↓
Simulation Augmentation
       ↓
VLA Fine-Tuning
       ↓
Closed-Loop Robot Policy
       ↓
Evaluation on Unseen Tasks
```


## Challenge Summary

This project demonstrates how a small amount of applicant-recorded real-world manipulation data can be transformed into robot-learning demonstrations for a different embodiment.


