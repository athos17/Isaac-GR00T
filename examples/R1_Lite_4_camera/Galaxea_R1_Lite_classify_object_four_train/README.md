---
task_categories:
  - robotics

language:
  - en


extra_gated_prompt: 'By accessing this dataset, you agree to cite the associated paper in your research/publications—see the "Citation" section for details. You agree to not use the dataset to conduct experiments that cause harm to human subjects.'



extra_gated_fields:

  Company/Organization:
    type: 'text'
    description: 'e.g., "ETH Zurich", "Boston Dynamics", "Independent Researcher"'

  Country:
    type: 'country'
    description: 'e.g., "Germany", "China", "United States"'



tags:
  - RoboCOIN
  - LeRobot

license: apache-2.0

configs:
- config_name: default
  data_files: data/chunk-{id}/episode_{id}.parquet
---

# Galaxea_R1_Lite_classify_object_four

## Dataset Description

This dataset uses an extended format based on LeRobot and is fully compatible with LeRobot.

## Task Preview

<video src="videos/chunk-000/observation.images.cam_head_left_rgb/episode_000000.mp4" controls width="640"></video>

[View Video Directly](videos/chunk-000/observation.images.cam_head_left_rgb/episode_000000.mp4)

### Overview

- **Total Episodes:** 191
- **Total Frames:** 153386
- **FPS:** 30
- **Dataset Size:** 8.39 GB
- **Robot Name:** `Galaxea_R1_Lite`
- **End-Effector Type:** `two_finger_gripper`
- **Teleoperation Type:** `Due to some reasons, this dataset temporarily cannot provide the teleoperation type information.`
- **Sensors:** `cam_head_left_rgb`, 
`cam_head_right_rgb`, 
`cam_left_wrist_rgb`, 
`cam_right_wrist_rgb`

- **Camera Information:** cam_head_left_rgb; 
cam_head_right_rgb; 
cam_left_wrist_rgb; 
cam_right_wrist_rgb

- **Scene:** `commercial_convenience->supermarket`
- **Objects:** `brown_basket(unknown)`, 
`yellow_basket(unknown)`, 
`any_fruits(unknown)`, 
`any_vegetables(unknown)`, 
`any_snacks(unknown)`, 
`any_bread(unknown)`

- **Task Description:** place the food in the right basket with the right gripper, and place the non food items in the left basket with the left gripper.


### Primary Task Instruction
> place the food in the right basket with the right gripper, and place the non food items in the left basket with the left gripper.

### Robot Configuration

- **Robot Name:** `Galaxea_R1_Lite`
- **Codebase Version:** `v2.1`
- **End-Effector Type:** `two_finger_gripper`
- **Teleoperation Type:** `Due to some reasons, this dataset temporarily cannot provide the teleoperation type information.`

## Scene and Objects

### Scene Type
`commercial_convenience->supermarket`

### Objects
- `brown_basket(unknown)`
- `yellow_basket(unknown)`
- `any_fruits(unknown)`
- `any_vegetables(unknown)`
- `any_snacks(unknown)`
- `any_bread(unknown)`


## Task Descriptions

- **Standardized Task Description:** `place the food in the right basket with the right gripper, and place the non food items in the left basket with the left gripper.`
- **Operation Type:** `Due to some reasons, this dataset temporarily cannot provide the operation type information.`

- **Environment Type:** `Due to some reasons, this dataset temporarily cannot provide the environment type information.`

### Sub-Tasks
This dataset includes 67 distinct subtasks:

1. **Grasp the potato chips and put it in the left basket** (Index: 0)
2. **Grasp the mineral water and put it in the right basket** (Index: 1)
3. **Grasp the rubiks cube and put it in the left basket** (Index: 2)
4. **Grasp the waffle and put it in the right basket** (Index: 3)
5. **Grasp the soft cleanser and put it in the left basket** (Index: 4)
6. **Grasp the back scratcher and put it in the left basket** (Index: 5)
7. **Grasp the apple and put it in the right basket** (Index: 6)
8. **End** (Index: 7)
9. **Grasp the white eraser and put it in the left basket** (Index: 8)
10. **Grasp the square chewing gum and put it in the right basket** (Index: 9)
11. **Grasp the power strip and put it in the left basket** (Index: 10)
12. **Grasp the green lemon and put it in the right basket** (Index: 11)
13. **Grasp the coke and put it in the right basket** (Index: 12)
14. **Grasp the cleaning agent and put it in the left basket** (Index: 13)
15. **Grasp the soda water and put it in the right basket** (Index: 14)
16. **Grasp the spoon and put it in the left basket** (Index: 15)
17. **Grasp the duck toys and put it in the left basket** (Index: 16)
18. **Grasp the triangle cake and put it in the right basket** (Index: 17)
19. **Grasp the cookie and put it in the right basket** (Index: 18)
20. **Grasp the yellow cake and put it in the right basket** (Index: 19)
21. **Grasp the shower sphere and put it in the left basket** (Index: 20)
22. **Grasp the compass and put it in the left basket** (Index: 21)
23. **Grasp the orange and put it in the right basket** (Index: 22)
24. **Grasp the broom and put it in the left basket** (Index: 23)
25. **Grasp the back scratcher and put it in the right basket** (Index: 24)
26. **Grasp the ballpoint pen and put it in the left basket** (Index: 25)
27. **Grasp the round bread and put it in the right basket** (Index: 26)
28. **Grasp the egg yolk pastry and put it in the right basket** (Index: 27)
29. **Grasp the soap and put it in the left basket** (Index: 28)
30. **Grasp the washing liquid and put it in the left basket** (Index: 29)
31. **Grasp the hard cleanser and put it in the left basket** (Index: 30)
32. **Grasp the milk and put it in the right basket** (Index: 31)
33. **Grasp the black marker and put it in the left basket** (Index: 32)
34. **Grasp the banana and put it in the right basket** (Index: 33)
35. **Grasp the can and put it in the left basket** (Index: 34)
36. **Grasp the black glass cup and put it in the left basket** (Index: 35)
37. **Grasp the brush and put it in the left basket** (Index: 36)
38. **Grasp the bath ball and put it in the left basket** (Index: 37)
39. **Grasp the blue towel and put it in the left basket** (Index: 38)
40. **Grasp the peeler and put it in the left basket** (Index: 39)
41. **Grasp the brown towel and put it in the left basket** (Index: 40)
42. **Grasp the peach and put it in the right basket** (Index: 41)
43. **Grasp the tea cup and put it in the left basket** (Index: 42)
44. **Grasp the round bread and put it in the left basket** (Index: 43)
45. **Grasp the chocolate and put it in the right basket** (Index: 44)
46. **Grasp the grey towel and put it in the left basket** (Index: 45)
47. **Grasp the canned cola and put it in the right basket** (Index: 46)
48. **Grasp the tape and put it in the left basket** (Index: 47)
49. **Grasp the bread slice and put it in the right basket** (Index: 48)
50. **Grasp the glasses case and put it in the left basket** (Index: 49)
51. **Grasp the triangle cake and put it in the left basket** (Index: 50)
52. **Grasp the peach doll and put it in the right basket** (Index: 51)
53. **Grasp the blue cup and put it in the left basket** (Index: 52)
54. **Grasp the pen container and put it in the left basket** (Index: 53)
55. **Grasp the red duck and put it in the left basket** (Index: 54)
56. **Grasp the long bread and put it in the right basket** (Index: 55)
57. **Grasp the yogurt and put it in the right basket** (Index: 56)
58. **Grasp the potato chips and put it in the right basket** (Index: 57)
59. **Grasp the can and put it in the right basket** (Index: 58)
60. **Grasp the egg beater and put it in the right basket** (Index: 59)
61. **Place the cookie in the center of the table** (Index: 60)
62. **Grasp the square chewing gum and put it in the left basket** (Index: 61)
63. **Grasp the ad milk and put it in the right basket** (Index: 62)
64. **Grasp the detergent and put it in the left basket** (Index: 63)
65. **Grasp the yellow duck and put it in the left basket** (Index: 64)
66. **Grasp the blue marker and put it in the left basket** (Index: 65)
67. **null** (Index: 66)


### Atomic Actions
- `grasp`
- `pick`
- `place`


## Hardware and Sensors


### Sensors

- `cam_head_left_rgb`

- `cam_head_right_rgb`

- `cam_left_wrist_rgb`

- `cam_right_wrist_rgb`




### Camera Information


- `cam_head_left_rgb`: dtype=video, shape=720x1280x3, resolution=1280x720, codec=av1, pix_fmt=yuv420p

- `cam_head_right_rgb`: dtype=video, shape=720x1280x3, resolution=1280x720, codec=av1, pix_fmt=yuv420p

- `cam_left_wrist_rgb`: dtype=video, shape=720x1280x3, resolution=1280x720, codec=av1, pix_fmt=yuv420p

- `cam_right_wrist_rgb`: dtype=video, shape=720x1280x3, resolution=1280x720, codec=av1, pix_fmt=yuv420p




### Coordinate System
- **Definition:** `right-hand-frame`


### Dimensions & Units
- **Joint Rotation:** `radian`
- **End-Effector Rotation:** `end_rotation_dim`
- **End-Effector Translation:** `end_translation_dim`




## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Episodes** | 191 |
| **Total Frames** | 153386 |
| **Total Tasks** | 67 |
| **Total Videos** | 764 |
| **Total Chunks** | 1 |
| **Chunk Size** | 1000 |
| **FPS** | 30 |
| **State Dimensions** | 14 |
| **Action Dimensions** | 14 |
| **Camera Views** | 4 |
| **Dataset Size** | 8.39 GB |


## Data Splits

The dataset is organized into the following splits:

- **Training**: Episodes 0:190


## Dataset Structure

This dataset follows the LeRobot format and contains the following components:

### Data Files
- **Videos**: Compressed video files containing RGB camera observations
- **State Data**: Robot joint positions, velocities, and other state information
- **Action Data**: Robot action commands and trajectories
- **Metadata**: Episode metadata, timestamps, and annotations

### File Organization
- **Data Path Pattern**: `data/chunk-{id}/episode_{id}.parquet`
- **Video Path Pattern**: `videos/chunk-{id}/observation.images.cam_left_wrist_rgb/episode_{id}.mp{id}`
- **Chunking**: Data is organized into 1 chunk(s)
of size 1000

### Data Structure (Tree)

```
Galaxea_R1_Lite_classify_object_four_qced_hardlink/
|-- annotations
|   |-- eef_acc_mag_annotation.jsonl
|   |-- eef_direction_annotation.jsonl
|   |-- eef_velocity_annotation.jsonl
|   |-- gripper_activity_annotation.jsonl
|   |-- gripper_mode_annotation.jsonl
|   |-- scene_annotations.jsonl
|   `-- subtask_annotations.jsonl
|-- data
|   `-- chunk-000
|       |-- episode_000000.parquet
|       |-- episode_000001.parquet
|       |-- episode_000002.parquet
|       |-- episode_000003.parquet
|       |-- episode_000004.parquet
|       |-- episode_000005.parquet
|       |-- episode_000006.parquet
|       |-- episode_000007.parquet
|       |-- episode_000008.parquet
|       |-- episode_000009.parquet
|       |-- episode_000010.parquet
|       `-- episode_000011.parquet
|       `-- ... (179 more entries)
|-- meta
|   |-- episodes.jsonl
|   |-- episodes_stats.jsonl
|   |-- info.json
|   `-- tasks.jsonl
|-- videos
|   `-- chunk-000
|       |-- observation.images.cam_head_left_rgb
|       |-- observation.images.cam_head_right_rgb
|       |-- observation.images.cam_left_wrist_rgb
|       `-- observation.images.cam_right_wrist_rgb
|-- info.yaml
`-- README.md
```

## Camera Views






This dataset includes 4 camera views: `cam_head_left_rgb`, `cam_head_right_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`.


## Features (Full YAML)

```yaml
observation.images.cam_head_left_rgb:
  dtype: video
  shape:
  - 720
  - 1280
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 720
    video.width: 1280
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.images.cam_head_right_rgb:
  dtype: video
  shape:
  - 720
  - 1280
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 720
    video.width: 1280
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.images.cam_left_wrist_rgb:
  dtype: video
  shape:
  - 720
  - 1280
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 720
    video.width: 1280
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.images.cam_right_wrist_rgb:
  dtype: video
  shape:
  - 720
  - 1280
  - 3
  names:
  - height
  - width
  - channels
  info:
    video.height: 720
    video.width: 1280
    video.codec: av1
    video.pix_fmt: yuv420p
    video.is_depth_map: false
    video.fps: 30
    video.channels: 3
    has_audio: false
observation.state:
  dtype: float32
  shape:
  - 14
  names:
  - left_arm_joint_1_rad
  - left_arm_joint_2_rad
  - left_arm_joint_3_rad
  - left_arm_joint_4_rad
  - left_arm_joint_5_rad
  - left_arm_joint_6_rad
  - right_arm_joint_1_rad
  - right_arm_joint_2_rad
  - right_arm_joint_3_rad
  - right_arm_joint_4_rad
  - right_arm_joint_5_rad
  - right_arm_joint_6_rad
  - left_gripper_open
  - right_gripper_open
action:
  dtype: float32
  shape:
  - 14
  names:
  - left_arm_joint_1_rad
  - left_arm_joint_2_rad
  - left_arm_joint_3_rad
  - left_arm_joint_4_rad
  - left_arm_joint_5_rad
  - left_arm_joint_6_rad
  - right_arm_joint_1_rad
  - right_arm_joint_2_rad
  - right_arm_joint_3_rad
  - right_arm_joint_4_rad
  - right_arm_joint_5_rad
  - right_arm_joint_6_rad
  - left_gripper_open
  - right_gripper_open
timestamp:
  dtype: float32
  shape:
  - 1
  names: null
frame_index:
  dtype: int64
  shape:
  - 1
  names: null
episode_index:
  dtype: int64
  shape:
  - 1
  names: null
index:
  dtype: int64
  shape:
  - 1
  names: null
task_index:
  dtype: int64
  shape:
  - 1
  names: null
subtask_annotation:
  names: null
  shape:
  - 5
  dtype: int32
scene_annotation:
  names: null
  shape:
  - 1
  dtype: int32
eef_sim_pose_state:
  names:
  - left_eef_pos_x
  - left_eef_pos_y
  - left_eef_pos_z
  - left_eef_rot_x
  - left_eef_rot_y
  - left_eef_rot_z
  - right_eef_pos_x
  - right_eef_pos_y
  - right_eef_pos_z
  - right_eef_rot_x
  - right_eef_rot_y
  - right_eef_rot_z
  shape:
  - 12
  dtype: float32
eef_sim_pose_action:
  names:
  - left_eef_pos_x
  - left_eef_pos_y
  - left_eef_pos_z
  - left_eef_rot_x
  - left_eef_rot_y
  - left_eef_rot_z
  - right_eef_pos_x
  - right_eef_pos_y
  - right_eef_pos_z
  - right_eef_rot_x
  - right_eef_rot_y
  - right_eef_rot_z
  shape:
  - 12
  dtype: float32
eef_direction_state:
  names:
  - left_eef_direction
  - right_eef_direction
  shape:
  - 2
  dtype: int32
eef_direction_action:
  names:
  - left_eef_direction
  - right_eef_direction
  shape:
  - 2
  dtype: int32
eef_velocity_state:
  names:
  - left_eef_velocity
  - right_eef_velocity
  shape:
  - 2
  dtype: int32
eef_velocity_action:
  names:
  - left_eef_velocity
  - right_eef_velocity
  shape:
  - 2
  dtype: int32
eef_acc_mag_state:
  names:
  - left_eef_acc_mag
  - right_eef_acc_mag
  shape:
  - 2
  dtype: int32
eef_acc_mag_action:
  names:
  - left_eef_acc_mag
  - right_eef_acc_mag
  shape:
  - 2
  dtype: int32
gripper_open_scale_state:
  names:
  - left_gripper_open_scale
  - right_gripper_open_scale
  shape:
  - 2
  dtype: float32
gripper_open_scale_action:
  names:
  - left_gripper_open_scale
  - right_gripper_open_scale
  shape:
  - 2
  dtype: float32
gripper_mode_state:
  names:
  - left_gripper_mode
  - right_gripper_mode
  shape:
  - 2
  dtype: int32
gripper_mode_action:
  names:
  - left_gripper_mode
  - right_gripper_mode
  shape:
  - 2
  dtype: int32
gripper_activity_state:
  names:
  - left_gripper_activity
  - right_gripper_activity
  shape:
  - 2
  dtype: int32
gripper_activity_action:
  names:
  - left_gripper_activity
  - right_gripper_activity
  shape:
  - 2
  dtype: int32

```

## Available Annotations

This dataset includes rich annotations to support diverse learning approaches:

- `eef_acc_mag_annotation.jsonl`
- `eef_direction_annotation.jsonl`
- `eef_velocity_annotation.jsonl`
- `gripper_activity_annotation.jsonl`
- `gripper_mode_annotation.jsonl`
- `scene_annotations.jsonl`
- `subtask_annotations.jsonl`


## Dataset Tags

- `RoboCOIN`
- `LeRobot`


## Authors

### Contributors
This dataset is contributed by:-RoboCOIN Team at Beijing Academy of Artificial Intelligence (BAAI)

### Annotators
No annotator information available.

## Links

- **Homepage:** [https://flagopen.github.io/RoboCOIN/](https://flagopen.github.io/RoboCOIN/)
- **Paper:** [https://arxiv.org/abs/2511.17441](https://arxiv.org/abs/2511.17441)
- **Repository:** [https://github.com/FlagOpen/RoboCOIN](https://github.com/FlagOpen/RoboCOIN)
## Contact and Support

For questions, issues, or feedback regarding this dataset, please contact us.
### Support
For technical support, please open an issue on our GitHub repository.

## License

apache-2.0

## Citation

If you use this dataset in your research, please cite:

```bibtex
@article{robocoin,
  title={RoboCOIN: An Open-Sourced Bimanual Robotic Data Collection for Integrated Manipulation},
  author={Shihan Wu, Xuecheng Liu, Shaoxuan Xie, Pengwei Wang, Xinghang Li, Bowen Yang, Zhe Li, Kai Zhu, Hongyu Wu, Yiheng Liu, Zhaoye Long, Yue Wang, Chong Liu, Dihan Wang, Ziqiang Ni, Xiang Yang, You Liu, Ruoxuan Feng, Runtian Xu, Lei Zhang, Denghang Huang, Chenghao Jin, Anlan Yin, Xinlong Wang, Zhenguo Sun, Junkai Zhao, Mengfei Du, Mingyu Cao, Xiansheng Chen, Hongyang Cheng, Xiaojie Zhang, Yankai Fu, Ning Chen, Cheng Chi, Sixiang Chen, Huaihai Lyu, Xiaoshuai Hao, Yequan Wang, Bo Lei, Dong Liu, Xi Yang, Yance Jiao, Tengfei Pan, Yunyan Zhang, Songjing Wang, Ziqian Zhang, Xu Liu, Ji Zhang, Caowei Meng, Zhizheng Zhang, Jiyang Gao, Song Wang, Xiaokun Leng, Zhiqiang Xie, Zhenzhen Zhou, Peng Huang, Wu Yang, Yandong Guo, Yichao Zhu, Suibing Zheng, Hao Cheng, Xinmin Ding, Yang Yue, Huanqian Wang, Chi Chen, Jingrui Pang, YuXi Qian, Haoran Geng, Lianli Gao, Haiyuan Li, Bin Fang, Gao Huang, Yaodong Yang, Hao Dong, He Wang, Hang Zhao, Yadong Mu, Di Hu, Hao Zhao, Tiejun Huang, Shanghang Zhang, Yonghua Lin, Zhongyuan Wang and Guocai Yao},
  journal={arXiv preprint arXiv:2511.17441},
  url = {https://arxiv.org/abs/2511.17441},
  year={2025},
  }

```

### Additional References

If you use this dataset, please also consider citing:
LeRobot Framework: https://github.com/huggingface/lerobot


## Version Information

Initial Release
