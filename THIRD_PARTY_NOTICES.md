# Third-Party Notices

このリポジトリには、NVIDIA Isaac Sim のサンプルに由来するファイルが含まれる。

- 出典: [isaac-sim/IsaacSim](https://github.com/isaac-sim/IsaacSim)
  `source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/`
- 著作権: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- ライセンス: Apache License 2.0（本文: [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)）

| このリポジトリのファイル | 出典 | 変更 |
|---|---|---|
| `src/sorter/planning/rmpflow/robot_descriptor.yaml` | `rmpflow/robot_descriptor.yaml` | なし |
| `src/sorter/planning/rmpflow/ur10e_rmpflow_common.yaml` | `rmpflow/ur10e_rmpflow_common.yaml` | なし |
| `src/sorter/planning/rmpflow/ur10e.urdf` | `rmpflow/ur10e.urdf` | `ee_link_root_joint` の yaw を -90° から +90° に変更（Isaac Sim 6.0 の UR10e サンプルの組み付け姿勢に合わせた） |
| `src/sorter/planning/profiles/ur10e_85/ur10e.urdf` | `rmpflow/ur10e.urdf` | アームの運動学のみ残し、visual / collision 形状を削除 |
| `src/sorter/planning/profiles/ur10e_85/robot_descriptor.yaml` | `rmpflow/robot_descriptor.yaml` | Robotiq 2F-85 構成向けに再構成 |
| `src/sorter/tasks/pick_place.py` | `tasks/pick_place.py` | 搬送元・搬送先の KLT ビンと追加の直方体を配置、物体へのセマンティックラベル付与を追加 |
| `src/sorter/planning/pick_place_controller.py` | `controller/pick_place.py` | IK + 補間による経路生成、把持姿勢の yaw 指定などを追加 |
| `src/sorter/planning/rmpflow_controller.py` | `controller/rmpflow.py` | 設定ファイルの参照先と初期化の書き方を変更 |

## 同梱していないもの

- URDF が参照するメッシュ（`./meshes/*.obj`）、ロボット・トレイ等の USD アセット、Isaac Sim 本体は同梱していない。
  これらは実行時に Isaac Sim のアセットから参照する。NVIDIA のアセットには
  [Isaac Sim Additional Software and Materials License](https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/)
  が適用される。
