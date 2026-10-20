# UR10e + Robotiq 2F-85 プロファイル

`robot_profiles.create_robot(world, asset_root, profile='robotiq85')` は、NVIDIA の組立済み UR10e USDA を参照し、トップの `Physics=physx`、`Gripper=Robotiq_2F_85`、子グリッパーの `Physics=Physx_Mimic` を選択する。USD・メッシュを本リポジトリへコピーしない。

```python
from src.sorter.robot_profiles import create_robot

robot = create_robot(world, asset_root, profile='robotiq85')
# world.reset() 後に articulation / gripper を使用する。
# IK controller の設定先は robot.sorter_kinematics_dir。
```

root は `/ur`、実際の EE は `/ur/Robotiq_2F_85/base_link`、駆動ジョイントは `/ur/Robotiq_2F_85/Joints/finger_joint`。開放は0 rad、閉鎖指令は0.8 rad。USD の上限47度に対して0.8 radは範囲内であり、factoryでも取得した制限を確認する。接触時の挙動は実Simで別途検証する。

`ur10e.urdf` は既存の NVIDIA 由来 UR10e URDF から arm 6軸と固定rootを再利用した、IK専用の定義。指のジョイント、メッシュ、慣性、衝突回避の形状は含まない。既存 `IKLerpController` に合わせて、EEフレーム名は `ee_link_robotiq_arg2f_base_link` を維持した。実際のUSDの85baseと同じ座標系を、この名前で表現する。

マウントは新USDの `wrist_3_link/ft_frame`、組立 `AssemblerFixedJoint` の local pose 0・1、および85baseの配置から2通りで求めた。

```text
T(wrist3,85base) = T(wrist3,ft_frame) × T(joint0) × inverse(T(joint1))
                 ≈ translation(0,0,0) × rotation_z(+pi/2)
```

`static_evidence.json` に取得元URL・SHA256・数値比較を保存した。armの各ジョイント局所変換とゼロ姿勢FKの最大成分誤差は1.47e-7、マウントはFixedJoint経由1.34e-7・配置姿勢経由1.63e-7で、許容1e-6以内。これは静的な整合性確認であり、把持成功率を表さない。

`verify_static.py` は USDA をメモリへ取得し、Sdf の属性だけを読む再検証スクリプト。SimulationAppを起動せず、メッシュを展開せず、取得したNVIDIAファイルも保存しない。Isaac同梱Pythonで `omni.usd.libs` のPythonパスと共有ライブラリパスを有効にして実行する。

```text
verify_static.py --out static_evidence.json
```

`contact_geometry.json` は開いた指先の可視メッシュ384点から求めた概算。左padの85base座標は X約±0.011 m、Y=-0.06271〜-0.04355 m、Z=0.11042〜0.14842 m。右padはYが反対側なので、**閉じる方向はbaseローカルY**。Z中央は0.12942 mであり、初期 `EE_OFFSET=0.13` は指先領域の中央を部品中心へ合わせる目安になる。0.14なら部品の上半分寄りを挟む。

base姿勢が roll=0、pitch=pi、yaw=0 のとき、閉じる方向はworld Y。この部品の短辺0.0515 mを挟める向きになる。yaw=pi/2を追加するとworld Xに変わり、長辺0.10 mが開いたpad間隔約0.0871 mを超える。メッシュの可視範囲は衝突形状・接触位置の完全な保証ではないため、実Simでは指と部品の位置も確認する。

参照元は NVIDIA Isaac Sim 6.0 の `Isaac/Robots_Multiphysics/UniversalRobots/ur10e/` と `Isaac/Robots_Multiphysics/Robotiq/2F-85/`。本フォルダにあるのは、再利用したarmの運動学数値、小さな独自設定・検証コード・数値証跡だけ。
