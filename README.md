# AI外観検査・仕分けセル

Physical AI 応用1講座の最終課題。Isaac Sim 上の UR10e + Robotiq 2F-85 で直方体部品を撮影し、小型 CNN が上面の模擬汚れを判定して「良品 / 不良品 / 要確認」のトレイに運び分ける。

供給位置と姿勢は固定し、部品を1個ずつ検査する。部品の供給装置や実機への接続は対象外。分類器への入力は RGB 画像だけで、シミュレータの正解ラベル・物体名・搬送先を特徴に使わない。

実測値と未達事項は [docs/RESULTS.md](docs/RESULTS.md)、提出前の確認は [docs/SUBMISSION.md](docs/SUBMISSION.md) を参照。実装済みの機能と実測で成功した範囲を区別して記録する。

## 構成

```text
main.py                         検査 → 判定 → 搬送 → 結果保存
run.sh                          Isaac Sim アプリ起動
run_python.sh                   Isaac 同梱 Python で学習・評価などを実行
scripts/generate_dataset.py     条件グループ単位で分離した画像を生成
scripts/train_classifier.py     小型 CNN 学習・検証しきい値決定
scripts/evaluate_classifier.py  固定済みモデルを独立 test で評価
scripts/baseline.py             流用元の搬送ベースライン
scripts/summarize_runs.py       搬送 metrics JSON / 検査 JSONL の集計
scripts/make_video.py           カメラ PNG 連番から MP4 を作成
src/sorter/cell_scene.py        部品・模擬汚れ・照明・カメラ・3トレイ
src/sorter/vision/              前処理・CNN・画像だけの推論・評価
src/sorter/planning/            ロボットの把持・搬送制御
src/sorter/measurement.py       全体形状・開放後静止の検証、CSV / JSONL
tests/                         分離漏れ・判定・搬送測定の検証
```

`data/`、`models/`、`artifacts/`、`recordings/` は生成物で、Git の追跡対象から除外する。Isaac Sim 本体と NVIDIA アセットは同梱しない。

## セットアップ

検証環境は NVIDIA DGX Spark / GB10、Linux aarch64、Isaac Sim **6.0.0-rc.22**。Isaac Sim 付属の Python・PyTorch・NumPy・Pillow を利用する。ROS 2、scikit-learn、外部 AI API は不要。

Isaac Sim を別途用意してから、リポジトリのルートで実行する。インストール場所は環境変数で指定できる。

```bash
export ISAAC_ROOT=/path/to/IsaacSim/release
./run.sh --help
```

`ISAAC_PYTHON` で `python.sh` 自体の場所も指定できる。初期値は `ISAAC_ROOT/python.sh`。ランチャーは GB10 の実行に必要な Isaac 同梱 PyTorch ライブラリへの参照を追加する。システム Python へのパッケージ追加や sudo は不要。

初回のロボットアセット取得にはネットワークと待ち時間が必要。既存のローカルアセットを利用する場合は `ROBOT_USD`、アセットの参照ルートを変える場合は `ISAAC_ASSET_ROOT` を指定する。

```bash
export ROBOT_USD=/path/to/ur_gripper.usd
```

アセット本体は公開リポジトリにコピーしない。起動が止まったように見える場合は `[startup]` の最後の段階と Isaac の警告・例外を確認する。モデル学習時には SimulationApp を起動する必要はない。

## 画像生成・学習・評価

以下は再現用の実行手順であり、成功済みの実測結果を表すものではない。出力済みのモデルや test 評価を残すため、再実験には新しい出力先を使う。

```bash
SORTER_ENTRYPOINT=scripts/generate_dataset.py ./run.sh --headless \
  --out-dir data/inspection --groups 150 --seed 42

./run_python.sh scripts/train_classifier.py \
  --manifest data/inspection/manifest.json \
  --out-dir models/inspection --epochs 100 --seed 42

./run_python.sh scripts/evaluate_classifier.py \
  --manifest data/inspection/manifest.json \
  --model-dir models/inspection --out artifacts/test_evaluation.json
```

150条件グループで各条件の良品・不良品を対にして、合計300枚を生成する。train / val / test は条件グループの 60% / 20% / 20%（180 / 60 / 60枚）。照明、部品の色、汚れの位置・大きさ・濃さを変え、test には暗い照明の条件も含める。同じ条件の対を別の split に分けない。manifest の読み込み時にもグループ漏れ・画像重複・クラス不足を検査する。

CNN は 64×64 RGB を入力とする3層の小型畳み込みモデルで、学習を最初から行う。train だけで重みを更新し、val で採用 epoch、スコアの temperature、要確認の信頼度しきい値を決定する。`confidence = max(不良スコア, 1 - 不良スコア)` が val で決めた値を下回れば `review`、それ以外はスコア0.5を境に `good` / `defect` とする。

比較方式は画像中央の明度の下位10%点を使う単純しきい値。明度方式も train で基準を求め、val で要確認の範囲を決める。test は設定変更に使用せず、`evaluate_classifier.py` で独立して測る。出力には正解率、要確認率、採用率、採用分の誤り率、条件別評価、誤分類画像名を残す。要確認を増やすほど誤りが小さく見えるため、正解率と採用率を併記する。

主な出力:

- `data/inspection/manifest.json`: 条件・ラベル・split と相対画像パス。
- `models/inspection/classifier.pt`: CNN の重み・前処理・校正値・判定しきい値。
- `models/inspection/brightness.json`: 明度方式の設定。
- `models/inspection/training.json`: 学習履歴と val 評価。test は含まない。
- `artifacts/test_evaluation.json` / `.predictions.csv`: 固定済み2方式の test 評価。

## 検査・仕分けの実行

学習済みモデルを使い、良品と不良品を交互に検査する例:

```bash
./run.sh --headless --trials 2 \
  --model models/inspection/classifier.pt --sequence good,defect \
  --output-dir artifacts/demo
```

`--sequence` はシミュレータで生成する部品の順番であり、分類器には渡さない。`good` が良品、`defect` が模擬汚れあり。行先は撮影画像への推論結果で選択する。要確認の発生は val で固定した信頼度範囲による。

分類器を使わず、搬送機構だけを個別に検証する例:

```bash
./run.sh --headless --trials 3 --motion-only good \
  --output-dir artifacts/motion_good
./run.sh --headless --trials 3 --motion-only defect \
  --output-dir artifacts/motion_defect
./run.sh --headless --trials 3 --motion-only review \
  --output-dir artifacts/motion_review
```

このモードの結果は AI による仕分け成功率と分けて集計する。機構が安定した後、`--trials 20` などで通し動作を測定する。失敗時停止のため、指定試行数より少ない件数で終わる場合がある。

失敗例も含めて指定件数を測る評価専用モードでは、明示的に `--continue-after-failure` を付ける。試行間で供給位置とロボット姿勢をリセットする。通常のデモ実行は失敗時停止を使う。

```bash
./run.sh --headless --trials 20 --continue-after-failure \
  --model models/inspection/classifier.pt --sequence good,defect \
  --output-dir artifacts/evaluation20
```

成功判定は、**部品の全体が指定トレイ内に入り、床の近くにあり、グリッパー開放後に連続して静止すること**。回転した部品の全体 AABB を使い、中心位置だけでは成功にしない。標準の静止しきい値は並進速度0.015 m/s未満・角速度0.15 rad/s未満を連続60 physics steps。200 Hz の場合は0.30秒に相当する。各試行の実際の設定と判定理由は保存された metrics を優先する。

検査画像、不良スコア、信頼度、行先、搬送結果・失敗理由を出力先へ保存する。`metrics.json` が試行全体、`inspections.csv` / `inspections.jsonl` が検査ログ。CSV と JSONL は試行ごとに追記し、途中で停止しても確定済みの結果を残す。`METRICS_OUT` で metrics JSON の保存先を上書きできる。物理搬送の成功と、正解ラベルに対する仕分けの正しさは別に確認し、要確認に運んだ件数も併記する。

```bash
python3 scripts/summarize_runs.py 'artifacts/motion_*/metrics.json' \
  --out artifacts/motion_summary.json
```

同一試行の metrics と JSONL を二重に入力しない。未確定・不明の結果は実測成功率の分母から分けて報告する。旧 `scripts/baseline.py` の距離中心の成功判定と、新セルの厳密な成功判定も混ぜない。

## フレーム保存と試し動画

ターミナルやデスクトップ全体を撮らず、シミュレーションのカメラ画像を連番保存する。

```bash
./run.sh --headless --trials 2 \
  --model models/inspection/classifier.pt --sequence good,defect \
  --output-dir artifacts/recording_demo --record-dir recordings/demo/frames

python3 scripts/make_video.py recordings/demo/frames recordings/demo.mp4 --fps 20
```

エンコードには ffmpeg が必要。利用できない場合も PNG 連番を保存できる。標準は `--record-every 10`、物理時間刻み0.005秒なので20 fps。保存間隔を変える場合は `--fps = 200 / record-every` に合わせる。動画はシミュレーション時間に対応し、処理にかかった実時間とは異なる。提出動画で倍速を使う場合は動画内または概要欄に明記する。

## 検証と制約

```bash
./run_python.sh -m unittest discover -s tests -v
```

分類器の評価対象はシミュレーションで生成した画像。実カメラ・実部品への汎化性能は未評価。固定位置・固定姿勢・既知寸法で制御するため、未知姿勢の検出や供給装置の自動化は含まない。把持の失敗、到達不能、カメラ画像取得失敗、タイムアウトは成功に数えず、停止理由を残して調整する。

実測で未完の部分は [docs/RESULTS.md](docs/RESULTS.md) に記載する。結果や動画を公開する前に、生成物・ログ・画面内の文字に個人情報、認証情報、業務データが含まれないことを確認する。
