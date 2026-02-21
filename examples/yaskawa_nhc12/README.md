# YASKAWA NHC12 with LeRobot - Direct Teach ACT Implementation

このガイドでは、YASKAWA NHC12ロボットコントローラーをLeRobotに統合し、ダイレクトティーチモードでデータを収集してACT (Action Chunking Transformer) モデルをトレーニングする方法を説明します。

## 目次

1. [必要な機器](#必要な機器)
2. [セットアップ](#セットアップ)
3. [ロボット通信プロトコルの設定](#ロボット通信プロトコルの設定)
4. [データ収集](#データ収集)
5. [ACTモデルのトレーニング](#actモデルのトレーニング)
6. [モデルの展開](#モデルの展開)
7. [トラブルシューティング](#トラブルシューティング)

## 必要な機器

- YASKAWA NHC12ロボットコントローラー
- YASKAWA製ロボットアーム（6軸推奨）
- RealSenseカメラ（D435/D455など）1台以上
- Linux PC（Ubuntu 20.04以上推奨）
- Ethernet接続（ロボットコントローラーとPC間）

## セットアップ

### 1. LeRobotのインストール

```bash
# LeRobotのインストール
cd /home/kazu/lerobot_fork
pip install -e .

# RealSenseカメラのサポートをインストール
pip install pyrealsense2
```

### 2. ネットワーク設定

YASKAWA NHC12コントローラーとPCを同じネットワークに接続します。

#### ロボット側の設定:
1. コントローラーのネットワーク設定を開く
2. IPアドレスを設定（例: `192.168.1.31`）
3. サブネットマスクを設定（例: `255.255.255.0`）
4. TCP/IP通信ポートを設定（デフォルト: `10040`）

#### PC側の設定:
```bash
# Ethernet接続のIPアドレスを同じサブネットに設定
sudo ip addr add 192.168.1.100/24 dev eth0
```

### 3. カメラの設定

RealSenseカメラを接続し、動作確認します。

```bash
# RealSenseカメラのシリアル番号を確認
realsense-viewer

# または、Pythonで確認
python -c "import pyrealsense2 as rs; \
    ctx = rs.context(); \
    [print(f'Camera {i}: {d.get_info(rs.camera_info.serial_number)}') \
    for i, d in enumerate(ctx.query_devices())]"
```

カメラのシリアル番号をメモして、`record_direct_teach.py`の設定に追加してください。

## ロボット通信プロトコルの設定

**重要**: このLeRobot統合は、YASKAWA NHC12の通信プロトコルの詳細に基づいてカスタマイズする必要があります。

### 必要な情報

以下の情報をYASKAWAのマニュアルから取得し、実装を更新してください:

1. **コマンド形式**: ジョイント位置の読み取りコマンド
   ```
   例: "READ_JOINT_POS\r\n"
   ```

2. **レスポンス形式**: ジョイント位置のレスポンスフォーマット
   ```
   例: "J1=10.5,J2=20.3,J3=30.1,J4=40.2,J5=50.0,J6=60.1\r\n"
   ```

3. **動作コマンド**: 目標位置への移動コマンド
   ```
   例: "MOVE_JOINT J1=10.5,J2=20.3,...\r\n"
   ```

4. **ダイレクトティーチモード**: 重力補償モードの有効化/無効化コマンド
   ```
   例: "TEACH_MODE ON\r\n" / "TEACH_MODE OFF\r\n"
   ```

### 実装の更新

[robot_yaskawa_nhc12.py](../../src/lerobot/robots/yaskawa_nhc12/robot_yaskawa_nhc12.py) ファイルの以下のメソッドを更新してください:

- `_send_command()`: 通信プロトコルの実装
- `_read_joint_positions()`: ジョイント位置読み取りの実装
- `_write_joint_positions()`: ジョイント位置書き込みの実装
- `_enable_direct_teach_mode()`: ダイレクトティーチモードの有効化
- `_disable_direct_teach_mode()`: ダイレクトティーチモードの無効化

## データ収集

### 1. 設定ファイルの編集

`record_direct_teach.py` を開き、以下のパラメータを更新してください:

```python
# ロボット設定
ROBOT_IP = "192.168.1.31"  # ロボットのIPアドレス
ROBOT_PORT = 10040  # 通信ポート

# データセット設定
HF_REPO_ID = "<your_hf_username>/<dataset_name>"  # HuggingFaceのユーザー名とデータセット名
TASK_DESCRIPTION = "Pick and place object"  # タスクの説明

# ジョイント設定（ロボットの仕様に合わせて更新）
JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
JOINT_LIMITS = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-130.0, 90.0),
    # ... 残りのジョイント
}

# カメラ設定
CAMERA_CONFIGS = {
    "wrist": RealSenseCameraConfig(
        serial_number="123456789",  # カメラのシリアル番号
        width=640,
        height=480,
        fps=30,
    ),
}
```

### 2. データ収集の実行

```bash
cd /home/kazu/lerobot_fork/examples/yaskawa_nhc12
python record_direct_teach.py
```

### 3. データ収集の流れ

1. スクリプトが起動し、ロボットに接続します
2. ダイレクトティーチモードが有効になります（ロボットを手で動かせる状態）
3. Enterキーを押すと、エピソードの記録が開始されます
4. ロボットを手で動かしてタスクをデモンストレーションします
5. 記録が終了したら、エピソードを保存するか再記録するか選択します
6. 必要なエピソード数（デフォルト50）を収集します

### データ収集のヒント

- **一貫性**: 各エピソードで同じ開始位置と終了位置を使用
- **スムーズな動き**: 急激な動きを避け、滑らかに動かす
- **視野の確保**: カメラが対象物をしっかり捉えられる位置に配置
- **多様性**: 若干異なる軌道でタスクを実行（オーバーフィッティングを防ぐ）
- **品質**: 失敗したエピソードは再記録する

## ACTモデルのトレーニング

データ収集が完了したら、ACTモデルをトレーニングします。

### 1. トレーニングコマンド

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=<your_hf_username>/<dataset_name> \
  --training.num_epochs=1000 \
  --training.batch_size=8 \
  --training.learning_rate=1e-4 \
  --policy.chunk_size=100 \
  --policy.n_obs_steps=1 \
  --wandb.enable=true \
  --wandb.project=yaskawa_nhc12_act
```

### 2. トレーニングパラメータの説明

- `--policy=act`: ACTポリシーを使用
- `--dataset.repo_id`: 収集したデータセットのHuggingFace Hub ID
- `--training.num_epochs`: トレーニングエポック数（1000推奨）
- `--training.batch_size`: バッチサイズ（GPUメモリに応じて調整）
- `--policy.chunk_size`: アクションチャンクのサイズ（100推奨）
- `--wandb.enable`: Weights & Biasesでのログを有効化（オプション）

### 3. トレーニングの監視

Weights & Biasesを使用している場合、トレーニングの進捗を確認できます:
- ロス曲線
- 検証精度
- サンプル予測の可視化

トレーニングには、データセットサイズとGPUによりますが、数時間から1日程度かかります。

## モデルの展開

トレーニングが完了したら、ロボットでモデルを実行します。

### 1. 評価スクリプトの作成

```python
# evaluate_act.py
from lerobot.policies.factory import make_policy
from lerobot.robots.yaskawa_nhc12 import YaskawaNHC12Config, YaskawaNHC12Robot
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# ロボット設定
robot_config = YaskawaNHC12Config(
    ip_address="192.168.1.31",
    port=10040,
    cameras={"wrist": RealSenseCameraConfig(...)},
    enable_direct_teach=False,  # 自動制御モード
)

# ロボット接続
robot = YaskawaNHC12Robot(robot_config)
robot.connect()

# ACTモデルのロード
policy = make_policy(
    policy_path="<your_hf_username>/<model_name>",
    device="cuda",
)

# 推論ループ
try:
    for _ in range(100):  # 100ステップ実行
        # 観測を取得
        observation = robot.get_observation()

        # ポリシーから動作を予測
        action = policy.select_action(observation)

        # ロボットに動作を送信
        robot.send_action(action)

finally:
    robot.disconnect()
```

### 2. 評価の実行

```bash
python evaluate_act.py
```

### 安全上の注意

- **初回実行**: ロボットの動作範囲に障害物がないことを確認
- **緊急停止**: 非常停止ボタンをすぐに押せる位置に配置
- **監視**: ロボットの動作を常に監視し、異常な動きがあればすぐに停止
- **安全柵**: 可能であれば安全柵を設置

## トラブルシューティング

### 接続エラー

**症状**: `ConnectionError: Could not connect to YASKAWA NHC12`

**解決策**:
1. ロボットコントローラーの電源が入っているか確認
2. ネットワーク接続を確認: `ping 192.168.1.31`
3. IPアドレスとポート番号が正しいか確認
4. ファイアウォールが通信をブロックしていないか確認

### カメラが見つからない

**症状**: カメラが検出されない

**解決策**:
1. RealSenseカメラが正しく接続されているか確認
2. `realsense-viewer` で動作確認
3. USB3.0ポートに接続されているか確認
4. シリアル番号が正しいか確認

### データ収集のFPSが低い

**症状**: 目標FPSに達しない

**解決策**:
1. カメラの解像度を下げる（640x480推奨）
2. 画像書き込みスレッド数を調整: `image_writer_threads=4`
3. 不要なバックグラウンドプロセスを停止
4. より高性能なPCを使用

### トレーニングのロスが下がらない

**症状**: トレーニングロスが収束しない

**解決策**:
1. データセットの品質を確認（失敗したエピソードを削除）
2. より多くのエピソードを収集（最低50エピソード推奨）
3. 学習率を調整: `--training.learning_rate=5e-5`
4. バッチサイズを調整
5. データの多様性を増やす

### ロボットの動きがおかしい

**症状**: モデルが予期しない動作をする

**解決策**:
1. ジョイント制限が正しく設定されているか確認
2. データ収集時とテスト時で同じ開始位置を使用
3. より多くのデータでモデルを再トレーニング
4. カメラの位置が収集時と同じか確認

## 参考資料

- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [YASKAWA NHC12 Manual](https://www.yaskawa.com/) - コントローラーのマニュアルを参照
- [RealSense Documentation](https://github.com/IntelRealSense/librealsense)

## サポート

問題が発生した場合:

1. このREADMEのトラブルシューティングセクションを確認
2. LeRobotの[Issues](https://github.com/huggingface/lerobot/issues)を検索
3. [LeRobot Discord](https://discord.gg/q8Dzzpym3f)で質問

## ライセンス

このコードはApache 2.0ライセンスの下で提供されています。

## 貢献

改善提案やバグ報告は、GitHubのIssueまたはPull Requestで歓迎します。

---

Happy Robot Learning! 🤖✨
