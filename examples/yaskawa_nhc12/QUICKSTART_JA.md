# YASKAWA NHC12 + ACT クイックスタートガイド

このガイドでは、YASKAWA NHC12ロボットでダイレクトティーチを使ってACTモデルをトレーニングする最短手順を説明します。

## 前提条件

- YASKAWA NHC12コントローラーとロボットアーム
- RealSenseカメラ
- Linux PC（Ubuntu 20.04以上）
- Ethernet接続

## 1. 環境構築（5分）

```bash
# LeRobotのインストール
cd /home/kazu/lerobot_fork
pip install -e .
pip install pyrealsense2

# HuggingFace CLIにログイン（データセットのアップロードに必要）
huggingface-cli login
```

## 2. ロボット通信プロトコルの実装（重要）

**この手順が最も重要です。** YASKAWAのマニュアルを参照して、以下のファイルを更新してください:

[src/lerobot/robots/yaskawa_nhc12/robot_yaskawa_nhc12.py](../../src/lerobot/robots/yaskawa_nhc12/robot_yaskawa_nhc12.py)

### 更新が必要なメソッド:

1. **`_send_command()`** - 基本的なTCP/IP通信
2. **`_read_joint_positions()`** - ジョイント位置の読み取り
3. **`_write_joint_positions()`** - ジョイント位置の書き込み
4. **`_enable_direct_teach_mode()`** - ダイレクトティーチモードの有効化
5. **`_disable_direct_teach_mode()`** - ダイレクトティーチモードの無効化

各メソッドには `TODO:` コメントがあり、実装すべき内容が記載されています。

### YASKAWAマニュアルで確認すべき情報:

- TCP/IPコマンドフォーマット
- ジョイント位置の読み取りコマンド（例: `"READ_JOINT_POS\r\n"`）
- ジョイント位置の書き込みコマンド（例: `"MOVE_JOINT J1=10.5,...\r\n"`）
- レスポンスフォーマット
- ダイレクトティーチモードのコマンド

## 3. ネットワーク設定（5分）

### ロボット側:
1. コントローラーの設定画面でIPアドレスを設定
   - IP: `192.168.1.31`
   - サブネット: `255.255.255.0`

### PC側:
```bash
# Ethernetインターフェースを確認
ip link show

# IPアドレスを設定（eth0は環境に応じて変更）
sudo ip addr add 192.168.1.100/24 dev eth0

# 疎通確認
ping 192.168.1.31
```

## 4. データ収集設定（5分）

[record_direct_teach.py](./record_direct_teach.py) を編集:

```python
# 必須設定
ROBOT_IP = "192.168.1.31"  # ロボットのIPアドレス
HF_REPO_ID = "your_username/yaskawa_act_dataset"  # HuggingFaceのユーザー名を設定
TASK_DESCRIPTION = "ペットボトルを掴んで移動"  # タスクの説明

# カメラのシリアル番号を設定（realsense-viewerで確認）
CAMERA_CONFIGS = {
    "wrist": RealSenseCameraConfig(
        serial_number="123456789",  # ← カメラのシリアル番号に更新
        width=640,
        height=480,
        fps=30,
    ),
}

# ロボットのジョイント設定を確認・更新
JOINT_NAMES = ["joint_1", "joint_2", ...]  # ロボットの仕様に合わせる
JOINT_LIMITS = {...}  # ロボットの可動範囲を設定
```

## 5. データ収集の実行（約1-2時間）

```bash
python record_direct_teach.py
```

### データ収集の流れ:

1. スクリプトが起動し、ロボットに接続
2. ダイレクトティーチモードが有効化される
3. Enterキーを押して記録開始
4. **ロボットを手で動かしてタスクを実演**
5. 記録終了後、エピソードを保存（y）または再記録（n）
6. 50エピソード収集（推奨）

### データ収集のコツ:

- **開始位置を統一**: 毎回同じ場所から開始
- **滑らかに動かす**: 急激な動きは避ける
- **カメラの視野を確認**: 対象物が常に映っているか確認
- **多様性を持たせる**: 完全に同じ軌道ではなく、少しずつ変化をつける

## 6. ACTモデルのトレーニング（約4-8時間）

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=your_username/yaskawa_act_dataset \
  --training.num_epochs=1000 \
  --training.batch_size=8 \
  --policy.chunk_size=100 \
  --wandb.enable=true \
  --wandb.project=yaskawa_nhc12_act
```

### トレーニング中:

- Weights & Biasesでトレーニングの進捗を確認
- ロス曲線が下がっているか確認
- 通常、数時間から1日程度かかります

## 7. モデルの評価（30分）

トレーニング完了後、モデルをロボットで実行:

```bash
python evaluate_act.py --policy_path your_username/yaskawa_act_model
```

### 評価の手順:

1. ロボットを開始位置に配置
2. Enterキーを押してエピソード開始
3. ロボットが自動的にタスクを実行
4. 成功/失敗を記録
5. 10エピソード実行して成功率を計算

### 期待される結果:

- **良好**: 成功率 70-90%
- **要改善**: 成功率 < 50% → より多くのデータを収集してリトレーニング

## トラブルシューティング

### ロボットに接続できない

```bash
# 疎通確認
ping 192.168.1.31

# ポートが開いているか確認
telnet 192.168.1.31 10040
```

### カメラが認識されない

```bash
# カメラの確認
realsense-viewer

# シリアル番号の確認
python -c "import pyrealsense2 as rs; ctx = rs.context(); \
    [print(f'Camera {i}: {d.get_info(rs.camera_info.serial_number)}') \
    for i, d in enumerate(ctx.query_devices())]"
```

### データ収集のFPSが低い

- カメラの解像度を下げる（640x480推奨）
- `image_writer_threads` を調整
- 不要なプロセスを停止

## 次のステップ

1. **データの品質向上**: より多様で高品質なデータを収集
2. **ハイパーパラメータ調整**: 学習率、バッチサイズ、チャンクサイズを調整
3. **複数カメラの追加**: より多くの視点でデータを収集
4. **タスクの複雑化**: より複雑なタスクにチャレンジ

## サポート

- [詳細README](./README.md)
- [LeRobot Discord](https://discord.gg/q8Dzzpym3f)
- [LeRobot GitHub Issues](https://github.com/huggingface/lerobot/issues)

---

**重要な注意事項**: ロボットの動作中は常に監視し、非常停止ボタンをすぐに押せる状態にしてください。
