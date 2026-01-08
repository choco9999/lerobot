# SO101 + HIL-SERL（強化学習）手順

このリポジトリの `lerobot.rl.actor` / `lerobot.rl.learner` を使って、SO101実機でHIL-SERL（Human-in-the-Loop + SAC）を動かすための手順です。

## 前提

- SO101 follower（実機） + カメラ（OpenCV）を接続済み
- 介入用に以下のいずれかを用意
  - SO101 leader（推奨）
  - もしくは gamepad / keyboard_ee
- 依存関係の導入（未導入なら）
  - `pip install -e ".[hilserl]"`

## 設定ファイル

テンプレ（SACでゼロからRL）: `src/lerobot/configs/train_config_hilserl_so101.json`  
テンプレ（ACTを初期方策にしてRL微調整）: `src/lerobot/configs/train_config_hilserl_so101_act.json`

最低限、以下を自分の環境に合わせて編集してください。

- `env.robot.port`（例: `/dev/ttyACM0`）
- `env.teleop.port`（例: `/dev/ttyACM1`）
- `env.robot.cameras.*.index_or_path`（カメラ番号 or デバイスパス）
- `env.robot.id` / `env.teleop.id`（キャリブレーションファイルの紐付けに使います）
- `policy.device`（`cuda`/`cpu`/`mps`）
- `policy.dataset_stats.action.min/max`
  - これは「方策が出す [-1,1] のアクション」を「実機が受け取る関節指令値スケール」に戻すために使います。
  - 既に手元で作成した `train_config_record_home_rl.json` 等の `policy.dataset_stats.action.min/max` をコピーするのが簡単です。

## 実行（actor/learner）

### 1) learner を起動（ターミナルA）

```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so101.json
```

### 2) actor を起動（ターミナルB）

```bash
python -m lerobot.rl.actor --config_path src/lerobot/configs/train_config_hilserl_so101.json
```

停止は両方とも `Ctrl+C` です。

## 介入・成功判定（SO101 leader 使用時）

SO101 leader を teleop に指定している場合、以下の2通りで介入できます。

- **自動介入（推奨）**: leader arm を動かすと自動で介入ONになり、一定時間（デフォルト0.5s）動きがなければOFFになります。  
  - `teleop.auto_intervention=true`（デフォルト）で有効  
  - `teleop.auto_intervention_threshold` / `teleop.auto_intervention_hold_s` で感度調整

加えて、`pynput` が使える環境ならキーボードでイベントを送れます（ターミナルにフォーカスが必要です）。

- `space` : 介入ON/OFF（トグル）
- `s` : success（成功扱い）
- `q` / `esc` : terminate（失敗扱い）
- `r` : terminate + rerecord

## バッファ構成（上流 hil-serl に合わせた挙動）

Learner は以下の2種類のリプレイバッファを使い、原則 **50/50** で混ぜて学習します。

- **online buffer**: actor から来る全遷移（方策実行 + 介入を含む）
- **human buffer**: 介入遷移（＋「介入終了直後の1step」も next_state 整合のために保存）  
  - `dataset` を設定している場合は、ここに **デモ遷移** も入ります

`dataset` が `null` でも、介入が入れば human buffer が埋まり、介入の重みを増やして学習します（ただし初期デモがある方が安定します）。

## 出力

- 学習結果・ログは `outputs/train/...` 配下に出ます（実行時刻と `job_name` でフォルダが作られます）。
- **チェックポイント（ckpt）保存は learner 側のみ**です。`outputs/train/.../checkpoints/last/` を確認してください。
  - `output_dir` を `null` のまま起動すると、learner/actor で別フォルダが作られて混乱しやすいので、必要なら両方に同じ `output_dir` を指定してください（例: `--output_dir outputs/train/my_run`）。
  - `save_freq` は **optimization step** 単位です（環境ステップではありません）。早めに ckpt が欲しい場合は `save_freq` を小さくしてください。
  - `save_replay_buffer_dataset=true` にするとリプレイバッファを `<output_dir>/dataset` に書き出しますが、実機（画像あり）だと重くなりがちなので通常は `false` 推奨です。

## 注意（ACTとの関係）

- ACTチェックポイントを初期方策にして、そのままRLで微調整したい場合は `policy.type="sac_act"` を使います。
  - SACのcritic/temperatureは通常通りで、actorだけをACTチェックポイントから初期化して（以降RLで更新）動かします。

### 実機で「でたらめに動く」場合の設定

- `policy.act_deterministic_inference=true` にすると、方策実行時は **ACT（＋RLで更新された重み）の平均行動** をそのまま出すので、ランダム性が減って安定します。
- それでも動きが荒い場合は `policy.act_init_std` を小さくします（例: `0.05` や `0.02`）。

### ACT→RL（sac_act）を使う手順

1) `src/lerobot/configs/train_config_hilserl_so101_act.json` をベースに設定  
2) 最低限以下を編集
- `policy.act_pretrained_path`（ACTの `pretrained_model/` へのパス）
- `policy.dataset_stats.action.min/max`（実機スケールに合わせる）
- `env.robot.port` / `env.teleop.port` / `env.robot.cameras.*.index_or_path`
3) 起動コマンド（learner/actor）は同じ
```bash
python -m lerobot.rl.learner --config_path src/lerobot/configs/train_config_hilserl_so101_act.json
python -m lerobot.rl.actor   --config_path src/lerobot/configs/train_config_hilserl_so101_act.json
```
