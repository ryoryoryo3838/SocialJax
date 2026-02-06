# メモリ管理ガイド (GPU/CPU)

トレーニング時のGPUおよびCPUのメモリ使用量を管理・最適化するための設定について解説します。

## 1. GPUメモリ (VRAM) の管理

JAXはデフォルトでGPUメモリの90%を一括で確保（Preallocate）しようとします。これが原因で、他のプロセスがメモリを使えなくなったり、見かけ上の使用率が100%近くになったりすることがあります。

### `xla_python_client_mem_fraction` の調整

`scripts/config/train.yaml` にある `xla_python_client_mem_fraction` を設定することで、JAXが確保するメモリの割合を制限できます。

**設定ファイル**: `scripts/config/train.yaml`

```yaml
# デフォルトでは0.5 (50%) に設定されています
xla_python_client_mem_fraction: 0.5
```

- **0.5**: GPUメモリの50%を使用します。
- **0.9**: デフォルトに近い挙動です。
- **0.3**: メモリの30%のみを使用します。VRAM不足エラー（OOM）が出る場合は値を上げる必要がありますが、安全のために下げたい場合はここを調整します。

### 動的なメモリ確保 (Preallocationの無効化)

事前に一括確保するのではなく、必要な分だけ確保していく方式に変更することも可能です。これにより、アイドル時の見かけ上のメモリ使用量を減らすことができます。
これを有効にするには、環境変数を設定して実行します。

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/train.py
```
*(注意: メモリの断片化が起きやすくなるため、長時間の学習では推奨されない場合があります)*

## 2. CPUメモリ (RAM) の管理

CPUメモリの使用量は、主に「並列環境数」と「バッファサイズ」に依存します。

### 並列環境数 (`NUM_ENVS`)

`scripts/config/algorithm/mappo.yaml` の `NUM_ENVS` は、同時に立ち上げる環境のプロセス数（またはインスタンス数）です。

**設定ファイル**: `scripts/config/algorithm/mappo.yaml`

```yaml
NUM_ENVS: 64  # デフォルト
```

- **影響**: 値が大きいほどCPUメモリを消費します。環境が重い（画像処理などを含む）場合、影響は顕著です。
- **対策**: CPUメモリが100%になる場合は、この値を **32** や **16** に減らしてください。
  - ※減らす場合は、データの収集効率が下がるため、`NUM_MINIBATCHES` も合わせて調整（減らす）すると学習が安定しやすくなります。

### ロールアウト長 (`NUM_STEPS`)

1回の更新のために各環境で何ステップ進めるかを決めます。

**設定ファイル**: `scripts/config/algorithm/mappo.yaml`

```yaml
NUM_STEPS: 5
```

- **影響**: `NUM_ENVS * NUM_STEPS * エージェント数 * 観測サイズ` 分のデータをメモリに保持します。
- **対策**: 極端にメモリを圧迫している場合は減らすことも検討できますが、通常は `NUM_ENVS` の方が支配的です。

## 設定変更の例

**安全重視の設定例** (CPU/GPUともに余裕を持たせる):

1. `scripts/config/train.yaml`:
   ```yaml
   xla_python_client_mem_fraction: 0.3  # VRAMを30%に制限
   ```

2. `scripts/config/algorithm/mappo.yaml`:
   ```yaml
   NUM_ENVS: 32      # 並列数を64から32に半減
   NUM_MINIBATCHES: 8 # ミニバッチ数も合わせて調整
   ```

## コマンドラインからの動的変更

設定ファイルを書き換えずに、実行時に一時的に値を変更して試すことも可能です（Hydraの機能）。

```bash
# GPUメモリ30%、並列環境数16で実行
python scripts/train.py xla_python_client_mem_fraction=0.3 algorithm.NUM_ENVS=16 algorithm.NUM_MINIBATCHES=4
```
