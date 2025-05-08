# 人間のような思考プロセスを持つLLMシステム

このプロジェクトは、人間のような思考プロセスを模倣した言語モデルシステムを実装しています。主な特徴として、リアルタイムで思考しながら回答を生成する能力や、複数の脳機能を模した並列処理システム、妹キャラクターによる親しみやすい応答生成機能を備えています。

EN: https://github.com/rintaro-s/lal_v3/blob/main/readme_en.md
メモ：教師データed
## 主な特徴

1. マルチプロセス脳モデル
   - 左脳（論理的思考）と右脳（直感的思考）の並列処理
   - 前頭葉による実行機能と意思決定
   - 海馬モジュールによる記憶の管理
   - 感情処理モジュールによる感情表現

2. 多層記憶システム
   - ワーキングメモリ、短期記憶、長期記憶の階層構造
   - Miller's Law (7±2) に基づく記憶管理
   - 手続き記憶によるパターン認識

3. リアルタイム思考・出力メカニズム
   - 思考完了を待たずに出力開始
   - 不確実性の自然な表現
   - 思考ストリーミング機構

4. 特定分野に特化した学習
   - 高校の勉強内容を重点的に学習
   - 電気電子工学の専門知識
   - IT・プログラミング関連の知識

5. 妹キャラクター対応
   - ユーザーを「お兄ちゃん」と呼ぶ親しみやすい応答
   - 高度な知識を優しく噛み砕いた説明
   - 甘えた口調での自然な会話

## 使用方法

### 環境構築

依存関係のインストール:
```bash
pip install -r requirements.txt
```

PyTorch nightly版使用時の設定:
```bash
# PyTorch nightlyを使用している場合
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

Windows環境でのGPU機能を最大限活用するための依存関係:
```bash
pip install xformers>=0.0.20 accelerate>=0.20.0 bitsandbytes>=0.40.0
```

Windows環境でtritonを実験的に使用 (高度な機能):
```bash
pip install triton-windows>=2.0.0 accelerate>=0.20.0 bitsandbytes>=0.40.0
```

GPU直接アクセスのための設定:
```bash
pip install accelerate>=0.20.0 bitsandbytes>=0.40.0 safetensors>=0.3.0
```

Linux環境では追加でインストール可能:
```bash
pip install triton>=2.0.0 flash-attn>=2.3.0  # Linux環境のみ対応
```

これらのパッケージは特定のGPU（CUDA）環境のみでサポートされています。互換性がない場合は、CPUフォールバックモードを使用してください。

開発モードでのインストール:
```bash
pip install -e .
```

### 知識蒸留の実行

#### 基本的な使用方法

標準の教師モデルを使用:
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B
```

LMstudioのAPIを使用:
```bash
python main.py distill --use_lmstudio --lmstudio_url http://localhost:1234/v1
```

#### 特定分野に特化した学習

高校の勉強と電気電子、IT分野に特化:
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --focus_subjects highschool,electronics,it
```

#### 最適な設定

ELYZA-Thinkingの性能をより継承するための推奨設定:
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --num_examples 5000 --num_epochs 5 --batch_size 2 --gradient_accumulation 16 --max_length 768 --imouto_mode
```

#### リソース節約モード

GPU機能がサポートされていない環境では、より軽量な代替モデルを使用できます:
```bash
python main.py distill --teacher_model elyza/elyza-japanese-llama-2-7b --use_cpu_only
```

#### モデルの保存

Hugging Face形式でのモデル保存（モデルハブにアップロード可能）:
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --save_hf_format --hf_model_name "your-username/lal-brain-model" --imouto_mode
```

### モデル選択ガイド

| モデル名 | サイズ | 特徴 | 必要なGPU RAM | PyTorch nightly対応 |
|---------|--------|------|-------------|-------------|
| elyza/ELYZA-Thinking-1.0-Qwen-32B | 32B | 最高品質・思考能力 | 12GB以上 | 要直接GPU設定 |
| elyza/Llama-3-ELYZA-JP-8B | 8B | 高性能・日本語特化 | 8GB以上 | 完全対応 |
| microsoft/Phi-4-reasoning | 4B | 高効率・推論特化 | 6GB以上 | 完全対応 |
| elyza/ELYZA-japanese-Llama-2-13b-instruct | 13B | 高品質・日本語 | 10GB以上 | 完全対応 |
| elyza/elyza-japanese-llama-2-7b | 7B | バランス・軽量 | 8GB以上 | 完全対応 |
| stabilityai/stablelm-base-alpha-7b | 7B | 軽量・英語 | 8GB以上 | 完全対応 |
| cyberagent/calm2-7b | 7B | 日本語特化・軽量 | 8GB以上 | 完全対応 |

### パラメータ詳細

| パラメータ | 説明 | デフォルト値 |
|------------|------|------------|
| `--num_examples` | 生成する蒸留データ数 | 5000 |
| `--batch_size` | バッチサイズ | 2 |
| `--num_epochs` | エポック数 | 5 |
| `--gradient_accumulation` | 勾配累積ステップ数 | 16 |
| `--quantize` | 教師モデルの4ビット量子化 | 有効 |
| `--cpu_offload` | モデルの一部をCPUにオフロード | 有効 |
| `--max_length` | 最大シーケンス長 | 768 |
| `--save_hf_format` | Hugging Face形式で保存 | 有効 |
| `--hf_model_name` | Hugging Face用のモデル名 | "lal-brain-model" |
| `--use_lmstudio` | LMstudioからAPIで学習データを収集 | 無効 |
| `--lmstudio_url` | LMstudioのAPIエンドポイント | "http://localhost:1234/v1" |
| `--focus_subjects` | 重点的に学習する分野（カンマ区切り） | "highschool,electronics,it" |
| `--imouto_mode` | 妹口調で出力するモードを有効化 | 有効 |

### 既知の問題と解決策

- **tokenizersのバージョンエラー**: `tokenizers>=0.13.3 is required`というエラーが表示される場合は、tokenizersパッケージをアップグレードしてください:
  ```bash
  pip install --upgrade tokenizers>=0.13.3
  ```

- **LMstudio接続エラー**: LMstudio APIに接続できない場合は、LMstudioが正しく起動していることを確認し、URLとポートが正しいことを確認してください:
  ```bash
  # デフォルトのURL確認
  python main.py distill --use_lmstudio --lmstudio_url http://localhost:1234/v1
  ```

- **tritonモジュールエラー (Windows環境)**: `No module named 'triton'`というエラーが表示される場合:
  ```bash
  # 通常のWindows最適化モード (推奨)
  python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --windows_mode
  
  # 実験的：triton-windowsを使用する場合
  pip install triton-windows>=2.0.0
  python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --use_triton_windows
  ```
  または、完全にWindows対応のモデルを使用:
  ```bash
  python main.py distill --teacher_model elyza/elyza-japanese-llama-2-7b
  ```

- **GPUアクセスエラー (PyTorch nightly使用時)**: PyTorch nightlyバージョンでは、一部の最適化ライブラリ(triton, xformers)が利用できません。代わりにGPUへの直接アクセスを設定します:
  ```bash
  # コマンドラインオプション
  python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --use_direct_gpu
  
  # または代替モデル
  python main.py distill --teacher_model elyza/elyza-japanese-llama-2-7b
  ```

- **CUDAエラー**: CUDA関連のエラーが発生する場合は、システムのCUDAバージョンと互換性のあるPyTorchバージョンを使用してください:
  ```bash
  # CUDAに互換性のあるPyTorchをインストール
  pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  ```

- **メモリ不足エラー**: GPUメモリ不足エラーが発生する場合、以下のパラメータを調整してください:
  ```bash
  python main.py distill --batch_size 1 --gradient_accumulation 32 --quantize
  ```

### Hugging Faceモデルのアップロード

学習したモデルをHugging Face Hubにアップロードする方法:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='./models/lal-brain-model', repo_id='YOUR_USERNAME/lal-brain-model')"
```

### 対話モード

通常モード:
```bash
python main.py chat --model_path ./models/brain_model_best.pt
```

妹口調モード (imouto_modeで訓練されたモデル使用):
```bash
python main.py chat --model_path ./models/brain_model_best.pt
```
※妹口調で訓練されたモデルは自動的に妹キャラとして応答します

Hugging Faceからロードする場合:
```bash
python main.py chat --model_path YOUR_USERNAME/lal-brain-model
```

## システム構成

- `brain_model.py`: 脳モデルの実装
- `memory_system.py`: 多層記憶システムの実装
- `real_time_thoughts.py`: リアルタイム思考と出力のメカニズム
- `distillation.py`: 知識蒸留のプロセス
- `inference.py`: 推論エンジン
- `main.py`: コマンドラインインターフェイス
- `imouto_template.py`: 妹キャラ応答の生成テンプレート

## 技術的詳細

### マルチプロセス脳モデル

システムは複数の並列処理モジュールを持ち、それぞれが異なる役割を担います：

- 左脳モジュール：深い分析と論理的思考
- 右脳モジュール：直感的な反応と創造性
- 前頭葉モジュール：実行機能と最終的な意思決定
- 海馬モジュール：記憶の形成と検索
- 感情モジュール：感情状態の追跡と表現

### 思考プロセス

1. 直感的な第一印象（右脳）
2. 探索的思考（記憶からの関連情報検索）
3. 分析的思考（左脳）
4. ひらめき生成（確率的に発生）
5. 結論の形成（前頭葉）

各段階が非同期に進行し、思考の途中経過をストリーミング出力することで、人間のように「考えながら話す」効果を実現しています。

### 蒸留プロセスの最適化

ELYZAモデルからの知識蒸留を最適化するために以下の工夫をしています:

- 勾配累積による効率的なトレーニング
- 4ビット量子化による大規模モデルの効率的な使用
- CPU/GPUメモリの動的割り当てによるリソース最適化
- より長いシーケンス長での学習による知識の完全な継承
- 温度パラメータの最適化による多様性確保

### 特定分野の学習強化

システムは以下の分野に特化した学習が可能です:

- **高校学習**: 数学、物理、化学、生物、地学、国語、英語など高校レベルの学習内容を重点的に学習
- **電気電子**: 回路、半導体、電子部品、電気理論など電気電子工学の基礎から応用まで
- **IT知識**: プログラミング言語、アルゴリズム、データ構造、ウェブ技術など

これらの分野は `--focus_subjects` パラメータで指定できます。

### 妹キャラクター機能

`imouto_mode` を有効にすると、以下の特徴を持つ妹キャラクターとして応答します:

- ユーザーを「お兄ちゃん」と呼ぶ親しみやすい口調
- 学術的・技術的な内容を分かりやすく噛み砕いた説明
- 質問に対して丁寧に回答する優しい性格
- 「〜だよ」「〜だね」「〜かな？」などの柔らかい口調
- 必要に応じて簡単な例え話を使用
