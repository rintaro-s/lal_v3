import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer
import numpy as np
import os
import json
import gc
import shutil
import requests
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm import tqdm
import logging
import math
import psutil
from datetime import datetime, timedelta
import sys
import time

# GGUF対応
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

from brain_model import BrainModel

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("distillation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DistillationDataset(Dataset):
    """知識蒸留用のデータセット"""
    def __init__(self, data_path: str, teacher_tokenizer, max_length: int = 512):
        self.tokenizer = teacher_tokenizer
        self.max_length = max_length
        self.data = []
        
        # データを読み込む
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # データを処理
        for item in raw_data:
            if "input" in item and "output" in item:
                self.data.append({
                    "input": item["input"],
                    "output": item["output"]
                })
        
        logger.info(f"Loaded {len(self.data)} examples for distillation")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item["input"]
        output_text = item["output"]
        
        # 入力をトークン化
        input_encodings = self.tokenizer(
            input_text,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 出力をトークン化
        output_encodings = self.tokenizer(
            output_text,
            max_length=self.max_length // 2,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 入力と出力を連結
        input_ids = torch.cat([input_encodings.input_ids[0], output_encodings.input_ids[0]])
        attention_mask = torch.cat([input_encodings.attention_mask[0], output_encodings.attention_mask[0]])
        
        # 入力部分はラベル-100（無視）、出力部分は実際のラベル
        labels = torch.cat([
            torch.ones_like(input_encodings.input_ids[0]) * -100,
            output_encodings.input_ids[0]
        ])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

class KnowledgeDistiller:
    """知識蒸留システム"""
    def __init__(
        self,
        teacher_model_name: Optional[str],
        student_model: BrainModel,
        tokenizer,
        device: torch.device,
        config: Dict,
        quantize: bool = True,
        cpu_offload: bool = True,
        use_cpu_only: bool = False
    ):
        self.logger = logging.getLogger(__name__)
        self.teacher_model_name = teacher_model_name
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.use_lmstudio = config.get("use_lmstudio", False)
        self.lmstudio_url = config.get("lmstudio_url", "http://localhost:1234/v1")
        self.lmstudio_model = config.get("lmstudio_model", None)
        self.structured_output = config.get("structured_output", False)
        self.windows_mode = config.get("windows_mode", False)
        self.use_cpu_only = config.get("use_cpu_only", False) or use_cpu_only
        self.use_direct_gpu = config.get("use_direct_gpu", False)
        self.use_triton_windows = config.get("use_triton_windows", False)
        self.focus_subjects = config.get("focus_subjects", ["highschool", "electronics", "it"])
        self.imouto_mode = config.get("imouto_mode", True)
        self.use_gguf = config.get('use_gguf', False)
        self.gguf_model_path = config.get('gguf_model_path', None)
        self.gguf_context_length = config.get('gguf_context_length', 4096)
        self.thinking_llm = config.get('thinking_llm', False)
        self.llama_cpp_model = None  # GGUF用モデル
        self.teacher_model = None  # 教師モデル初期化を追加
        
        # データキャッシュ
        self.data_cache = {}
        self.max_cache_size = 10000  # 最大キャッシュサイズ
        
        # RAM使用状況を記録
        ram_usage = psutil.virtual_memory()
        self.logger.info(f"RAM使用状況: {ram_usage.percent}% (使用中: {ram_usage.used/1024**3:.1f}GB, 空き: {ram_usage.available/1024**3:.1f}GB)")
        
        # 教師モデルのテンプレート（妹口調または通常）
        self.thinking_template = """質問: {question}

回答:
"""
        # 妹口調版のプロンプトテンプレート
        self.imouto_template = """質問: {question}

回答（妹口調で、お兄ちゃんと呼んで）:
"""
        
        # LMstudioを使用する場合
        if self.use_lmstudio:
            self.logger.info(f"Using LMstudio API at {self.lmstudio_url}")
            try:
                # API接続確認
                response = requests.get(f"{self.lmstudio_url}/v1/models")  # 修正: /models -> /v1/models
                if response.status_code == 200:
                    models = response.json()
                    if self.lmstudio_model:
                        self.logger.info(f"Using LMstudio model: {self.lmstudio_model}")
                    else:
                        self.logger.info(f"Available LMstudio models: {', '.join([m.get('id', 'unknown') for m in models])}")
                    self.teacher_model = None  # APIを使うので実際のモデルはロードしない
                else:
                    raise ConnectionError(f"Failed to connect to LMstudio API: {response.status_code}")
            except Exception as e:
                self.logger.error(f"LMstudio API connection error: {e}")
                raise ConnectionError(f"LMstudio API connection error: {e}")
            return
        
        if self.use_gguf:
            if not LLAMA_CPP_AVAILABLE:
                raise ImportError("GGUFモデルを使用するには llama-cpp-python パッケージが必要です。"
                                 "pip install llama-cpp-python または "
                                 "pip install llama-cpp-python --upgrade をお試しください。")
            
            self.logger.info(f"Loading GGUF model: {self.gguf_model_path}")
            try:
                # GPU設定を取得
                use_gpu = config.get('gguf_use_gpu', False) 
                n_gpu_layers = config.get('gguf_n_gpu_layers', -1)
                n_gpu = config.get('gguf_n_gpu', 1)
                n_batch = config.get('gguf_n_batch', 512)
                
                # GPUパラメータのログ出力
                self.logger.info(f"GGUF GPU設定: use_gpu={use_gpu}, n_gpu_layers={n_gpu_layers}, n_gpu={n_gpu}, n_batch={n_batch}")
                
                if use_gpu and torch.cuda.is_available():
                    # GPUが利用可能な場合
                    gpu_name = torch.cuda.get_device_name(0)
                    self.logger.info(f"GGUF: GPU使用を試みます: {gpu_name}")
                    
                    # CUDAの情報を詳細に出力
                    self.logger.info(f"CUDA バージョン: {torch.version.cuda}")
                    self.logger.info(f"cuDNN バージョン: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
                    self.logger.info(f"CUDA デバイス数: {torch.cuda.device_count()}")
                    
                    # llama-cpp-pythonのGPUモードオプション
                    # CPUモードの設定
                    model_options = {
                        "model_path": self.gguf_model_path,
                        "n_ctx": self.gguf_context_length,
                        "n_threads": os.cpu_count() or 4,
                        "verbose": True  # デバッグ用に詳細なログを有効化
                    }
                    
                    # GPUモードの追加設定
                    gpu_options = {
                        "n_gpu_layers": n_gpu_layers,
                        "n_batch": n_batch,
                        "offload_kqv": False,  # KQV操作をGPUにオフロード
                    }
                    
                    # 新しいバージョンのLlama-cpp-pythonでは'n_gpu'ではなく'main_gpu'を使用
                    # バージョンに合わせて適切なパラメータを使用
                    try:
                        from llama_cpp import __version__ as llama_cpp_version
                        version_parts = llama_cpp_version.split('.')
                        if int(version_parts[0]) > 0 or int(version_parts[1]) >= 2:  # 0.2.0以上
                            self.logger.info(f"llama-cpp-python バージョン {llama_cpp_version} を検出")
                            if hasattr(Llama, '__init__') and 'main_gpu' in Llama.__init__.__code__.co_varnames:
                                gpu_options["main_gpu"] = 0
                            else:
                                gpu_options["n_gpu"] = n_gpu
                        else:
                            gpu_options["n_gpu"] = n_gpu
                    except (ImportError, AttributeError, ValueError):
                        # バージョン情報を取得できない場合は従来のパラメータを使用
                        gpu_options["n_gpu"] = n_gpu
                        
                    try:
                        # まずGPUモードでの初期化を試行
                        self.logger.info(f"GPUモードでGGUFモデルを初期化します: {gpu_options}")
                        self.llama_cpp_model = Llama(**model_options, **gpu_options)
                        
                        # テスト推論を実行してGPU使用を確認
                        self.logger.info("GPUモード: テスト推論を実行...")
                        test_prompt = "これはテストです。"
                        start = datetime.now()
                        test_result = self.llama_cpp_model(test_prompt, max_tokens=10)
                        elapsed = (datetime.now() - start).total_seconds()
                        self.logger.info(f"テスト推論完了: 処理時間 {elapsed:.2f}秒")
                        
                        # GPUメモリ使用状況を確認
                        if torch.cuda.is_available():
                            mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                            mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                            self.logger.info(f"GPUメモリ: 割り当て {mem_allocated:.2f}GB, 予約 {mem_reserved:.2f}GB")
                            
                        self.logger.info("GGUF: GPUモードで正常に初期化されました")
                        
                    except Exception as gpu_error:
                        # GPUモードが失敗した場合の詳細なエラー情報
                        self.logger.error(f"GPUモード初期化エラー: {str(gpu_error)}")
                        self.logger.error(f"エラータイプ: {type(gpu_error).__name__}")
                        import traceback
                        self.logger.error(f"スタックトレース: {traceback.format_exc()}")
                        
                        # 'AARCH64'エラーの可能性があるため、特定のGPUオプションを調整して再試行
                        if "AARCH64" in str(gpu_error) or "buffer type" in str(gpu_error):
                            self.logger.warning("AARCH64バッファタイプの問題を検出。異なるGPUオプションで再試行します...")
                            try:
                                # 一部の設定を変更して再試行
                                gpu_options["offload_kqv"] = False  # KQVオフロードを無効化
                                gpu_options["n_batch"] = 128  # バッチサイズを縮小
                                
                                self.logger.info(f"調整後のGPUオプション: {gpu_options}")
                                self.llama_cpp_model = Llama(**model_options, **gpu_options)
                                
                                # 再度テスト
                                test_result = self.llama_cpp_model("テスト", max_tokens=5)
                                self.logger.info("調整後のGPUモードで正常に初期化されました")
                                
                            except Exception as retry_error:
                                # それでも失敗した場合はCPUモードにフォールバック
                                self.logger.error(f"調整後のGPU初期化も失敗: {str(retry_error)}")
                                self.logger.warning("CPUモードにフォールバックします")
                                self.llama_cpp_model = Llama(**model_options)
                        else:
                            # その他のエラーはCPUモードにフォールバック
                            self.logger.warning("CPUモードにフォールバックします")
                            self.llama_cpp_model = Llama(**model_options)
                else:
                    # CPU専用モード
                    self.logger.info("GGUF: CPUモードで実行します")
                    self.llama_cpp_model = Llama(
                        model_path=self.gguf_model_path,
                        n_ctx=self.gguf_context_length,
                        n_threads=os.cpu_count() or 4
                    )
                
                # モデル情報をログに記録
                self.logger.info("GGUF model loaded successfully")
                if hasattr(self.llama_cpp_model, "model_path"):
                    self.logger.info(f"GGUF モデルパス: {self.llama_cpp_model.model_path}")
                if hasattr(self.llama_cpp_model, "n_ctx"):
                    self.logger.info(f"コンテキスト長: {self.llama_cpp_model.n_ctx}")
                    
                # GPUレイヤー数を確認
                gpu_layers_info = getattr(self.llama_cpp_model, "n_gpu_layers", 0)
                if gpu_layers_info > 0:
                    self.logger.info(f"GPU上のレイヤー数: {gpu_layers_info}")
                    self.logger.info("GGUF: GPUモードで実行中")
                else:
                    self.logger.info("GGUF: CPUモードで実行中")
                
                # GGUFモデル使用時はtransformersのtokenizerを使用しない
                self.use_transformers_tokenizer = False
                
            except Exception as e:
                self.logger.error(f"Failed to load GGUF model: {e}")
                import traceback
                self.logger.error(f"詳細なエラー: {traceback.format_exc()}")
                raise
        else:
            self.use_transformers_tokenizer = True
            self.logger.info(f"Loading teacher model: {teacher_model_name}")
            
            # 教師モデルをロード
            try:
                # 量子化設定
                quantization_config = None
                if quantize:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True
                    )
                
                # モデル関連の設定を取得
                load_in_8bit = config.get('load_in_8bit', False)
                load_in_4bit = quantize and not load_in_8bit
                trust_remote_code = config.get('trust_remote_code', False)
                use_flash_attention = config.get('use_flash_attention', False)
                
                # GPUオフロード設定
                device_map = "auto" if cpu_offload and not use_cpu_only else None
                
                # CPU専用モードの場合
                if use_cpu_only:
                    device_map = {"": "cpu"}
                    quantization_config = None
                    load_in_8bit = False
                    load_in_4bit = False
                
                # 直接GPUモード（8ビット/4ビット量子化なしで高速化）
                if self.use_direct_gpu:
                    device_map = {"": 0}
                    quantization_config = None
                    load_in_8bit = False
                    load_in_4bit = False
                
                # モデルパラメータ
                model_kwargs = {
                    "pretrained_model_name_or_path": teacher_model_name,
                    "device_map": device_map,
                    "load_in_8bit": load_in_8bit,
                    "load_in_4bit": load_in_4bit,
                    "quantization_config": quantization_config if load_in_4bit else None,
                    "torch_dtype": torch.float16 if not use_cpu_only else torch.float32,
                    "trust_remote_code": trust_remote_code
                }
                
                # Flash Attentionが利用可能なら使用
                if use_flash_attention:
                    model_kwargs["use_flash_attention_2"] = True
                    self.logger.info("Using Flash Attention 2 for teacher model")
                
                # Windowsでのtritonサポートオプション
                if self.windows_mode and self.use_triton_windows:
                    self.logger.info("Enabling triton support for Windows")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    os.environ["TRITON_BACKENDS_PATH"] = r"C:\Triton\backends"
                
                self.logger.info(f"Loading teacher model with parameters: {model_kwargs}")
                self.teacher_model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                
                # 通常のGPUモード
                if not cpu_offload and not use_cpu_only and not self.use_direct_gpu:
                    self.teacher_model = self.teacher_model.to(self.device)
                
                self.logger.info("Teacher model loaded successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load teacher model: {e}")
                import traceback
                self.logger.error(f"詳細なエラー: {traceback.format_exc()}")
                raise

    def filter_by_subjects(self, questions: List[str], subjects: List[str]) -> List[str]:
        """特定の分野に関連する質問をフィルタリング"""
        # 分野ごとのキーワード
        subject_keywords = {
            "highschool": ["高校", "学校", "勉強", "数学", "国語", "英語", "理科", "社会", "歴史", "地理", 
                        "物理", "化学", "生物", "地学", "現代社会", "倫理", "高校数学", "受験"],
            "electronics": ["電気", "電子", "回路", "半導体", "抵抗", "コンデンサ", "インダクタ", "トランジスタ", 
                          "ダイオード", "CPU", "マイコン", "電圧", "電流", "電力", "オームの法則", "キルヒホッフ"],
            "it": ["プログラミング", "コンピュータ", "アルゴリズム", "データ構造", "ウェブ", "インターネット",
                 "セキュリティ", "ネットワーク", "クラウド", "サーバー", "データベース", "Arduino", "HTML", "マイコン",
                 "JavaScript", "Python", "Java", "C++", "C#", "AI", "人工知能", "機械学習", "深層学習"]
        }
        
        # 指定された分野に関連するキーワードのリストを作成
        target_keywords = []
        for subject in subjects:
            if subject in subject_keywords:
                target_keywords.extend(subject_keywords[subject])
        
        # キーワードを含む質問をフィルタリング
        filtered_questions = []
        for question in questions:
            if any(keyword in question for keyword in target_keywords):
                filtered_questions.append(question)
        
        # フィルタリング後、質問が少なすぎる場合はランダムに追加
        if len(filtered_questions) < 0.5 * len(questions):
            remaining = [q for q in questions if q not in filtered_questions]
            additional = random.sample(remaining, min(len(remaining), len(questions) - len(filtered_questions)))
            filtered_questions.extend(additional)
        
        self.logger.info(f"Filtered questions: {len(filtered_questions)} out of {len(questions)} original questions")
        return filtered_questions

    def generate_questions_if_needed(self, questions_file: str, num_samples: int) -> str:
        """質問ファイルが存在しない場合、自動生成する"""
        self.logger.info("自動的に質問データを生成します")
        
        try:
            # 質問リストの基本テーマ
            question_themes = [
                # 高校教科: 数学を強化
                "数学の微分・積分の基本概念",
                "数学の極限の考え方",
                "数学における三角関数の応用",
                "数列の漸化式と一般項",
                "ベクトルの内積と外積の違い",
                "確率統計の基本定理",
                "複素数平面の図形的意味",
                "二次方程式と判別式の関係",
                "指数関数と対数関数の性質",
                "空間座標系と立体図形",
                "行列の固有値と固有ベクトル",
                
                #高校物理
                "力学における運動の法則",
                "エネルギー保存の法則",
                "波動の性質と干渉",
                "電磁気学におけるクーロンの法則",
                "熱力学の基本法則",
                "光の屈折と全反射",
                "原子核の構造と放射線",
                "電流と磁場の関係",
                "運動量保存の法則",
                "振動と波の関係",


                # その他の高校教科
                "英語の関係代名詞の使い方",
                "日本史における江戸時代の特徴",
                "化学の酸化と還元反応",
                "物理学における運動方程式",
                "古典文学「源氏物語」の魅力",
                "世界史における産業革命の影響",
                "生物の細胞分裂の過程",
                "地理学における気候帯の分類",
                "現代社会における憲法の役割",
                
                # 電子工学
                "トランジスタの基本的な仕組み",
                "オームの法則",
                "電子回路の並列接続と直列接続の違い",
                "半導体の種類と特性",
                "アナログ回路とデジタル回路の違い",
                "マイクロコントローラの役割と機能",
                "電圧と電流の関係性",
                "ダイオードの整流作用",
                "論理ゲートの種類と機能",
                "電磁誘導の原理",
                
                # IT・プログラミング
                "プログラミング言語Pythonの特徴",
                "データベースのSQLクエリの基本構文",
                "オブジェクト指向プログラミングの概念",
                "人工知能の機械学習とディープラーニングの違い",
                "ウェブセキュリティにおけるXSSとは",
                "クラウドコンピューティングのメリット",
                "アルゴリズムの計算量とビッグO記法",
                "ネットワークのTCP/IPプロトコル",
                "Gitのような分散型バージョン管理システムの利点",
                "APIの役割と活用方法"
            ]
            
            self.logger.info(f"基本的な質問テーマ数: {len(question_themes)}")
            
            # 質問の生成パターン (自然な質問文になるように設計)
            question_patterns = [
                "{}について教えてください。",
                "{}を詳しく説明してもらえますか？",
                "{}とは何ですか？",
                "{}の基本概念を教えてください。",
                "{}について知りたいです。",
                "{}の仕組みはどうなっていますか？"
            ]
            
            # 妹口調用の追加パターン
            imouto_patterns = [
                "{}について教えて欲しいな、お兄ちゃん。",
                "お兄ちゃん、{}ってどういうこと？",
                "{}のこと、わかりやすく説明してほしいな。",
                "お兄ちゃん、{}について教えてくれる？",
                "{}って難しそうだけど、お兄ちゃんなら分かるよね？"
            ]
            
            self.logger.info(f"質問生成パターン数: {len(question_patterns)} (通常) + {len(imouto_patterns)} (妹口調)")
            
            # 質問を生成
            questions = []
            self.logger.info("質問の生成を開始します...")
            
            # 進捗状況の表示準備
            total_themes = len(question_themes)
            
            # 通常の質問を生成
            for i, theme in enumerate(question_themes):
                # コンソール上に進捗バーを表示
                progress = (i + 1) / total_themes
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\r生成中: [{bar}] {progress*100:.1f}% ({i+1}/{total_themes})")
                sys.stdout.flush()
                
                # このテーマに対して複数のパターンを適用
                theme_questions = []
                for pattern in question_patterns:
                    theme_questions.append(pattern.format(theme))
                
                # ランダムに最大2つを選んで追加
                selected = random.sample(theme_questions, min(2, len(theme_questions)))
                questions.extend(selected)
                
                # 妹口調テーマも追加
                if self.imouto_mode and random.random() < 0.3:  # 30%の確率で妹口調の質問も追加
                    imouto_pattern = random.choice(imouto_patterns)
                    questions.append(imouto_pattern.format(theme))
                
                if len(questions) >= num_samples:
                    break
            
            print()  # 進捗バーの後に改行
            
            # 必要数に達しない場合、テーマをバリエーション
            if len(questions) < num_samples:
                self.logger.info(f"追加の質問を生成します。現在: {len(questions)}/{num_samples}")
                
                # テーマのバリエーションを作成
                variations = []
                for theme in question_themes:
                    # テーマの応用的なバリエーション
                    variations.append(f"{theme}の応用例")
                    variations.append(f"{theme}の歴史")
                    variations.append(f"{theme}の重要性")
                    variations.append(f"{theme}の最新動向")
                    variations.append(f"{theme}の学習方法")
                
                # バリエーションからランダムに選択して質問を生成
                random.shuffle(variations)
                for theme in variations:
                    pattern = random.choice(question_patterns)
                    new_q = pattern.format(theme)
                    
                    if new_q not in questions:
                        questions.append(new_q)
                    
                    if len(questions) >= num_samples:
                        break
            
            # 質問をランダムに並び替え
            random.shuffle(questions)
            
            # 質問数を最終確認
            self.logger.info(f"生成された質問数: {len(questions)}")
            if len(questions) < num_samples:
                self.logger.warning(f"要求された {num_samples} 個の質問を生成できませんでしたが、{len(questions)}個生成しました")
            
            # 質問をファイルに保存
            output_path = questions_file
            dir_path = os.path.dirname(output_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            # ファイル書き込みを処理
            self.logger.info(f"質問をファイルに書き込み中: {output_path}")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for q in questions[:min(len(questions), num_samples)]:
                        f.write(q + '\n')
                self.logger.info(f"質問ファイルの書き込みが完了しました: {output_path}")
            except Exception as e:
                self.logger.error(f"ファイル書き込み中にエラーが発生: {e}")
                # 緊急用の代替ファイルに書き込み
                fallback_path = "questions_emergency.txt"
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    for q in questions[:min(len(questions), num_samples)]:
                        f.write(q + '\n')
                self.logger.info(f"代替ファイルに保存しました: {fallback_path}")
                output_path = fallback_path
                
            return output_path
            
        except Exception as e:
            self.logger.error(f"質問生成中に予期しないエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # エラー回復: 最低限の質問セットを生成して返す
            emergency_questions = [
                "数学について教えてください",
                "英語学習のコツは何ですか",
                "プログラミングの基礎を説明してください",
                "電子工学とは何ですか",
                "人工知能の仕組みを教えてください"
            ]
            
            # 緊急用ファイルに書き込み
            emergency_path = "questions_error_recovery.txt"
            with open(emergency_path, 'w', encoding='utf-8') as f:
                for q in emergency_questions:
                    f.write(q + '\n')
            
            self.logger.info(f"エラー回復: 緊急質問ファイルを生成しました: {emergency_path}")
            return emergency_path

    def prepare_distillation_data(self, questions_file: str, output_file: str, num_samples: int = 100, 
                                batch_size: int = 4, cache_to_ram: bool = True, focus_subjects: List[str] = None,
                                imouto_mode: bool = True, thinking_llm: bool = False):
        """教師モデルからの出力を生成してデータを準備（バッチ処理対応）"""
        self.logger.info(f"Preparing distillation data from {questions_file}")
        
        # 質問ファイルが存在しない場合は自動生成
        if not os.path.exists(questions_file):
            self.logger.info(f"Questions file {questions_file} not found, generating automatically")
            questions_file = self.generate_questions_if_needed(questions_file, num_samples)
        
        # 質問を読み込む
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]
        
        # 重点分野にフィルタリング
        if focus_subjects:
            questions = self.filter_by_subjects(questions, focus_subjects)
        
        # サンプル数を制限
        if num_samples < len(questions):
            questions = random.sample(questions, num_samples)
        
        distillation_data = []
        
        # バッチ処理で効率化
        total_batches = math.ceil(len(questions) / batch_size)
        start_time = time.time()
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            current_batch = i // batch_size + 1
            
            # 進捗状況の表示
            progress = current_batch / total_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            
            # 経過時間と推定残り時間を計算
            elapsed = time.time() - start_time
            if current_batch > 1:  # 1バッチ目は推定に使えない
                estimated_total = elapsed / (current_batch - 1) * total_batches
                remaining = max(0, estimated_total - elapsed)
                remaining_str = str(timedelta(seconds=int(remaining)))
                elapsed_str = str(timedelta(seconds=int(elapsed)))
            else:
                remaining_str = "計算中..."
                elapsed_str = "0:00:00"
                
            sys.stdout.write(f"\r進捗: [{bar}] {progress*100:.1f}% - バッチ {current_batch}/{total_batches} - 経過: {elapsed_str} - 残り: {remaining_str}")
            sys.stdout.flush()
            
            self.logger.info(f"Processing batch {current_batch}/{total_batches}")
            
            # 妹モードの場合はプロンプトテンプレートを変更
            template = self.imouto_template if imouto_mode else self.thinking_template
            batch_prompts = [template.format(question=q) for q in batch_questions]
            
            if self.use_gguf:
                for j, prompt in enumerate(batch_prompts):
                    try:
                        self.logger.info(f"Generating GGUF response for question {i+j+1}/{len(questions)}")
                        response = self._generate_gguf_response(prompt)
                        if not response:
                            continue
                        
                        if thinking_llm:
                            thoughts, answer = self._extract_thinking_answer(response)
                            distillation_data.append({
                                "input": prompt,
                                "thoughts": thoughts,
                                "output": answer
                            })
                        else:
                            distillation_data.append({
                                "input": prompt,
                                "output": response
                            })
                    except Exception as e:
                        self.logger.error(f"Error generating response for prompt: {e}")
                        continue
            elif self.use_lmstudio:
                import json, requests
                endpoint = f"{self.lmstudio_url}/v1/chat/completions"
                headers = {"Content-Type": "application/json"}
                
                for prompt in batch_prompts:
                    payload = {
                        "model": self.lmstudio_model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "type": "object",
                                "properties": {
                                    "response": {"type": "string"}
                                },
                                "required": ["response"]
                            }
                        },
                        "temperature": 0.7,
                        "max_tokens": 150,
                        "stream": False
                    }
                    try:
                        response = requests.post(endpoint, headers=headers, json=payload)
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                # レスポンスが期待される形式か確認
                                if not isinstance(data, dict) or "choices" not in data:
                                    self.logger.error(f"Unexpected LMstudio API response format: {data}")
                                    distillation_data.append({
                                        "input": prompt,
                                        "output": ""
                                    })
                                    continue
                                # 正常なレスポンスからテキストを抽出
                                output_text = json.loads(data["choices"][0]["message"]["content"])["response"]
                                distillation_data.append({
                                    "input": prompt,
                                    "output": output_text
                                })
                            except (json.JSONDecodeError, KeyError) as e:
                                self.logger.error(f"Failed to parse JSON response: {e}")
                                distillation_data.append({
                                    "input": prompt,
                                    "output": ""
                                })
                        else:
                            self.logger.error(f"LMstudio API error: {response.status_code}, {response.text}")
                            distillation_data.append({
                                "input": prompt,
                                "output": ""
                            })
                    except Exception as e:
                        self.logger.error(f"Error calling LMstudio API: {e}")
                        distillation_data.append({
                            "input": prompt,
                            "output": ""
                        })
            else:
                # 通常の教師モデルを使用する場合
                # 教師モデルのチェック
                if self.teacher_model is None:
                    self.logger.error("Teacher model is not initialized")
                    raise ValueError("Teacher model is not initialized. Please check your configuration.")
                
                try:
                    batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
                    
                    with torch.no_grad():
                        # 教師モデルから出力を生成
                        outputs = self.teacher_model.generate(
                            input_ids=batch_inputs.input_ids,
                            attention_mask=batch_inputs.attention_mask,
                            max_new_tokens=512,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        # 出力をデコード
                        for j, output in enumerate(outputs):
                            self.logger.info(f"Processing output {j+1}/{len(outputs)} in batch {i//batch_size + 1}")
                            teacher_output = self.tokenizer.decode(output, skip_special_tokens=True)
                            
                            # 入力プロンプトと出力から質問部分を取得
                            original_question = batch_questions[j % len(batch_questions)]
                            
                            distillation_data.append({
                                "input": original_question,
                                "output": teacher_output
                            })
                    
                    # バッチ処理後にメモリを解放
                    del batch_inputs, outputs
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    self.logger.error(f"Error generating teacher model outputs: {e}")
                    import traceback
                    self.logger.error(f"詳細なエラー: {traceback.format_exc()}")
                    # エラーが発生しても次のバッチに進む
                    continue
            
            # 定期的に進捗を保存（万が一のクラッシュに備える）
            if (i // batch_size) % 10 == 0 and i > 0:
                temp_file = f"{output_file}.temp_{i}"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(distillation_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Saved progress to temporary file: {temp_file}")
        
        print()  # 進捗バー後の改行
        
        # 結果を保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(distillation_data, f, ensure_ascii=False, indent=2)
        
        # 完了メッセージと合計時間
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        self.logger.info(f"Distillation data saved to {output_file} with {len(distillation_data)} examples")
        self.logger.info(f"Total processing time: {total_time_str}")
        
        return output_file

    def _generate_gguf_response(self, prompt: str) -> str:
        """GGUF（llama-cpp-python）モデルから応答を生成"""
        if not self.llama_cpp_model:
            raise ValueError("GGUF model is not initialized")
        
        try:
            # より詳細なログ出力
            self.logger.debug(f"GGUF生成: プロンプト長 {len(prompt)} 文字")
            start_time = datetime.now()
            
            # llama-cpp-pythonは直接テキストを処理できるので、tokenizer不要
            response = self.llama_cpp_model(
                prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                stop=["質問:", "回答:", "\n\n質問:"],
                echo=False
            )
            
            # 生成時間を計測
            generation_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"GGUF生成完了: 処理時間 {generation_time:.2f}秒")
            
            if "choices" in response and len(response["choices"]) > 0:
                return response["choices"][0]["text"].strip()
            return ""
        except Exception as e:
            self.logger.error(f"Error in GGUF response generation: {e}")
            return ""

    def _extract_thinking_answer(self, response: str) -> tuple:
        """思考過程と回答を分離する"""
        # 「回答:」で分割を試みる
        parts = response.split("回答:")
        
        # 明示的な分割がある場合
        if len(parts) >= 2:
            thoughts = parts[0].strip()
            answer = "回答:" + "".join(parts[1:]).strip()
            return thoughts, answer
            
        # 「思考過程:」で始まる場合は、段落で分割を試みる
        if response.startswith("思考過程:") or "思考過程:" in response:
            paragraphs = response.split("\n\n")
            if len(paragraphs) >= 2:
                # 最初の段落を思考過程とみなす
                thoughts = paragraphs[0].strip()
                # 残りを回答とみなす
                answer = "\n\n".join(paragraphs[1:]).strip()
                return thoughts, answer
                
        # 明確な区切りがない場合は、中間点で分割
        mid_point = len(response) // 2
        thoughts = response[:mid_point].strip()
        answer = response[mid_point:].strip()
        
        return thoughts, answer

    def distill(self, train_data_path, val_data_path, output_dir, batch_size=4, num_epochs=3,
            gradient_accumulation_steps=8, use_ram_cache=True, checkpoint_every=500, config=None):
        """知識蒸留を実行"""
        self.logger.info("Starting distillation process")
        
        # トレーニングデータと検証データの読み込み
        train_dataset = DistillationDataset(train_data_path, self.tokenizer, max_length=config.get("max_length", 512))
        val_dataset = DistillationDataset(val_data_path, self.tokenizer, max_length=config.get("max_length", 512))
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 最適化アルゴリズム
        optimizer = optim.AdamW(self.student_model.parameters(), lr=config.get("learning_rate", 5e-5), 
                              weight_decay=config.get("weight_decay", 0.01))
        
        # 学習率スケジューラ
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = config.get("warmup_steps", 500)
        scheduler = None
        
        # モデルをデバイスに移動
        self.student_model.to(self.device)
        
        # ベストモデルの追跡
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        best_model_path = os.path.join(output_dir, "brain_model_best.pt")
        
        # 学習ループ
        self.logger.info(f"Training for {num_epochs} epochs, {len(train_dataloader)} steps per epoch")
        start_time = time.time()
        global_step = 0
        
        for epoch in range(num_epochs):
            # トレーニングフェーズ
            self.student_model.train()
            epoch_loss = 0
            
            # 進捗バー付きの学習ループ
            epoch_start_time = time.time()
            train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                                     leave=True, position=0)
            
            for step, batch in enumerate(train_progress_bar):
                # データをGPUに移動
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # フォワードパス
                outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # 勾配累積の処理
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                
                # ロスの記録
                epoch_loss += loss.item() * gradient_accumulation_steps
                global_step += 1
                
                # 進捗バーの更新
                train_progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 
                    'avg_loss': f'{epoch_loss / (step + 1):.4f}'
                })
                
                # 経過時間と推定残り時間を計算
                if step > 0:
                    step_time = (time.time() - epoch_start_time) / step
                    steps_remaining = len(train_dataloader) - step
                    time_remaining = steps_remaining * step_time
                    train_progress_bar.set_description(
                        f"Epoch {epoch+1}/{num_epochs} - "
                        f"残り時間: {str(timedelta(seconds=int(time_remaining)))}"
                    )
                
                # チェックポイントの保存
                if global_step % checkpoint_every == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}.pt")
                    torch.save({
                        'global_step': global_step,
                        'epoch': epoch,
                        'model_state_dict': self.student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'loss': loss.item(),
                    }, checkpoint_path)
                    self.logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # トレーニングエポックの平均損失
            avg_train_loss = epoch_loss / len(train_dataloader)
            self.logger.info(f"Epoch {epoch+1} - Average training loss: {avg_train_loss:.4f}")
            
            # 検証フェーズ
            self.student_model.eval()
            val_loss = 0
            
            # 進捗バー付きの検証ループ
            val_progress_bar = tqdm(val_dataloader, desc=f"Validation", leave=True, position=0)
            
            with torch.no_grad():
                for batch in val_progress_bar:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    
                    val_loss += loss.item()
                    
                    val_progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
            # 検証セットの平均損失
            val_loss = val_loss / len(val_dataloader)
            self.logger.info(f"Epoch {epoch+1} - Validation loss: {val_loss:.4f}")
            
            # ベストモデル保存
            if val_loss < best_val_loss:
                self.logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
                best_val_loss = val_loss
                best_model_state = self.student_model.state_dict().copy()
                best_epoch = epoch + 1
                
                # モデルの保存
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.student_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': val_loss,
                    'model_config': config,  # 設定情報を保存
                    'imouto_mode': self.imouto_mode,  # 妹モードの設定を保存
                    'focus_subjects': self.focus_subjects,  # 重点分野の設定を保存
                }, best_model_path)
                
                self.logger.info(f"Saved best model to {best_model_path}")
            
            # エポックごとのモデル保存
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'model_config': config,  # 設定情報を保存
                'imouto_mode': self.imouto_mode,  # 妹モードの設定を保存
                'focus_subjects': self.focus_subjects,  # 重点分野の設定を保存
            }, os.path.join(output_dir, f'brain_model_epoch_{epoch+1}.pt'))
            
            # エポックごとの進捗サマリー
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} 完了 - "
                           f"経過時間: {str(timedelta(seconds=int(elapsed_time)))} - "
                           f"推定残り時間: {str(timedelta(seconds=int(estimated_remaining)))}")
        
        # 訓練終了、最終的なベストモデルの状態を保存
        if best_model_state is not None:
            # ベストモデルの保存
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'val_loss': best_val_loss,
                'model_config': config,
                'imouto_mode': self.imouto_mode,
                'focus_subjects': self.focus_subjects,
                'thinking_llm': self.thinking_llm,  # 考えてから出力するタイプのLLMの設定を保存
                'use_gguf': self.use_gguf,  # GGUFの使用状況を保存
                'gguf_model_path': repr(self.gguf_model_path) if self.gguf_model_path else None,  # パスはreprで保存
            }, best_model_path)
        
        # 訓練完了時間の表示
        total_time = time.time() - start_time
        self.logger.info(f"Distillation completed in {str(timedelta(seconds=int(total_time)))}")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
        
        return best_model_path
    
    def save_model_hf_format(self, model, tokenizer, output_dir):
        """モデルをHugging Face形式で保存"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        self.logger.info(f"Model and tokenizer saved in Hugging Face format to {output_dir}")
