import os
import sys
import argparse
import torch
import logging
import numpy as np
import psutil
import gc
from typing import Dict, Optional, List
from importlib.metadata import version
from datetime import datetime
from importlib import import_module

# バージョン確認と修正のため、transformersのインポート前にtokenizersのバージョンを確認
try:
    tokenizers_version = version('tokenizers')
    required_version = '0.13.3'
    if tokenizers_version < required_version:
        logger = logging.getLogger(__name__)
        logger.warning(f"tokenizers version {tokenizers_version} is installed, but {required_version}+ is required.")
        logger.info("Attempting to upgrade tokenizers...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "tokenizers>=0.13.3"])
            logger.info("tokenizers successfully upgraded. Restarting may be required.")
            # 環境変数を設定して、このスクリプト内では警告のみとする
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        except Exception as e:
            logger.error(f"Failed to upgrade tokenizers: {e}")
            logger.error("Please manually upgrade tokenizers with: pip install --upgrade tokenizers>=0.13.3")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                sys.exit(1)
except ImportError:
    pass  # tokenizersがインストールされていない場合は後の依存関係エラーで対応

def check_required_modules():
    """必要なモジュールが存在するか確認"""
    required_modules = {
        "brain_model": ["BrainModel"],
        "distillation": ["KnowledgeDistiller"],
        "inference": ["InferenceEngine", "StreamingCallback"],
        "memory_system": ["MemorySystem", "WorkingMemory"],
        "memory_util": ["log_memory_usage"]
    }
    
    missing_modules = []
    
    for module_name, classes in required_modules.items():
        try:
            module = import_module(module_name)
            for class_name in classes:
                if not hasattr(module, class_name):
                    missing_modules.append(f"{module_name}.{class_name}")
        except ImportError:
            missing_modules.append(module_name)
    
    if missing_modules:
        logger.error(f"以下の必要なモジュールまたはクラスが見つかりません: {', '.join(missing_modules)}")
        logger.error("プロジェクトの構造が正しいか確認してください。")
        return False
    
    return True

from transformers import AutoTokenizer
from brain_model import BrainModel
from distillation import KnowledgeDistiller
from inference import InferenceEngine, StreamingCallback
from memory_util import log_memory_usage

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(description="Human-like LLM System")
    
    # 明示的に指定されたオプションを追跡するための変数
    explicitly_set = set()
    original_parse_args = parser.parse_args
    
    # parse_argsをオーバーライドして明示的に指定されたオプションを記録
    def custom_parse_args():
        args = original_parse_args()
        for action in parser._actions:
            if action.dest != 'help' and sys.argv.count(f"--{action.dest.replace('_', '-')}"):
                explicitly_set.add(action.dest)
        return args
    
    parser.parse_args = custom_parse_args
    
    subparsers = parser.add_subparsers(dest="mode", help="操作モード")
    
    # 蒸留モード
    distill_parser = subparsers.add_parser("distill", help="知識蒸留を実行")
    distill_parser.add_argument("--teacher_model", type=str, default="elyza/ELYZA-Thinking-1.0-Qwen-32B",
                              help="教師モデル名")
    distill_parser.add_argument("--use_lmstudio", action="store_true", default=False,
                              help="LMstudioからAPIで学習データを収集")
    distill_parser.add_argument("--use_gguf", action="store_true", default=False,
                              help="GGUFモデルを使用する")
    distill_parser.add_argument("--gguf_model_path", type=str, default=None, 
                              help="GGUFモデルファイルのパス")
    distill_parser.add_argument("--gguf_context_length", type=int, default=4096,
                              help="GGUFモデルのコンテキスト長")
    distill_parser.add_argument("--gguf_gpu_layers", type=int, default=-1,
                              help="GPU上で実行するレイヤー数。-1=全レイヤー、0=CPU only")
    distill_parser.add_argument("--gguf_n_gpu", type=int, default=1,
                              help="使用するGPUの数")
    distill_parser.add_argument("--gguf_n_batch", type=int, default=512,
                               help="GGUFのバッチサイズ")
    distill_parser.add_argument("--thinking_llm", action="store_true", default=False,
                              help="考えてから出力するタイプのLLM向けの訓練を行う")
    distill_parser.add_argument("--lmstudio_url", type=str, default="http://localhost:1234/v1",
                              help="LMstudioのAPIエンドポイント")
    distill_parser.add_argument("--lmstudio_model", type=str, default=None,
                              help="LMstudioで使用するモデル名")
    distill_parser.add_argument("--output_dir", type=str, default="./models",
                              help="モデル出力ディレクトリ")
    distill_parser.add_argument("--num_examples", type=int, default=5000,
                              help="生成する蒸留データ数")
    distill_parser.add_argument("--batch_size", type=int, default=2,
                              help="バッチサイズ")
    distill_parser.add_argument("--num_epochs", type=int, default=5,
                              help="エポック数")
    distill_parser.add_argument("--quantize", action="store_true", default=True,
                              help="教師モデルの4ビット量子化を有効化")
    distill_parser.add_argument("--cpu_offload", action="store_true", default=True,
                              help="モデルの一部をCPUにオフロードする")
    distill_parser.add_argument("--gradient_accumulation", type=int, default=16,
                              help="勾配累積ステップ数")
    distill_parser.add_argument("--use_ram_cache", action="store_true", default=True,
                              help="RAMキャッシュを使用する")
    distill_parser.add_argument("--checkpoint_every", type=int, default=500,
                              help="チェックポイント保存間隔（ステップ数）")
    distill_parser.add_argument("--resume_from", type=str, default=None,
                              help="チェックポイントから学習を再開")
    distill_parser.add_argument("--save_hf_format", action="store_true", default=True,
                              help="Hugging Face形式でモデルを保存する")
    distill_parser.add_argument("--hf_model_name", type=str, default="lal-brain-model",
                              help="Hugging Face用のモデル名")
    distill_parser.add_argument("--distillation_temperature", type=float, default=2.0,
                              help="蒸留時の温度パラメータ")
    distill_parser.add_argument("--max_length", type=int, default=768,
                              help="最大シーケンス長")
    distill_parser.add_argument("--windows_mode", action="store_true", default=False,
                              help="Windows向けの最適化を有効化 (tritonの代わりにxformersを使用)")
    distill_parser.add_argument("--use_cpu_only", action="store_true", default=False,
                              help="GPUを使用せずCPUのみで実行 (GPUが利用できない環境用)")
    distill_parser.add_argument("--use_direct_gpu", action="store_true", default=False,
                              help="PyTorch nightly版でGPUに直接アクセス（最適化ライブラリなし）")
    distill_parser.add_argument("--use_triton_windows", action="store_true", default=False,
                              help="Windows環境でTritonを使用する")
    distill_parser.add_argument("--focus_subjects", type=str, default="highschool,electronics,it",
                              help="重点的に学習する分野（カンマ区切り）")
    distill_parser.add_argument("--imouto_mode", action="store_true", default=True,
                              help="妹口調で出力するモードを有効化")
    
    # 推論モード
    infer_parser = subparsers.add_parser("infer", help="推論を実行")
    infer_parser.add_argument("--model_path", type=str, default="./models/brain_model_best.pt",
                            help="モデルパス")
    infer_parser.add_argument("--tokenizer_name", type=str, default="elyza/ELYZA-Thinking-1.0-Qwen-32B",
                            help="トークナイザー名")
    infer_parser.add_argument("--interactive", action="store_true",
                            help="対話モード")
    
    # チャットモード
    chat_parser = subparsers.add_parser("chat", help="チャットを開始")
    chat_parser.add_argument("--model_path", type=str, default="./models/brain_model_best.pt",
                           help="モデルパス")
    chat_parser.add_argument("--tokenizer_name", type=str, default="elyza/ELYZA-Thinking-1.0-Qwen-32B",
                           help="トークナイザー名")
    
    args = parser.parse_args()
    
    # 明示的に指定されたオプションを属性として追加
    args.explicitly_set = explicitly_set
    
    # モードが指定されていない場合はヘルプを表示
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    return args

def print_streaming_text(text):
    """ストリーミングテキストを出力"""
    sys.stdout.write(text)
    sys.stdout.flush()

def force_gc():
    """強制的にガベージコレクションを実行してメモリを解放"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("ガベージコレクションを実行してメモリを解放しました")

def check_model_tokenizer_compatibility(model_path: str, tokenizer_name: str) -> bool:
    """モデルとトークナイザーの互換性を確認"""
    # Hugging Faceモデルの場合
    if not os.path.exists(model_path) and '/' in model_path:
        logger.info(f"Hugging Faceモデル {model_path} を使用します")
        return True
    
    # ローカルファイルの場合
    if not os.path.exists(model_path):
        logger.error(f"モデルファイルが存在しません: {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        if "model_config" in checkpoint and "tokenizer_name" in checkpoint["model_config"]:
            saved_tokenizer = checkpoint["model_config"]["tokenizer_name"]
            if saved_tokenizer != tokenizer_name:
                logger.warning(f"モデルは {saved_tokenizer} トークナイザーで訓練されましたが、{tokenizer_name} を使用しようとしています")
                logger.warning("これにより予期しない結果が発生する可能性があります")
                return input("続行しますか？ (y/n): ").lower() == 'y'
    except Exception as e:
        logger.warning(f"モデルファイルの読み込み中にエラーが発生しました: {e}")
        logger.warning("モデルとトークナイザーの互換性を確認できません")
        return input("続行しますか？ (y/n): ").lower() == 'y'
    
    return True

def run_distillation(args):
    """知識蒸留を実行"""
    logger.info("Starting knowledge distillation process")
    start_time = datetime.now()
    
    # システムリソースを表示
    log_memory_usage(logger)
    
    # LMstudioとGGUFモデルの優先順位付け
    using_lmstudio = False
    using_gguf = False
    
    if args.use_gguf:
        using_gguf = True
        logger.info("GGUFモデルを使用して蒸留データを生成します")
        if not args.gguf_model_path:
            logger.error("GGUF モデルパスが指定されていません。--gguf_model_path を指定してください")
            sys.exit(1)
        if not os.path.exists(args.gguf_model_path):
            logger.error(f"指定されたGGUFモデルファイルが存在しません: {args.gguf_model_path}")
            sys.exit(1)
        if args.teacher_model and 'teacher_model' in args.explicitly_set:
            logger.warning("GGUFモードが有効なため、指定されたteacher_modelは無視されます")
        if args.use_lmstudio:
            logger.warning("GGUFモードとLMstudioモードは同時に使用できません。GGUFモードを優先します")
            args.use_lmstudio = False
    elif args.use_lmstudio:
        using_lmstudio = True
        logger.info("LMstudioのAPIを使用して蒸留データを生成します")
        if args.teacher_model and 'teacher_model' in args.explicitly_set:
            logger.warning("LMstudioモードが有効なため、指定されたteacher_modelは無視されます")
    elif not args.teacher_model:
        logger.error("教師モデルが指定されていません。--teacher_model、--use_lmstudio、または --use_gguf を指定してください")
        sys.exit(1)
    
    # DeepSeek-R1などの「考えてから出力するタイプのLLM」の設定
    if args.thinking_llm or (args.teacher_model and "deepseek" in args.teacher_model.lower()):
        logger.info("考えてから出力するタイプのLLM向けの訓練を有効化しました")
        thinking_llm = True
    else:
        thinking_llm = False
    
    # PyTorch nightlyを検出
    is_nightly = False
    try:
        torch_version = torch.__version__
        if 'dev' in torch_version or 'nightly' in torch_version:
            is_nightly = True
            logger.info(f"PyTorch nightly版を検出: {torch_version}")
            if not args.use_direct_gpu and 'use_direct_gpu' not in args.explicitly_set:
                logger.warning("PyTorch nightly版ではGPU最適化ライブラリが使用できない場合があります")
                logger.info("--use_direct_gpu オプションを使用することを推奨します")
                if input("use_direct_gpuモードを有効にしますか？ (y/n): ").lower() == 'y':
                    args.use_direct_gpu = True
    except:
        pass
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu_only else "cpu")
    logger.info(f"Using device: {device}")
    
    # GPU情報を表示
    if torch.cuda.is_available() and not args.use_cpu_only:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        if using_gguf:
            # GGUFモードでGPU使用時は追加情報をログ
            gpu_layers = args.gguf_gpu_layers if hasattr(args, 'gguf_gpu_layers') else -1
            gpu_count = args.gguf_n_gpu if hasattr(args, 'gguf_n_gpu') else 1
            batch_size = args.gguf_n_batch if hasattr(args, 'gguf_n_batch') else 512
            logger.info(f"GGUF GPU設定: GPU層数={gpu_layers}, GPU数={gpu_count}, バッチサイズ={batch_size}")
    
    # トークナイザーの読み込み
    tokenizer = None
    if using_gguf:
        # GGUFモデル使用時はベースモデルのトークナイザーを使用
        try:
            # DeepSeek-R1などはQwen系のトークナイザーを使用
            if "deepseek" in args.gguf_model_path.lower() and "qwen" in args.gguf_model_path.lower():
                logger.info("DeepSeek-Qwen系のトークナイザーを使用します")
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
            else:
                # デフォルトはLLaMAトークナイザーを使用
                logger.info("デフォルトのLLaMAトークナイザーを使用します")
                tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
            
            # 必要に応じてPADトークンを設定
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer for GGUF model: {e}")
            if "custom code" in str(e) and "trust_remote_code=True" in str(e):
                logger.error("カスタムコードを含むモデルを使用しています。再試行します。")
                try:
                    if "deepseek" in args.gguf_model_path.lower() and "qwen" in args.gguf_model_path.lower():
                        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
                    elif "qwen" in args.gguf_model_path.lower():
                        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
                    elif "chatglm" in args.gguf_model_path.lower():
                        tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
                    logger.info("trust_remote_code=Trueオプションでトークナイザーの読み込みに成功しました")
                except Exception as e2:
                    logger.error(f"トークナイザー読み込みの2回目の試行も失敗しました: {e2}")
                    sys.exit(1)
            else:
                sys.exit(1)
    elif not using_lmstudio:
        logger.info(f"Loading tokenizer for {args.teacher_model}")
        try:
            needs_trust_remote = any(x in args.teacher_model.lower() for x in ["qwen", "chatglm", "deepseek", "baichuan", "internlm"])
            
            if needs_trust_remote:
                logger.info(f"カスタムコードを含むモデルと判断し、trust_remote_code=Trueを使用します: {args.teacher_model}")
                tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            if "custom code" in str(e) and "trust_remote_code=True" in str(e):
                logger.info("カスタムコードを含むモデルと判明しました。trust_remote_code=Trueで再試行します。")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
                    logger.info("trust_remote_code=Trueオプションでトークナイザーの読み込みに成功しました")
                except Exception as inner_e:
                    logger.error(f"trust_remote_code=Trueでも失敗しました: {inner_e}")
                    logger.info("Trying slow tokenizer instead...")
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=False, trust_remote_code=True)
                        logger.info("Successfully loaded slow tokenizer with trust_remote_code=True")
                    except Exception as slow_e:
                        logger.error(f"Also failed to load slow tokenizer: {slow_e}")
                        logger.error("Please upgrade tokenizers package: pip install --upgrade tokenizers>=0.13.3")
                        sys.exit(1)
            else:
                logger.info("Trying slow tokenizer instead...")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=False)
                    logger.info("Successfully loaded slow tokenizer")
                except Exception as inner_e:
                    logger.error(f"Also failed to load slow tokenizer: {inner_e}")
                    logger.error("Please upgrade tokenizers package: pip install --upgrade tokenizers>=0.13.3")
                    sys.exit(1)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            logger.info("LMstudio用にデフォルトのGPT-2トークナイザーを使用します")
        except Exception as e:
            logger.error(f"Failed to load default tokenizer for LMstudio: {e}")
            sys.exit(1)
    
    # 生徒モデルの初期化
    logger.info("Initializing student model")
    student_model = BrainModel(
        vocab_size=len(tokenizer),
        embedding_dim=768,
        hidden_dim=1024,
        memory_size=1000
    )
    
    # チェックポイントから再開する場合
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        student_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 蒸留設定
    config = {
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "distillation_temperature": args.distillation_temperature,
        "distillation_alpha": 0.5,
        "max_length": args.max_length,  # 最大シーケンス長を設定
        "windows_mode": args.windows_mode,  # Windows対応モード
        "use_cpu_only": args.use_cpu_only,  # CPU専用モード
        "use_direct_gpu": args.use_direct_gpu or is_nightly,  # PyTorch nightly対応
        "use_triton_windows": args.use_triton_windows,  # triton-windows使用
        "tokenizer_name": args.teacher_model if not (using_lmstudio or using_gguf) else "gpt2",  # トークナイザー名
        "focus_subjects": args.focus_subjects.split(','),
        "imouto_mode": args.imouto_mode,
        "use_lmstudio": using_lmstudio,
        "lmstudio_url": args.lmstudio_url,
        "lmstudio_model": args.lmstudio_model,
        "use_gguf": using_gguf,
        "gguf_model_path": args.gguf_model_path if using_gguf else None,
        "gguf_context_length": args.gguf_context_length if using_gguf else None,
        "thinking_llm": thinking_llm,
        # GGUF用GPUオプションを強化
        "gguf_use_gpu": not args.use_cpu_only and torch.cuda.is_available(),
        "gguf_n_gpu_layers": args.gguf_gpu_layers if hasattr(args, 'gguf_gpu_layers') else -1,
        "gguf_n_gpu": args.gguf_n_gpu if hasattr(args, 'gguf_n_gpu') else 1,
        "gguf_n_batch": args.gguf_n_batch if hasattr(args, 'gguf_n_batch') else 512,
    }
    
    # 知識蒸留器の初期化
    try:
        distiller = KnowledgeDistiller(
            teacher_model_name=None if using_lmstudio or using_gguf else args.teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            quantize=args.quantize and not (using_lmstudio or using_gguf),  # 外部モデル利用時は量子化不要
            cpu_offload=args.cpu_offload and not (using_lmstudio or using_gguf)  # 外部モデル利用時はオフロード不要
        )
    except Exception as e:
        if using_gguf:
            model_name = args.gguf_model_path
        elif using_lmstudio:
            model_name = args.lmstudio_url
        else:
            model_name = args.teacher_model
        logger.error(f"Error initializing distiller with {model_name}: {e}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 蒸留データの生成
    logger.info(f"Generating {args.num_examples} distillation examples focusing on: {args.focus_subjects}")
    train_data_path = os.path.join(args.output_dir, "train_data.json")
    val_data_path = os.path.join(args.output_dir, "val_data.json")
    if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
        distiller.prepare_distillation_data(
            questions_file="questions.txt",
            output_file=train_data_path,
            num_samples=int(args.num_examples * 0.8),
            cache_to_ram=args.use_ram_cache,
            focus_subjects=args.focus_subjects.split(','),
            imouto_mode=args.imouto_mode,
            thinking_llm=thinking_llm  # 考えてから出力するタイプのLLM用のデータ生成
        )
        distiller.prepare_distillation_data(
            questions_file="questions.txt",
            output_file=val_data_path,
            num_samples=int(args.num_examples * 0.2),
            cache_to_ram=args.use_ram_cache,
            focus_subjects=args.focus_subjects.split(','),
            imouto_mode=args.imouto_mode,
            thinking_llm=thinking_llm  # 考えてから出力するタイプのLLM用のデータ生成
        )
    else:
        logger.info(f"Using existing data files: {train_data_path}, {val_data_path}")
    
    # 知識蒸留の実行
    logger.info("Starting distillation")
    try:
        best_model_path = distiller.distill(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            use_ram_cache=args.use_ram_cache,
            checkpoint_every=args.checkpoint_every,
            config=config  # 設定情報もチェックポイントに保存
        )
    except Exception as e:
        logger.error(f"知識蒸留中にエラーが発生しました: {e}")
        logger.info("最後のチェックポイントから再開できる可能性があります")
        return None
    
    # モデル使用後のメモリ解放
    force_gc()
    
    # Hugging Face形式でモデルを保存
    if args.save_hf_format:
        logger.info(f"Saving model in Hugging Face format as {args.hf_model_name}")
        hf_output_dir = os.path.join(args.output_dir, args.hf_model_name)
        os.makedirs(hf_output_dir, exist_ok=True)
        student_model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
        distiller.save_model_hf_format(
            model=student_model,
            tokenizer=tokenizer,
            output_dir=hf_output_dir
        )
        logger.info(f"Model saved in Hugging Face format at {hf_output_dir}")
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"Distillation completed in {elapsed}")
    return best_model_path

def run_inference(args):
    """推論を実行"""
    logger.info("Starting inference")
    
    # モデルとトークナイザーの互換性チェック
    if not check_model_tokenizer_compatibility(args.model_path, args.tokenizer_name):
        logger.error("モデルとトークナイザーの互換性の問題により中止します")
        return
    
    try:
        engine = InferenceEngine(
            model_path=args.model_path,
            tokenizer_name=args.tokenizer_name,
            stream_output=True
        )
    except Exception as e:
        logger.error(f"推論エンジンの初期化に失敗しました: {e}")
        return
    
    if args.interactive:
        print("\n=== 対話モード開始 ===")
        print("終了するには 'exit' または 'quit' と入力してください。")
        
        while True:
            try:
                user_input = input("\n質問: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                callback = StreamingCallback(print_streaming_text)
                print("\n回答: ", end="")
                engine.generate_response(user_input, callback)
                print("\n")
                engine.maintenance()
            except KeyboardInterrupt:
                print("\n\n終了します...")
                break
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                print(f"\nエラーが発生しました: {e}\n")
    else:
        sample_questions = [
            "人工知能とは何ですか？",
            "気候変動対策で最も効果的な方法は何だと思いますか？",
            "量子コンピュータの将来性について教えてください"
        ]
        for question in sample_questions:
            print(f"\n質問: {question}")
            print("回答: ", end="")
            callback = StreamingCallback(print_streaming_text)
            engine.generate_response(question, callback)
            print("\n")

def run_chat(args):
    """チャットモードを実行"""
    logger.info("Starting chat mode")
    
    # モデルとトークナイザーの互換性チェック
    if not check_model_tokenizer_compatibility(args.model_path, args.tokenizer_name):
        logger.error("モデルとトークナイザーの互換性の問題により中止します")
        return
    
    try:
        engine = InferenceEngine(
            model_path=args.model_path,
            tokenizer_name=args.tokenizer_name,
            stream_output=True
        )
    except Exception as e:
        logger.error(f"推論エンジンの初期化に失敗しました: {e}")
        return
    
    print("\n=== チャットモード開始 ===")
    print("終了するには 'exit' または 'quit' と入力してください。")
    
    chat_history = []
    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # チャット履歴が大きくなりすぎないように制限
            if len(chat_history) > 100:
                chat_history = chat_history[-50:]
                logger.info("チャット履歴を整理しました")
            
            chat_history.append({"role": "user", "content": user_input})
            context = ""
            for i, message in enumerate(chat_history[-5:]):
                role = "あなた" if message["role"] == "user" else "AI"
                context += f"{role}: {message['content']}\n"
            context += "AI: "
            print("\nAI: ", end="")
            callback = StreamingCallback(print_streaming_text)
            response = engine.generate_response(context, callback)
            chat_history.append({"role": "assistant", "content": response})
            
            # メモリの定期的なクリーンアップ
            if len(chat_history) % 10 == 0:
                force_gc()
                
            engine.maintenance()
        except KeyboardInterrupt:
            print("\n\n終了します...")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"\nエラーが発生しました: {e}\n")
            # エラー発生後もメモリをクリーンアップ
            force_gc()

def main():
    """メイン関数"""
    # 必須モジュールのチェック
    if not check_required_modules():
        logger.error("必須モジュールが見つからないため、プログラムを終了します。")
        sys.exit(1)
        
    args = parse_arguments()
    try:
        if args.mode == "distill":
            run_distillation(args)
        elif args.mode == "infer":
            run_inference(args)
        elif args.mode == "chat":
            run_chat(args)
    except Exception as e:
        logger.critical(f"予期しないエラーが発生しました: {e}")
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
    finally:
        # プログラム終了時にメモリを解放
        force_gc()

if __name__ == "__main__":
    main()
