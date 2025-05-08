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

# 依存モジュールのインポート
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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

def check_model_parameters(model):
    """モデルパラメータの整合性をチェックする関数"""
    try:
        # 次元の整合性を確認
        issues = []
        
        # 各モジュールの次元を確認
        if hasattr(model, 'left_brain') and hasattr(model, 'right_brain'):
            left_dim = model.left_brain.hidden_dim if hasattr(model.left_brain, 'hidden_dim') else -1
            right_dim = model.right_brain.hidden_dim if hasattr(model.right_brain, 'hidden_dim') else -1
            
            if left_dim != right_dim and left_dim > 0 and right_dim > 0:
                issues.append(f"左脳と右脳の次元が一致しません: {left_dim} vs {right_dim}")
        
        # 埋め込み層と出力層の次元を確認
        if hasattr(model, 'embedding') and hasattr(model, 'output'):
            embed_dim = model.embedding.embedding_dim if hasattr(model.embedding, 'embedding_dim') else -1
            output_dim = model.output.in_features if hasattr(model.output, 'in_features') else -1
            
            if embed_dim != output_dim and embed_dim > 0 and output_dim > 0:
                issues.append(f"埋め込み次元と出力次元が一致しません: {embed_dim} vs {output_dim}")
        
        # 問題があれば報告
        if issues:
            logger.warning("モデルパラメータに問題が見つかりました:")
            for issue in issues:
                logger.warning(f" - {issue}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"モデルパラメータ確認中にエラーが発生: {e}")
        return False

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
    distill_parser.add_argument("--questions_file", type=str, default="questions.txt",
                              help="質問ファイル名")
    distill_parser.add_argument("--batch_size", type=int, default=2,
                              help="バッチサイズ")
    distill_parser.add_argument("--num_epochs", type=int, default=5,
                              help="エポック数")
    distill_parser.add_argument("--quantize", action="store_true", default=True,
                              help="教師モデルの8ビット量子化を有効化")
    distill_parser.add_argument("--cpu_offload", action="store_true", default=True,
                              help="モデルの一部をCPUにオフロードする")
    distill_parser.add_argument("--gradient_accumulation", type=int, default=16,
                              help="勾配累積ステップ数")
    distill_parser.add_argument("--learning_rate", type=float, default=5e-5,
                              help="学習率")
    distill_parser.add_argument("--weight_decay", type=float, default=0.01,
                              help="重み減衰") 
    distill_parser.add_argument("--warmup_steps", type=int, default=500,
                              help="ウォームアップステップ数")
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
    distill_parser.add_argument("--skip_teacher_model", action="store_true",
                              help="教師モデルをロードせず、直接学習する")
    distill_parser.add_argument("--hidden_size", type=int, default=768, help="隠れ層のサイズ")
    distill_parser.add_argument("--embedding_dim", type=int, default=None, help="埋め込み次元（指定された場合はhidden_sizeより優先）")
    distill_parser.add_argument("--model_dim", type=int, default=768, help="モデル次元（hidden_sizeと同じ値を推奨）")
    
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
    
    # focus_subjectsをリストに変換 - カンマ区切りの文字列から適切にリストに変換
    if hasattr(args, 'focus_subjects') and isinstance(args.focus_subjects, str):
        args.focus_subjects = [s.strip() for s in args.focus_subjects.split(',') if s.strip()]
        logger.info(f"重点分野: {', '.join(args.focus_subjects)}")
    
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

def load_tokenizer_and_models(args):
    """トークナイザーとモデルを読み込む"""
    # デバイスの設定
    if args.use_cpu_only:
        device = torch.device("cpu")
        logger.info("Using device: cpu (as specified by use_cpu_only)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using device: {device}")
            logger.info(f"GPU: {gpu_name}")
            logger.info(f"CUDA Version: {cuda_version}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
        else:
            logger.info("No GPU found, using CPU")
    
    # トークナイザーをロード
    logger.info(f"Loading tokenizer for {args.teacher_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info("パッドトークンがないため、EOSトークンをパッドトークンとして設定しました")
    except Exception as e:
        logger.error(f"トークナイザーのロードに失敗: {e}")
        logger.info("代わりにデフォルトのトークナイザー elyza/ELYZA-japanese-Llama-2-7b を使用します")
        try:
            tokenizer = AutoTokenizer.from_pretrained("elyza/ELYZA-japanese-Llama-2-7b")
        except:
            logger.critical("代替トークナイザーも読み込めませんでした。終了します。")
            sys.exit(1)
    
    # トークナイザーから語彙サイズを取得
    vocab_size = len(tokenizer)
    logger.info(f"語彙サイズ: {vocab_size}")
    
    # モデルのパラメータ設定（デフォルト値を用意）
    hidden_size = getattr(args, 'hidden_size', 768)
    embedding_dim = getattr(args, 'embedding_dim', None)
    num_layers = getattr(args, 'num_layers', 12)
    num_heads = getattr(args, 'num_heads', 12)
    dropout = getattr(args, 'dropout', 0.1)
    
    logger.info(f"モデル設定: hidden_size={hidden_size}, num_layers={num_layers}, num_heads={num_heads}")

    # 学生モデル初期化と次元調整
    try:
        logger.info("Initializing student model")
        student_model = BrainModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            output_size=vocab_size,  # 出力サイズは語彙サイズに合わせる
            dropout=dropout
        )
    except Exception as e:
        logger.error(f"学生モデルの初期化に失敗しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # フォールバック: シンプルな設定での再試行
        try:
            logger.info("シンプルな設定でモデル初期化を再試行します")
            student_model = BrainModel(
                vocab_size=vocab_size,
                hidden_size=768,  # 固定値を使用
                num_layers=12,
                num_heads=12,
                dropout=0.1
            )
        except Exception as fallback_error:
            logger.error(f"フォールバック初期化も失敗しました: {fallback_error}")
            exit(1)
    
    # 教師モデル（ベースとなる大規模モデル）の設定構成
    teacher_model = None
    if not args.use_lmstudio and not args.use_gguf and not args.skip_teacher_model:
        # 量子化の設定
        if args.quantize:
            logger.info("Initializing 8-bit quantization")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            except Exception as e:
                logger.error(f"量子化設定の初期化に失敗: {e}")
                logger.warning("量子化を無効化して続行します")
                bnb_config = None
        else:
            bnb_config = None
        
        # 教師モデルのロード
        logger.info(f"Loading teacher model: {args.teacher_model}")
        try:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                args.teacher_model,
                device_map="auto" if not args.use_cpu_only else "cpu",
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                quantization_config=bnb_config if bnb_config is not None else None,
                trust_remote_code=True
            )
            logger.info("教師モデルのロードが完了しました")
        except Exception as e:
            logger.error(f"教師モデルのロードに失敗: {e}")
            if not args.use_lmstudio and not args.use_gguf:
                logger.critical("教師モデルが必要ですが、ロードに失敗しました。LMStudioモードに変更するか、別のモデルを指定してください。終了します。")
                sys.exit(1)
    
    # KnowledgeDistillerインスタンスの初期化
    try:
        distiller = KnowledgeDistiller(
            teacher_model_name=args.teacher_model if not args.skip_teacher_model else None,
            student_model=student_model,
            tokenizer=tokenizer,
            device=device,
            config={
                "use_lmstudio": args.use_lmstudio,
                "lmstudio_url": args.lmstudio_url,
                "lmstudio_model": args.lmstudio_model,
                "windows_mode": args.windows_mode,
                "use_cpu_only": args.use_cpu_only,
                "use_direct_gpu": args.use_direct_gpu,
                "use_triton_windows": args.use_triton_windows,
                "focus_subjects": args.focus_subjects,
                "imouto_mode": args.imouto_mode,
                "use_gguf": args.use_gguf,
                "gguf_model_path": args.gguf_model_path,
                "gguf_context_length": args.gguf_context_length,
                "thinking_llm": args.thinking_llm,
                "tokenizer_name": args.teacher_model,
                "max_length": args.max_length,
                "num_samples": args.num_examples,
                "learning_rate": getattr(args, "learning_rate", 5e-5),
                "weight_decay": getattr(args, "weight_decay", 0.01),
                "warmup_steps": getattr(args, "warmup_steps", 500),
            },
            quantize=args.quantize,
            cpu_offload=args.cpu_offload,
            use_cpu_only=args.use_cpu_only,
            skip_teacher_model=args.skip_teacher_model
        )
        # 教師モデルをセット (LMStudioやGGUF以外の場合)
        if teacher_model is not None:
            distiller.teacher_model = teacher_model
    except Exception as e:
        logger.critical(f"Distillerの初期化に失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    return tokenizer, student_model, distiller

def run_distillation(args):
    """知識蒸留を実行する"""
    from memory_utils import validate_model_config
    
    # モデル設定の検証と補正
    args = validate_model_config(args)
    
    # ステップ1: トークナイザー、モデル、蒸留器をロード
    tokenizer, student_model, distiller = load_tokenizer_and_models(args)
    
    # モデル次元の整合性チェック
    def check_model_dimensions(vocab_size):
        """モデル次元の整合性をチェック"""
        # 出力サイズが語彙サイズと一致しているか確認
        if args.hidden_size != args.model_dim:
            logger.warning(f"モデル次元の不一致: hidden_size={args.hidden_size}, model_dim={args.model_dim}")
            # 自動修正
            args.model_dim = args.hidden_size
            logger.info(f"model_dimをhidden_sizeに合わせて自動調整: {args.model_dim}")
        
        logger.info(f"モデル次元情報: vocab_size={vocab_size}, hidden_size={args.hidden_size}, model_dim={args.model_dim}")
        return args.hidden_size, args.model_dim  # 調整済みの値を返す
    
    # 次元を確認・調整
    hidden_size, model_dim = check_model_dimensions(len(tokenizer))
    
    # デバイスの設定を取得して学生モデルを明示的に移動
    device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu_only else "cpu")
    student_model = student_model.to(device)
    logger.info(f"学生モデルを {device} デバイスに明示的に移動しました")
    
    # モデルが正しく初期化されているか確認
    logger.info("モデルパラメータの整合性をチェックしています...")
    if not check_model_parameters(student_model):
        logger.warning("モデルパラメータに問題が見つかりました。次元修正機能を有効化します。")
        distiller.config["use_dimension_fix"] = True

    # ステップ2: 蒸留パスの設定
    train_data_path = os.path.join("models", "train_data.json")
    val_data_path = os.path.join("models", "val_data.json")
    output_dir = os.path.join("models", "brain_model")
    
    # ステップ3: 出力ディレクトリを確保
    os.makedirs(output_dir, exist_ok=True)
    
    # ステップ4: 蒸留データを準備
    logger.info(f"Generating {args.num_examples} distillation examples focusing on: {', '.join(args.focus_subjects)}")
    
    # 訓練データと検証データがなければ生成
    if not os.path.exists(train_data_path) or not os.path.exists(val_data_path):
        try:
            generated_data_path = distiller.prepare_distillation_data(
                questions_file=args.questions_file,
                output_file=train_data_path,
                num_samples=args.num_examples,
                batch_size=args.batch_size,
                focus_subjects=args.focus_subjects,
                imouto_mode=args.imouto_mode,
                thinking_llm=args.thinking_llm
            )
            if generated_data_path and generated_data_path != train_data_path:
                logger.info(f"データが別のパス {generated_data_path} に生成されました。必要なら手動でコピーしてください。")
        except Exception as e:
            logger.error(f"蒸留データの準備中にエラー発生: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.critical("蒸留データの準備に失敗したため、蒸留プロセスを続行できません。")
            return
    
    # ステップ5: 蒸留の実行
    logger.info("Starting distillation")
    best_model_path = None
    try:
        best_model_path = distiller.distill(
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            output_dir=output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            gradient_accumulation_steps=args.gradient_accumulation,
            checkpoint_every=args.checkpoint_every,
            config={
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "max_length": args.max_length
            }
        )
    except Exception as e:
        logger.error(f"知識蒸留中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("最後のチェックポイントから再開できる可能性があります")
    
    # ステップ6: 最終モデルの取得とロード
    if best_model_path is None:
        logger.error("知識蒸留に失敗しました。モデル保存をスキップします。")
        # 最後のチェックポイントがあれば代わりに使う
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint-")]
        if checkpoints:
            # 数値順にソート
            checkpoints.sort(key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0, reverse=True)
            latest_checkpoint = os.path.join(output_dir, checkpoints[0], "model.pt")
            if os.path.exists(latest_checkpoint):
                logger.info(f"最新のチェックポイント {latest_checkpoint} を最終モデルとして使用します")
                best_model_path = latest_checkpoint
    
    if best_model_path:
        try:
            # 最後のチェックポイントがある場合、ロード
            logger.info(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location="cpu")
            if "model_state_dict" in checkpoint:
                student_model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("モデルのロードが成功しました")
            else:
                logger.error("チェックポイントにmodel_state_dictが見つかりません")
        except Exception as e:
            logger.error(f"最後のチェックポイントの読み込みに失敗: {e}")
    
    # ステップ7: Hugging Face形式で保存
    if args.save_hf_format:
        logger.info(f"Saving model in Hugging Face format as {args.hf_model_name}")
        hf_output_dir = os.path.join("models", "hf_" + (args.hf_model_name.split("/")[-1] if "/" in args.hf_model_name else args.hf_model_name))
        
        try:
            # モデルが読み込まれているかどうかに関わらず保存を試みる
            success = distiller.save_model_hf_format(
                model_path=best_model_path, 
                output_dir=hf_output_dir,
                model_name=args.hf_model_name
            )
            
            if success:
                logger.info(f"Model successfully saved in Hugging Face format to {hf_output_dir}")
            else:
                logger.error("Failed to save model in Hugging Face format")
        except Exception as e:
            logger.error(f"Hugging Face形式での保存中にエラーが発生: {e}")
    
    # ガベージコレクション実行
    force_gc()

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
    # RAM使用状況をログ出力
    ram = psutil.virtual_memory()
    logger.info(f"RAM: 使用中 {ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB ({ram.percent:.1f}%)")
    
    # GPUメモリ使用状況をログ出力
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): 確保 {allocated_mem:.1f}GB / 予約 {reserved_mem:.1f}GB / 合計 {total_mem:.1f}GB")
    
    # PyTorch nightly版の検出
    if torch.__version__.find('dev') >= 0 or torch.__version__.find('+') >= 0:
        logger.info(f"PyTorch nightly版を検出: {torch.__version__}")
        logger.warning("PyTorch nightly版ではGPU最適化ライブラリが使用できない場合があります")
        logger.info("--use_direct_gpu オプションを使用することを推奨します")
        if input("use_direct_gpuモードを有効にしますか？ (y/n): ").lower() == 'y':
            # args.use_direct_gpu = True
            pass
    
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
