import os
import sys
import argparse
import torch
import logging
import numpy as np
import psutil
from typing import Dict
from importlib.metadata import version
from datetime import datetime

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
    
    subparsers = parser.add_subparsers(dest="mode", help="操作モード")
    
    # 蒸留モード
    distill_parser = subparsers.add_parser("distill", help="知識蒸留を実行")
    distill_parser.add_argument("--teacher_model", type=str, default="elyza/ELYZA-Thinking-1.0-Qwen-32B",
                              help="教師モデル名")
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
    
    # モードが指定されていない場合はヘルプを表示
    if not args.mode:
        parser.print_help()
        sys.exit(1)
    
    return args

def print_streaming_text(text):
    """ストリーミングテキストを出力"""
    sys.stdout.write(text)
    sys.stdout.flush()

def run_distillation(args):
    """知識蒸留を実行"""
    logger.info("Starting knowledge distillation process")
    start_time = datetime.now()
    
    # システムリソースを表示
    log_memory_usage(logger)
    
    # システム環境のチェック
    if sys.platform.startswith('win'):
        logger.info("Windows環境を検出しました")
        # WindowsでQwen2モデルの場合、自動的にwindows_modeをオン
        if 'qwen' in args.teacher_model.lower() and not args.use_cpu_only:
            if not args.windows_mode:
                logger.warning("WindowsでQwen2モデルを使用する場合はwindows_modeを推奨します")
                if input("windows_modeを有効にしますか？ (y/n): ").lower() == 'y':
                    args.windows_mode = True
                    logger.info("windows_modeを有効にしました")
    
    # PyTorch nightlyを検出
    is_nightly = False
    try:
        torch_version = torch.__version__
        if 'dev' in torch_version or 'nightly' in torch_version:
            is_nightly = True
            logger.info(f"PyTorch nightly版を検出: {torch_version}")
            if not args.use_direct_gpu:
                logger.warning("PyTorch nightly版ではGPU最適化ライブラリが使用できない場合があります")
                logger.info("--use_direct_gpu オプションを使用することを推奨します")
                if input("use_direct_gpuモードを有効にしますか？ (y/n): ").lower() == 'y':
                    args.use_direct_gpu = True
    except:
        pass
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # GPU情報を表示
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # トークナイザーの読み込み
    logger.info(f"Loading tokenizer for {args.teacher_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        logger.info("Trying slow tokenizer instead...")
        try:
            # fast=FalseオプションでLLaMAのslow tokenizerを使用
            tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, use_fast=False)
            logger.info("Successfully loaded slow tokenizer")
        except Exception as inner_e:
            logger.error(f"Also failed to load slow tokenizer: {inner_e}")
            logger.error("Please upgrade tokenizers package: pip install --upgrade tokenizers>=0.13.3")
            sys.exit(1)
    
    # 必要に応じてPADトークンを設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    }
    
    # 知識蒸留器の初期化
    try:
        distiller = KnowledgeDistiller(
            teacher_model_name=args.teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            device=device,
            config=config,
            quantize=args.quantize,
            cpu_offload=args.cpu_offload
        )
    except Exception as e:
        logger.error(f"Error initializing distiller with {args.teacher_model}: {e}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 蒸留データの生成
    logger.info(f"Generating {args.num_examples} distillation examples")
    train_data_path = os.path.join(args.output_dir, "train_data.json")
    val_data_path = os.path.join(args.output_dir, "val_data.json")
    if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
        distiller.prepare_distillation_data(
            questions_file="questions.txt",
            output_file=train_data_path,
            num_samples=int(args.num_examples * 0.8),
            cache_to_ram=args.use_ram_cache
        )
        distiller.prepare_distillation_data(
            questions_file="questions.txt",
            output_file=val_data_path,
            num_samples=int(args.num_examples * 0.2),
            cache_to_ram=args.use_ram_cache
        )
    else:
        logger.info(f"Using existing data files: {train_data_path}, {val_data_path}")
    
    # 知識蒸留の実行
    logger.info("Starting distillation")
    best_model_path = distiller.distill(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_ram_cache=args.use_ram_cache,
        checkpoint_every=args.checkpoint_every
    )
    
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
    
    engine = InferenceEngine(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        stream_output=True
    )
    
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
    
    engine = InferenceEngine(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        stream_output=True
    )
    print("\n=== チャットモード開始 ===")
    print("終了するには 'exit' または 'quit' と入力してください。")
    
    chat_history = []
    while True:
        try:
            user_input = input("\nあなた: ")
            if user_input.lower() in ["exit", "quit"]:
                break
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
            engine.maintenance()
        except KeyboardInterrupt:
            print("\n\n終了します...")
            break
        except Exception as e:
            logger.error(f"Error during chat: {e}")
            print(f"\nエラーが発生しました: {e}\n")

def main():
    """メイン関数"""
    args = parse_arguments()
    if args.mode == "distill":
        run_distillation(args)
    elif args.mode == "infer":
        run_inference(args)
    elif args.mode == "chat":
        run_chat(args)

if __name__ == "__main__":
    main()
