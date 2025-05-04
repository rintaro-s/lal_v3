import os
import sys
import argparse
import torch
import logging
import numpy as np
import psutil
from typing import Dict
from transformers import AutoTokenizer
from datetime import datetime

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
    distill_parser.add_argument("--num_examples", type=int, default=1000,
                              help="生成する蒸留データ数")
    distill_parser.add_argument("--batch_size", type=int, default=4,
                              help="バッチサイズ")
    distill_parser.add_argument("--num_epochs", type=int, default=3,
                              help="エポック数")
    # 新しいメモリ最適化オプションを追加
    distill_parser.add_argument("--quantize", action="store_true", default=True,
                              help="教師モデルの4ビット量子化を有効化")
    distill_parser.add_argument("--cpu_offload", action="store_true", default=True,
                              help="モデルの一部をCPUにオフロードする")
    distill_parser.add_argument("--gradient_accumulation", type=int, default=8,
                              help="勾配累積ステップ数")
    distill_parser.add_argument("--use_ram_cache", action="store_true", default=True,
                              help="RAMキャッシュを使用する")
    distill_parser.add_argument("--checkpoint_every", type=int, default=500,
                              help="チェックポイント保存間隔（ステップ数）")
    distill_parser.add_argument("--resume_from", type=str, default=None,
                              help="チェックポイントから学習を再開")
    
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
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    
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
        # 他の必要な状態も読み込む
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # global_step = checkpoint['global_step']
        # epoch = checkpoint['epoch']
    
    # 蒸留設定
    config = {
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "distillation_temperature": 2.0,
        "distillation_alpha": 0.5,
    }
    
    # 知識蒸留器の初期化
    distiller = KnowledgeDistiller(
        teacher_model_name=args.teacher_model,
        student_model=student_model,
        tokenizer=tokenizer,
        device=device,
        config=config,
        quantize=args.quantize,
        cpu_offload=args.cpu_offload
    )
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 実行時間の見積もりを表示
    from time_estimation import estimate_distillation_time
    estimate = estimate_distillation_time(
        num_examples=args.num_examples,
        teacher_model_name=args.teacher_model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs
    )
    
    # 蒸留データの生成
    logger.info(f"Generating {args.num_examples} distillation examples")
    train_data_path = os.path.join(args.output_dir, "train_data.json")
    val_data_path = os.path.join(args.output_dir, "val_data.json")
    
    # データ生成ステップをスキップするオプション
    if not (os.path.exists(train_data_path) and os.path.exists(val_data_path)):
        # 訓練データと検証データを生成
        distiller.generate_distillation_examples(
            num_examples=int(args.num_examples * 0.8),
            output_file=train_data_path,
            batch_size=args.batch_size
        )
        
        distiller.generate_distillation_examples(
            num_examples=int(args.num_examples * 0.2),
            output_file=val_data_path,
            batch_size=args.batch_size
        )
    else:
        logger.info(f"Using existing data files: {train_data_path}, {val_data_path}")
    
    # 知識蒸留の実行
    logger.info("Starting distillation")
    distiller.distill(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_ram_cache=args.use_ram_cache,
        checkpoint_every=args.checkpoint_every
    )
    
    # 完了時間を記録
    end_time = datetime.now()
    elapsed = end_time - start_time
    logger.info(f"Distillation completed in {elapsed}")

def run_inference(args):
    """推論を実行"""
    logger.info("Starting inference")
    
    # 推論エンジンの初期化
    engine = InferenceEngine(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        stream_output=True
    )
    
    if args.interactive:
        # 対話モード
        print("\n=== 対話モード開始 ===")
        print("終了するには 'exit' または 'quit' と入力してください。")
        
        while True:
            try:
                user_input = input("\n質問: ")
                
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                # ストリーミングコールバックを設定
                callback = StreamingCallback(print_streaming_text)
                
                print("\n回答: ", end="")
                engine.generate_response(user_input, callback)
                print("\n")
                
                # メモリシステムのメンテナンスを実行
                engine.maintenance()
                
            except KeyboardInterrupt:
                print("\n\n終了します...")
                break
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                print(f"\nエラーが発生しました: {e}\n")
    else:
        # サンプル推論
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
    
    # 推論エンジンの初期化
    engine = InferenceEngine(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer_name,
        stream_output=True
    )
    
    print("\n=== チャットモード開始 ===")
    print("終了するには 'exit' または 'quit' と入力してください。")
    
    # チャット履歴
    chat_history = []
    
    while True:
        try:
            user_input = input("\nあなた: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            
            # 入力を履歴に追加
            chat_history.append({"role": "user", "content": user_input})
            
            # チャット履歴を含めた入力を作成
            context = ""
            for i, message in enumerate(chat_history[-5:]):  # 直近5回分の会話のみ
                role = "あなた" if message["role"] == "user" else "AI"
                context += f"{role}: {message['content']}\n"
            
            # 最後にAIへの入力として現在の質問を追加
            context += "AI: "
            
            print("\nAI: ", end="")
            
            # ストリーミングコールバックを設定
            callback = StreamingCallback(print_streaming_text)
            
            # 応答を生成
            response = engine.generate_response(context, callback)
            
            # 応答を履歴に追加
            chat_history.append({"role": "assistant", "content": response})
            
            # メモリシステムのメンテナンス
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
    
    # 選択されたモードに応じて実行
    if args.mode == "distill":
        run_distillation(args)
    elif args.mode == "infer":
        run_inference(args)
    elif args.mode == "chat":
        run_chat(args)

if __name__ == "__main__":
    main()
