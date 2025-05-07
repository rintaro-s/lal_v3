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
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import math
import psutil
from datetime import datetime
import sys

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
        teacher_model_name: str,
        student_model: BrainModel,
        tokenizer,
        device: torch.device,
        config: Dict,
        quantize: bool = True,
        cpu_offload: bool = True,
        use_cpu_only: bool = False
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = "cpu" if use_cpu_only else device
        self.config = config
        self.quantize = quantize and not use_cpu_only
        self.cpu_offload = cpu_offload and not use_cpu_only
        self.use_cpu_only = use_cpu_only
        self.windows_mode = config.get("windows_mode", False) or sys.platform.startswith('win')
        self.use_direct_gpu = config.get("use_direct_gpu", False)
        
        # PyTorch nightlyの検出
        is_nightly = False
        try:
            torch_version = torch.__version__
            if 'dev' in torch_version or 'nightly' in torch_version:
                is_nightly = True
                logger.info(f"PyTorch nightly版を検出: {torch_version}")
                self.use_direct_gpu = True  # 強制的に直接GPUモードを有効化
        except:
            pass
        
        # RAM使用状況を記録
        ram_usage = psutil.virtual_memory()
        logger.info(f"RAM使用状況: {ram_usage.percent}% (使用中: {ram_usage.used/1024**3:.1f}GB, 空き: {ram_usage.available/1024**3:.1f}GB)")
        
        logger.info(f"Loading teacher model: {teacher_model_name}")
        
        # 直接GPUアクセスモードの設定
        if self.use_direct_gpu:
            logger.info("GPU直接アクセスモードで実行します（最適化ライブラリ未使用）")
            
            # 環境変数を設定してQwen2の特殊機能を無効化
            os.environ["USE_FLASH_ATTENTION"] = "0"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # デバッグに役立つ場合がある
            
            # メモリの使用方法を設定
            torch.backends.cuda.matmul.allow_tf32 = True  # TF32を有効化（精度は下がるが高速）
            torch.backends.cudnn.benchmark = True  # 繰り返し同じサイズの計算に最適化
        
        # Windows環境の場合の設定
        if self.windows_mode:
            logger.info("Windows互換モードで実行しています")
            
            # 環境変数を設定してQwen2の特殊機能を無効化
            os.environ["USE_FLASH_ATTENTION"] = "0"
            
            # Windows環境ではxformersを使用可能か確認
            try:
                import xformers
                logger.info("xformersが使用可能です")
                os.environ["XFORMERS_ATTENTION"] = "1"  # xformers注意機構を有効化
            except ImportError:
                logger.warning("xformersが見つかりません。パフォーマンスが低下する可能性があります。")
                logger.info("pip install xformers>=0.0.20 でインストールを試みてください")
        
        # 量子化設定
        if self.quantize:
            logger.info("Configuring 4-bit quantization for teacher model")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
        else:
            bnb_config = None
        
        # デバイスマップ設定
        if self.use_cpu_only:
            device_map = "cpu"
            logger.info("Using CPU only mode")
        elif self.cpu_offload:
            device_map = "auto"
            logger.info("Using automatic device mapping with CPU offloading")
        else:
            device_map = {"": 0}  # すべてGPUに割り当て
        
        # 教師モデルの読み込み
        try:
            # モデル読み込み時のオプション
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch.float16 if not self.use_cpu_only else torch.float32,
                "use_cache": True
            }
            
            # Nightly版/直接GPUモード向けのカスタム設定
            if self.use_direct_gpu and not self.use_cpu_only:
                # 最も基本的なGPU設定
                model_kwargs = {
                    "torch_dtype": torch.float16,
                    "low_cpu_mem_usage": True,
                    "use_cache": True
                }
                
                # 環境変数設定
                os.environ["USE_FLASH_ATTENTION"] = "0"
                
                # device_mapを使わずに後でGPUに移動
                if "device_map" in model_kwargs:
                    del model_kwargs["device_map"]
            
            # 量子化設定を追加（CPU専用モードでは不要）
            if not self.use_cpu_only and self.quantize and not self.use_direct_gpu:
                model_kwargs["quantization_config"] = bnb_config
            
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name, 
                **model_kwargs
            )
            
            # 直接GPUモードの場合は手動でGPUに移動
            if self.use_direct_gpu and not self.use_cpu_only:
                logger.info(f"モデルを{device}に手動で移動します")
                self.teacher_model = self.teacher_model.to(device)
            
            # メモリ効率化のためのグラデーション計算無効化
            for param in self.teacher_model.parameters():
                param.requires_grad = False
                
            self.teacher_model.eval()  # 評価モードに設定
            logger.info("Teacher model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            
            # 特定のエラーパターンを検出して対処法を提案
            error_str = str(e).lower()
            
            if "no module named 'triton'" in error_str:
                logger.error("tritonモジュールが見つかりません。PyTorch nightlyを使用している可能性があります。")
                logger.info("--use_direct_gpu オプションを使用するか、別のモデルを選択してください。")
                
                # 代替モデルの提案
                alternative_models = [
                    "elyza/elyza-japanese-llama-2-7b",
                    "stabilityai/stablelm-base-alpha-7b",
                    "cyberagent/calm2-7b"
                ]
                
                logger.info("以下のモデルはWindows環境でも動作します:")
                for i, model in enumerate(alternative_models):
                    logger.info(f"{i+1}. {model}")
                
                choice = input("代替モデルを使用しますか？ 番号を入力するか、n で終了: ")
                if choice.isdigit() and 1 <= int(choice) <= len(alternative_models):
                    self.teacher_model_name = alternative_models[int(choice)-1]
                    logger.info(f"代替モデル {self.teacher_model_name} を使用します")
                    
                    # 再帰的に初期化を試みる（無限ループ防止のため1回のみ）
                    return KnowledgeDistiller(
                        teacher_model_name=self.teacher_model_name,
                        student_model=student_model,
                        tokenizer=tokenizer,
                        device=device,
                        config=config,
                        quantize=quantize,
                        cpu_offload=cpu_offload,
                        use_cpu_only=use_cpu_only
                    )
            
            # その他の一般的なエラー
            raise
        
        # テンプレート
        self.thinking_template = "以下の質問について考えていきます。\n\n質問: {question}\n\n思考過程:"
        
        # データキャッシュ（RAM活用）
        self.data_cache = {}
        self.max_cache_size = 1000  # 最大キャッシュサイズ
    
    def generate_questions_if_needed(self, output_file: str, num_samples: int = 1000):
        """質問ファイルが存在しない場合、自動生成する"""
        if os.path.exists(output_file):
            logger.info(f"Using existing questions file: {output_file}")
            return output_file
        
        logger.info(f"Generating {num_samples} questions for distillation")
        
        # 様々なトピックのリスト
        topics = [
            "科学", "技術", "歴史", "文化", "芸術", "哲学", "心理学", "経済", "政治", "言語",
            "教育", "環境", "健康", "宇宙", "生物学", "物理学", "数学", "文学", "音楽", "映画",
            "スポーツ", "旅行", "料理", "ファッション", "宗教", "神話", "ビジネス", "社会問題",
            "倫理", "テクノロジー", "人工知能", "インターネット", "プログラミング", "デザイン", "電気電子", "高校", "妹"
        ]
        
        # 質問の種類
        question_types = [
            "{}について説明してください。",
            "{}の歴史について教えてください。",
            "{}の重要性とは何ですか？",
            "{}の将来はどうなると思いますか？",
            "{}に関する最近の進展は？",
            "{}の利点と欠点を教えてください。",
            "{}と{}の関係について説明できますか？",
            "{}はどのように{}に影響を与えていますか？",
            "なぜ{}が重要なのですか？",
            "どうすれば{}をより良く理解できますか？",
            "{}の主な課題は何ですか？",
            "{}と{}を比較してください。",
            "{}の基本原則は何ですか？",
            "{}の実例を挙げてください。",
            "{}に取り組む最良の方法は？"
        ]
        
        questions = []
        
        # 基本的な質問を生成
        for i in range(num_samples):
            if i % 100 == 0:
                logger.info(f"Generated {i} questions")
                
            # 単一トピック質問
            if i % 3 != 0:  # 2/3の質問は単一トピック
                topic = np.random.choice(topics)
                q_type = np.random.choice([t for t in question_types if t.count('{}') == 1])
                question = q_type.format(topic)
            # 複数トピック質問
            else:
                topic1 = np.random.choice(topics)
                topic2 = np.random.choice([t for t in topics if t != topic1])
                q_type = np.random.choice([t for t in question_types if t.count('{}') == 2])
                question = q_type.format(topic1, topic2)
            
            questions.append(question)
        
        # ファイルに保存
        with open(output_file, 'w', encoding='utf-8') as f:
            for q in questions:
                f.write(q + '\n')
        
        logger.info(f"Generated questions saved to {output_file}")
        return output_file

    def prepare_distillation_data(self, questions_file: str, output_file: str, num_samples: int = 100, batch_size: int = 4, cache_to_ram: bool = True):
        """教師モデルからの出力を生成してデータを準備（バッチ処理対応）"""
        logger.info(f"Preparing distillation data from {questions_file}")
        
        # 質問ファイルが存在しない場合は自動生成
        if not os.path.exists(questions_file):
            logger.info(f"Questions file {questions_file} not found, generating automatically")
            questions_file = self.generate_questions_if_needed(questions_file, num_samples)
        
        # 質問を読み込む
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]
        
        # サンプル数を制限
        questions = questions[:num_samples]
        
        distillation_data = []
        
        # PyTorch nightly版を検出した場合の特別な処理
        is_nightly = False
        try:
            torch_version = torch.__version__
            if 'dev' in torch_version or 'nightly' in torch_version:
                is_nightly = True
                logger.info(f"Using PyTorch nightly version {torch_version} for data generation")
        except:
            pass
        
        # バッチ処理で効率化
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{math.ceil(len(questions)/batch_size)}")
            
            batch_prompts = [self.thinking_template.format(question=q) for q in batch_questions]
            batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # 教師モデルの出力を生成
            with torch.no_grad():
                # PyTorch nightly版/直接GPU使用モードの場合の特別な設定
                generate_kwargs = {
                    "input_ids": batch_inputs.input_ids,
                    "attention_mask": batch_inputs.attention_mask,
                    "max_length": 768,  # より長い出力を許可
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_return_sequences": 1,
                    "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
                }
                
                # PyTorch nightly版での最適化
                if is_nightly or self.use_direct_gpu:
                    # use_cacheをオンに
                    generate_kwargs["use_cache"] = True
                    
                    # メモリ効率モード
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            outputs = self.teacher_model.generate(**generate_kwargs)
                    else:
                        outputs = self.teacher_model.generate(**generate_kwargs)
                else:
                    outputs = self.teacher_model.generate(**generate_kwargs)
                
                # 出力をデコード
                for j, output in enumerate(outputs):
                    teacher_output = self.tokenizer.decode(output, skip_special_tokens=True)
                    
                    item = {
                        "input": batch_questions[j % len(batch_questions)],
                        "output": teacher_output
                    }
                    
                    distillation_data.append(item)
                    
                    # RAMキャッシュに保存（オプション）
                    if cache_to_ram and len(self.data_cache) < self.max_cache_size:
                        cache_key = f"item_{len(self.data_cache)}"
                        self.data_cache[cache_key] = item
                
                # バッチ処理後にメモリを解放
                del outputs, batch_inputs
                torch.cuda.empty_cache()
                gc.collect()
                
                # 定期的に進捗を保存（万が一のクラッシュに備える）
                if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(questions):
                    temp_output_file = output_file + f".temp_{i+batch_size}"
                    with open(temp_output_file, 'w', encoding='utf-8') as f:
                        json.dump(distillation_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"Intermediate data saved to {temp_output_file}")
        
        # 結果を保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(distillation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Distillation data saved to {output_file}")
        
        # RAM使用状況を記録
        ram_usage = psutil.virtual_memory()
        logger.info(f"RAM使用状況: {ram_usage.percent}% (使用中: {ram_usage.used/1024**3:.1f}GB, 空き: {ram_usage.available/1024**3:.1f}GB)")
        
        return output_file

    def distill(self, train_data_path, val_data_path, output_dir, batch_size=4, num_epochs=3,
                gradient_accumulation_steps=8, use_ram_cache=True, checkpoint_every=500, config=None):
        """知識蒸留の実行"""
        logger.info("Starting knowledge distillation process")
        
        # データセットの準備
        train_dataset = DistillationDataset(train_data_path, self.tokenizer)
        val_dataset = DistillationDataset(val_data_path, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 最適化設定
        optimizer = optim.AdamW(self.student_model.parameters(), lr=5e-5)
        scheduler = None  # スケジューラを使用する場合は設定
        
        # トレーニングループ
        best_val_loss = float('inf')
        best_model_state = None
        best_epoch = 0
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # トレーニング
            self.student_model.train()
            train_loss = 0.0
            
            for step, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                train_loss += loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # チェックポイントの保存
                if (step + 1) % checkpoint_every == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step+1}.pt")
                    torch.save({
                        'step': step + 1,
                        'model_state_dict': self.student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'train_loss': train_loss / (step + 1),
                        'model_config': config,  # 設定情報を保存
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved at {checkpoint_path}")
            
            # 検証
            self.student_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
            
            val_loss /= len(val_loader)
            logger.info(f"Validation loss: {val_loss}")
            
            # ベストモデルの更新
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.student_model.state_dict()
                best_epoch = epoch + 1
            
            # モデルの保存
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'model_config': config,  # 設定情報を保存
            }, os.path.join(output_dir, f'brain_model_epoch_{epoch+1}.pt'))
        
        # ベストモデルの保存
        best_model_path = os.path.join(output_dir, "best_brain_model.pt")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'val_loss': best_val_loss,
            'model_config': config,  # 設定情報を保存
        }, best_model_path)
        logger.info(f"Best model saved at {best_model_path}")
