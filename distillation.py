import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import numpy as np
import os
import json
import gc
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging
import math
import psutil
from datetime import datetime

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
        cpu_offload: bool = True
    ):
        self.teacher_model_name = teacher_model_name
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.quantize = quantize
        self.cpu_offload = cpu_offload
        
        # RAM使用状況を記録
        ram_usage = psutil.virtual_memory()
        logger.info(f"RAM使用状況: {ram_usage.percent}% (使用中: {ram_usage.used/1024**3:.1f}GB, 空き: {ram_usage.available/1024**3:.1f}GB)")
        
        logger.info(f"Loading teacher model: {teacher_model_name}")
        
        # 量子化設定
        if quantize:
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
        if cpu_offload:
            device_map = "auto"
            logger.info("Using automatic device mapping with CPU offloading")
        else:
            device_map = {"": 0}  # すべてGPUに割り当て
        
        # 教師モデルの読み込み
        try:
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_name, 
                device_map=device_map,
                torch_dtype=torch.float16,
                quantization_config=bnb_config if quantize else None,
            )
            
            # メモリ効率化のためのグラデーション計算無効化
            for param in self.teacher_model.parameters():
                param.requires_grad = False
                
            self.teacher_model.eval()  # 評価モードに設定
            logger.info("Teacher model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load teacher model: {e}")
            raise
        
        # テンプレート
        self.thinking_template = "以下の質問について考えていきます。\n\n質問: {question}\n\n思考過程:"
        
        # データキャッシュ（RAM活用）
        self.data_cache = {}
        self.max_cache_size = 1000  # 最大キャッシュサイズ
    
    def prepare_distillation_data(self, questions_file: str, output_file: str, num_samples: int = 100, batch_size: int = 4, cache_to_ram: bool = True):
        """教師モデルからの出力を生成してデータを準備（バッチ処理対応）"""
        logger.info(f"Preparing distillation data from {questions_file}")
        
        # 質問を読み込む
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f]
        
        # サンプル数を制限
        questions = questions[:num_samples]
        
        distillation_data = []
        
        # バッチ処理で効率化
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{math.ceil(len(questions)/batch_size)}")
            
            batch_prompts = [self.thinking_template.format(question=q) for q in batch_questions]
            batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # 教師モデルの出力を生成
            with torch.no_grad():
                outputs = self.teacher_model.generate(
                    input_ids=batch_inputs.input_ids,
                    attention_mask=batch_inputs.attention_mask,
                    max_length=768,  # より長い出力を許可
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
                )
            
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
    
    def distill(
        self,
        train_data_path: str,
        val_data_path: str,
        output_dir: str,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 8,
        checkpoint_every: int = 500,
        use_ram_cache: bool = True
    ):
        """知識蒸留の実行（メモリ最適化）"""
        logger.info("Starting knowledge distillation")
        
        # チェックポイントディレクトリの作成
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # データローダーの準備
        train_dataset = DistillationDataset(train_data_path, self.tokenizer)
        val_dataset = DistillationDataset(val_data_path, self.tokenizer)
        
        # RAMキャッシュの活用（オプション）
        if use_ram_cache and self.data_cache:
            logger.info(f"Using {len(self.data_cache)} items from RAM cache")
            # RAMキャッシュからデータを追加/置換
            # 実装は省略
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # より多くのワーカー（RAM活用）
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # オプティマイザとスケジューラの設定
        optimizer = optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='linear'
        )
        
        # モデルをデバイスに移動
        self.student_model.to(self.device)
        
        # トレーニングループ
        best_val_loss = float('inf')
        global_step = 0
        
        # 学習開始時間を記録（推定完了時間の計算用）
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # トレーニングフェーズ
            self.student_model.train()
            train_loss = 0.0
            optimizer.zero_grad()  # 最適化ステップ前にゼロ勾配
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                # バッチデータをデバイスに移動
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 教師モデルの出力を取得
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    teacher_logits = teacher_outputs.logits
                
                # 生徒モデルのフォワードパス
                student_outputs = self.student_model(input_ids)
                
                # 損失計算
                # 1. 教師からの蒸留損失（KL Divergence）
                distillation_loss = self._compute_distillation_loss(
                    student_outputs, 
                    teacher_logits,
                    attention_mask,
                    temperature=2.0
                )
                
                # 2. 言語モデリング損失
                lm_loss = F.cross_entropy(
                    student_outputs.view(-1, student_outputs.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # 損失の重み付け
                alpha = 0.5  # 蒸留の重み
                loss = alpha * distillation_loss + (1 - alpha) * lm_loss
                
                # 勾配累積による大バッチ効果
                scaled_loss = loss / gradient_accumulation_steps
                scaled_loss.backward()
                
                # 勾配累積ステップが完了したら最適化
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # 定期的にメモリ解放
                    if global_step % 10 == 0:
                        torch.cuda.empty_cache()
                
                train_loss += loss.item()
                
                # チェックポイント保存（定期的）
                if global_step > 0 and global_step % checkpoint_every == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': self.student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                    }, checkpoint_path)
                    logger.info(f"Checkpoint saved at step {global_step}")
                    
                    # 進捗と推定完了時間の計算
                    elapsed = datetime.now() - start_time
                    progress = global_step / total_steps
                    if progress > 0:
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                        estimated_completion = datetime.now() + remaining
                        logger.info(f"Progress: {progress*100:.1f}%. Estimated completion: {estimated_completion}")
            
            # エポック終了後のトレーニング損失
            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # 検証フェーズ
            self.student_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    
                    # 教師モデルの出力
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    teacher_logits = teacher_outputs.logits
                    
                    # 生徒モデルの出力
                    student_outputs = self.student_model(input_ids)
                    
                    # 損失計算
                    distillation_loss = self._compute_distillation_loss(
                        student_outputs, 
                        teacher_logits,
                        attention_mask,
                        temperature=2.0
                    )
                    
                    lm_loss = F.cross_entropy(
                        student_outputs.view(-1, student_outputs.size(-1)),
                        labels.view(-1),
                        ignore_index=-100
                    )
                    
                    loss = alpha * distillation_loss + (1 - alpha) * lm_loss
                    val_loss += loss.item()
            
            # 検証損失の評価
            avg_val_loss = val_loss / len(val_loader)
            logger.info(f"Validation loss: {avg_val_loss:.4f}")
            
            # モデルの保存（改善があった場合）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                
                # ディレクトリが存在しない場合は作成
                os.makedirs(output_dir, exist_ok=True)
                
                # モデルの保存
                model_path = os.path.join(output_dir, f"brain_model_epoch_{epoch+1}.pt")
                torch.save(self.student_model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
                
                # 最もよいモデルへのシンボリックリンク
                best_model_path = os.path.join(output_dir, "brain_model_best.pt")
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
                torch.save(self.student_model.state_dict(), best_model_path)
    
    def _compute_distillation_loss(self, student_logits, teacher_logits, attention_mask, temperature=2.0):
        """知識蒸留損失（KL-divergence）の計算"""
        # Softmax with temperature
        soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
        log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL-divergence loss
        loss = -(soft_targets * log_probs).sum(dim=-1) * (temperature ** 2)
        
        # マスクを適用（パディングトークンは無視）
        mask = attention_mask.float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def generate_distillation_examples(self, num_examples=1000, output_file="distillation_data.json", batch_size=4):
        """教師モデルから蒸留用の例を生成（バッチ処理で効率化）"""
        logger.info(f"Generating {num_examples} distillation examples")
        
        # サンプル質問のテンプレート
        question_templates = [
            "{}とはなんですか？",
            "{}の特徴を教えてください",
            "{}と{}の違いは何ですか？",
            "{}はなぜ起こりますか？",
            "{}の歴史について教えてください",
            "{}をするための最適な方法は？",
            "{}についてどう思いますか？",
            "{}が社会に与える影響は？",
            "{}と{}の関係性について説明してください",
            "{}を解決するためのアプローチを考えてください"
        ]
        
        # トピックのリスト（質問生成用）
        topics = [
            "人工知能", "気候変動", "量子コンピュータ", "ブロックチェーン", "宇宙探査",
            "再生可能エネルギー", "機械学習", "自動運転", "サイバーセキュリティ", "ロボット工学",
            "バイオテクノロジー", "ナノテクノロジー", "仮想現実", "拡張現実", "インターネット",
            "経済学", "心理学", "哲学", "歴史", "物理学", "化学", "生物学", "数学", "芸術", "文学"
        ]
        
        # より多様なトピックを追加（ELYZA-Thinkingのスペックを継承するため）
        additional_topics = [
            "深層学習", "自然言語処理", "強化学習", "進化計算", "ニューラルネットワーク",
            "認知科学", "情報理論", "暗号通貨", "量子暗号", "高速演算",
            "脳科学", "認知バイアス", "意思決定理論", "意識", "人間の知性",
            "創造性", "論理学", "倫理学", "メタ認知", "統計学", "確率論",
            "システム思考", "複雑系", "創発現象", "自己組織化", "カオス理論"
        ]
        
        topics.extend(additional_topics)
        
        # 質問生成
        questions = []
        for _ in range(num_examples):
            template = np.random.choice(question_templates)
            
            if "{}" in template and template.count("{}") == 1:
                topic = np.random.choice(topics)
                question = template.format(topic)
            elif "{}" in template and template.count("{}") == 2:
                topic1, topic2 = np.random.choice(topics, size=2, replace=False)
                question = template.format(topic1, topic2)
            else:
                question = template
                
            questions.append(question)
        
        # バッチ処理用の準備
        distillation_data = []
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # 開始時間記録
        start_time = datetime.now()
        
        # バッチ処理
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{math.ceil(len(questions)/batch_size)}")
            
            batch_prompts = [self.thinking_template.format(question=q) for q in batch_questions]
            
            try:
                batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
                
                # 教師モデルの出力を生成
                with torch.no_grad():
                    outputs = self.teacher_model.generate(
                        input_ids=batch_inputs.input_ids,
                        attention_mask=batch_inputs.attention_mask,
                        max_length=768,  # より深い思考を促すため長めに設定
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
                    )
                
                # 出力をデコード
                for j, output in enumerate(outputs):
                    teacher_output = self.tokenizer.decode(output, skip_special_tokens=True)
                    
                    distillation_data.append({
                        "input": batch_questions[j % len(batch_questions)],
                        "output": teacher_output
                    })
                
                # メモリ解放
                del outputs, batch_inputs
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # エラーが発生した場合、バッチサイズを半分に縮小してリトライ
                if batch_size > 1:
                    logger.info("Retrying with smaller batch size")
                    half_batch = batch_size // 2
                    for j in range(0, len(batch_questions), half_batch):
                        sub_batch = batch_questions[j:j+half_batch]
                        # 小さいバッチで再処理（実際の実装はもっと複雑になります）
                        # この例では省略
            
            # 定期的に中間結果を保存
            if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(questions):
                temp_file = output_file + f".temp_{i+batch_size}"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(distillation_data, f, ensure_ascii=False, indent=2)
                
                # 進捗と時間見積もり
                elapsed = datetime.now() - start_time
                progress = (i + batch_size) / len(questions)
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    logger.info(f"Progress: {progress*100:.1f}%. Estimated remaining time: {remaining}")
        
        # 最終結果を保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(distillation_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Distillation examples saved to {output_file}")
        
        return output_file
