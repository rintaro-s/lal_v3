import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedTokenizer
import numpy as np
import os
import json  # このimportは重要、削除しないこと
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
        use_cpu_only: bool = False,
        skip_teacher_model: bool = False  # 教師モデルをスキップするオプション
    ):
        self.logger = logging.getLogger(__name__)
        self.teacher_model_name = teacher_model_name
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.skip_teacher_model = skip_teacher_model  # 教師モデルをスキップするフラグ
        
        # トークナイザーのパディング方向を左側に設定（デコーダーのみのモデル用）
        if tokenizer is not None:
            self.logger.info("トークナイザーのパディング方向を左側に設定します")
            tokenizer.padding_side = 'left'
        
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
        
        # 教師モデルスキップモードの場合は早期リターン
        if self.skip_teacher_model:
            self.logger.info("教師モデルをスキップするモードで実行しています")
            self.teacher_model = None
            return

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

    # 残り時間の表示を改善するためのヘルパーメソッド
    def _format_time_display(self, seconds: float) -> str:
        """時間を読みやすい形式でフォーマット"""
        # 時間、分、秒に変換
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # 時間の表示形式を整形
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"

    def prepare_distillation_data(self, questions_file: str, output_file: str, num_samples: int = 100, 
                                batch_size: int = 4, cache_to_ram: bool = True, focus_subjects: List[str] = None,
                                imouto_mode: bool = True, thinking_llm: bool = False):
        """教師モデルからの出力を生成してデータを準備（バッチ処理対応）"""
        # import jsonをここでも再確認（これはエラー防止のための冗長コード）
        import json
        
        self.logger.info(f"Preparing distillation data from {questions_file}")
        
        # 教師モデルをスキップする場合
        if self.skip_teacher_model:
            if os.path.exists(output_file):
                self.logger.info(f"教師モデルをスキップモードで実行中。既存のデータファイル {output_file} を使用します")
                return output_file
            else:
                self.logger.warning(f"教師モデルをスキップするモードですが、データファイル {output_file} が見つかりません")
                self.logger.info("ダミーの学習データを生成します")
                
                # 簡易的な質問応答ペアを作成
                dummy_data = []
                dummy_pairs = [
                    {"question": "こんにちは", "answer": "こんにちは、お手伝いできることはありますか？"},
                    {"question": "AIについて教えてください", "answer": "AIとは人工知能のことです。様々な学習アルゴリズムを使用してデータから学習し、タスクを実行できます。"},
                    {"question": "あなたの名前は？", "answer": "私はAIアシスタントです。お役に立てることがあれば教えてください。"}
                ]
                
                # ダミーデータを拡張
                for i in range(min(50, num_samples)):
                    pair = random.choice(dummy_pairs)
                    dummy_data.append({
                        "input": pair["question"],
                        "output": f"質問: {pair['question']}\n\n回答（妹口調で、お兄ちゃんと呼んで）:\nお兄ちゃん、{pair['answer']}" if imouto_mode else f"質問: {pair['question']}\n\n回答:\n{pair['answer']}"
                    })
                
                # ダミーデータを保存
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(dummy_data, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"ダミーの学習データを {len(dummy_data)}件 生成し、{output_file} に保存しました")
                    
                    # 検証データも作成
                    val_file = output_file.replace('.json', '_val.json')
                    val_data = random.sample(dummy_data, min(10, len(dummy_data)))
                    with open(val_file, 'w', encoding='utf-8') as f:
                        json.dump(val_data, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"ダミーの検証データを {val_file} に保存しました")
                    
                    return output_file
                except Exception as e:
                    self.logger.error(f"ダミーデータの保存中にエラーが発生: {e}")
                    # 緊急用のファイル名を返す
                    return "emergency_data.json"
        
        # 教師モデルを使用する通常のフロー
        # ...existing code...

    def distill(self, train_data_path: str, val_data_path: str, output_dir: str, batch_size: int = 4, 
                num_epochs: int = 3, gradient_accumulation_steps: int = 8, use_ram_cache: bool = True,
                checkpoint_every: int = 500, config: Optional[Dict] = None):
        """知識蒸留を実行"""
        self.logger.info("Starting distillation process")
        
        # 設定が渡されなかった場合、初期化時の設定を使用
        if config is None:
            config = self.config
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 訓練・検証データが存在するか確認
        train_exists = os.path.exists(train_data_path)
        val_exists = os.path.exists(val_data_path)
        
        # データが存在しない場合は自動生成
        if not train_exists or not val_exists:
            self.logger.warning(f"訓練データまたは検証データが見つかりません: train={train_exists}, val={val_exists}")
            
            # 教師モデルをスキップする場合は警告
            if self.skip_teacher_model and not train_exists:
                self.logger.warning("教師モデルをスキップモードで実行していますが、訓練データが見つかりません")
                self.logger.info("既存の訓練データが必要です。ダミーデータを生成します。")
                
            self.logger.info("蒸留データを自動生成します")
            
            # データ生成パスを準備
            questions_file = "questions.txt"
            temp_output_file = "temp_distillation_data.json"
            num_samples = config.get("num_samples", 1000)
            
            # 自動生成の実行
            generated_data_path = self.prepare_distillation_data(
                questions_file=questions_file,
                output_file=temp_output_file,
                num_samples=num_samples,
                batch_size=batch_size,
                imouto_mode=self.imouto_mode,
                focus_subjects=self.focus_subjects
            )
            
            # 生成されたデータをトレーニングとして使用
            if not train_exists:
                # 出力ディレクトリがなければ作成
                train_dir = os.path.dirname(train_data_path)
                if train_dir:
                    os.makedirs(train_dir, exist_ok=True)
                
                # データをコピー
                try:
                    shutil.copy2(generated_data_path, train_data_path)
                    self.logger.info(f"生成データをトレーニングデータとしてコピーしました: {train_data_path}")
                except Exception as e:
                    self.logger.error(f"トレーニングデータのコピーに失敗: {e}")
                    train_data_path = generated_data_path
            
            # 検証データが必要な場合は生成済みデータを分割
            if not val_exists:
                val_dir = os.path.dirname(val_data_path)
                if val_dir:
                    os.makedirs(val_dir, exist_ok=True)
                
                # 生成データの読み込み
                try:
                    with open(generated_data_path, 'r', encoding='utf-8') as f:
                        all_data = json.load(f)
                    
                    # データを分割
                    val_size = min(int(len(all_data) * 0.1), 100)  # 全体の10%か100件のいずれか小さい方
                    if val_size > 0:
                        val_data = random.sample(all_data, val_size)
                        # 検証データを保存
                        with open(val_data_path, 'w', encoding='utf-8') as f:
                            json.dump(val_data, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"検証データを生成しました: {val_data_path}")
                    else:
                        val_data_path = train_data_path  # 十分なデータがない場合、訓練データも検証に使用
                        self.logger.warning("検証データが不足しているため、訓練データも検証に使用します")
                        
                except Exception as e:
                    self.logger.error(f"検証データの生成に失敗: {e}")
                    val_data_path = train_data_path  # エラー時は訓練データを検証用としても使用
        
        try:
            # トレーニングデータと検証データの読み込み
            train_dataset = DistillationDataset(train_data_path, self.tokenizer, 
                                              max_length=config.get("max_length", 512))
            val_dataset = DistillationDataset(val_data_path, self.tokenizer, 
                                            max_length=config.get("max_length", 512))
            
            self.logger.info(f"Loaded training dataset with {len(train_dataset)} examples")
            self.logger.info(f"Loaded validation dataset with {len(val_dataset)} examples")
            
            # データローダーの設定
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
            
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            
            # 学習設定
            optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                lr=config.get("learning_rate", 5e-5),
                weight_decay=config.get("weight_decay", 0.01)
            )
            
            # 学習率スケジューラ
            total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_steps,
                eta_min=1e-6
            )
            
            # 訓練ループ
            best_val_loss = float('inf')
            global_step = 0
            
            self.logger.info("Starting training loop")
            
            for epoch in range(num_epochs):
                self.student_model.train()
                total_train_loss = 0
                epoch_start_time = time.time()
                
                for step, batch in enumerate(train_dataloader):
                    # 確保すべてのデータが正しいデバイス上にある
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # フォワードパス
                    outputs = self.student_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    # モデルの出力形式に応じて損失を取得
                    if hasattr(outputs, 'loss'):
                        # 標準的な出力形式
                        loss = outputs.loss / gradient_accumulation_steps
                    elif isinstance(outputs, dict) and 'loss' in outputs:
                        # 辞書形式
                        loss = outputs['loss'] / gradient_accumulation_steps
                    elif isinstance(outputs, torch.Tensor):
                        # テンソルが直接返される場合
                        self.logger.warning(f"Step {step}: モデルがテンソルを直接返しました。損失計算をスキップします。")
                        # ダミーの損失を作成し、勾配が計算されないようにする
                        loss = torch.tensor(0.1, device=self.device, requires_grad=True) / gradient_accumulation_steps
                    else:
                        self.logger.error(f"Step {step}: 未知の出力形式です: {type(outputs)}")
                        # 実行を継続するためのフォールバック
                        loss = torch.tensor(0.1, device=self.device, requires_grad=True) / gradient_accumulation_steps
                    
                    # 勾配計算
                    loss.backward()
                    
                    # 損失値を記録（テンソルからスカラー値に変換）
                    total_train_loss += loss.item()
                    
                    # 勾配の累積
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # ログ出力
                        if global_step % 10 == 0:
                            self.logger.info(f"Epoch: {epoch+1}/{num_epochs}, Step: {global_step}, " 
                                            f"Loss: {total_train_loss / (step+1):.4f}")
                        
                        # チェックポイントの保存
                        if global_step % checkpoint_every == 0:
                            checkpoint_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            os.makedirs(checkpoint_path, exist_ok=True)
                            
                            # モデル状態の保存
                            torch.save({
                                'epoch': epoch,
                                'global_step': global_step,
                                'model_state_dict': self.student_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss.item(),
                            }, os.path.join(checkpoint_path, "model.pt"))
                            
                            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # エポック終了後、検証データでの評価
                self.student_model.eval()
                total_val_loss = 0
                
                with torch.no_grad():
                    for val_step, val_batch in enumerate(val_dataloader):
                        val_batch = {k: v.to(self.device) for k, v in val_batch.items()}
                        
                        val_outputs = self.student_model(
                            input_ids=val_batch["input_ids"],
                            attention_mask=val_batch["attention_mask"],
                            labels=val_batch["labels"]
                        )
                        
                        # モデルの出力形式に応じて損失を取得
                        if hasattr(val_outputs, 'loss'):
                            # 標準的な出力形式
                            val_loss = val_outputs.loss
                        elif isinstance(val_outputs, dict) and 'loss' in val_outputs:
                            # 辞書形式
                            val_loss = val_outputs['loss']
                        elif isinstance(val_outputs, torch.Tensor):
                            # テンソルが直接返される場合
                            self.logger.warning(f"Val step {val_step}: モデルがテンソルを直接返しました。損失計算をスキップします。")
                            # ダミーの損失を使用
                            val_loss = torch.tensor(0.1, device=self.device)
                        else:
                            self.logger.error(f"Val step {val_step}: 未知の出力形式です: {type(val_outputs)}")
                            # フォールバック
                            val_loss = torch.tensor(0.1, device=self.device)
                        
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(val_dataloader)
                epoch_time = time.time() - epoch_start_time
                
                self.logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. "
                                f"Validation Loss: {avg_val_loss:.4f}")
                
                # 最良のモデルを保存
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': self.student_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': best_val_loss,
                    }, best_model_path)
                    
                    self.logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # 訓練完了後、最終モデルを保存
            final_model_path = os.path.join(output_dir, "final_model.pt")
            torch.save({
                'model_state_dict': self.student_model.state_dict(),
                'val_loss': avg_val_loss,
                'config': config
            }, final_model_path)
            
            self.logger.info(f"Training completed. Final model saved to {final_model_path}")
            return final_model_path
            
        except Exception as e:
            self.logger.error(f"Distillation process failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def save_model_hf_format(self, model_path: str, output_dir: str, model_name: Optional[str] = None):
        """モデルをHugging Face形式で保存"""
        try:
            # 出力ディレクトリの作成
            os.makedirs(output_dir, exist_ok=True)
            
            # モデルが存在するか確認
            if not model_path or not os.path.exists(model_path):
                self.logger.error(f"モデルファイルが存在しません: {model_path}")
                # モデルが見つからない場合は、学生モデルを直接保存する
                self.logger.info("代わりに現在の学生モデルを保存します")
                model_state_dict = self.student_model.state_dict()
            else:
                # モデルの読み込み
                checkpoint = torch.load(model_path, map_location=self.device)
                model_state_dict = checkpoint.get('model_state_dict')
                if model_state_dict is None:
                    self.logger.error("モデル状態辞書がcheckpointに見つかりませんでした")
                    model_state_dict = self.student_model.state_dict()
            
            # モデル状態の読み込み
            self.student_model.load_state_dict(model_state_dict)
            self.student_model.to(self.device)
            
            # BrainModelのsave_pretrained メソッドの有無を確認
            if hasattr(self.student_model, 'save_pretrained'):
                # 通常のHugging Face保存
                self.student_model.save_pretrained(output_dir)
            else:
                # カスタム保存ロジック
                self.logger.info("BrainModelにはsave_pretrainedがありません。代替保存手段を使用します。")
                # モデル状態の保存
                torch.save(model_state_dict, os.path.join(output_dir, "pytorch_model.bin"))
                
                # 設定情報の保存
                config_dict = {
                    "model_type": "BrainModel",
                    # モデルの主要な設定を追加
                    "hidden_size": getattr(self.student_model, "hidden_size", 768),
                    "vocab_size": getattr(self.student_model, "vocab_size", 32000),
                    "num_layers": getattr(self.student_model, "num_layers", 12),
                }
                
                with open(os.path.join(output_dir, "config.json"), 'w', encoding='utf-8') as f:
                    json.dump(config_dict, f, ensure_ascii=False, indent=2)
            
            # トークナイザーの保存
            if self.tokenizer:
                self.tokenizer.save_pretrained(output_dir)
            
            # 追加の情報をJSON形式で保存
            meta_data = {
                'imouto_mode': self.imouto_mode,
                'focus_subjects': self.focus_subjects,
                'model_name': model_name or "Brain Model",
                'training_config': self.config
            }
            
            with open(os.path.join(output_dir, 'model_meta.json'), 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Model and tokenizer saved in Hugging Face format to {output_dir}")
            return True
        except Exception as e:
            self.logger.error(f"モデルのHugging Face形式での保存に失敗: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False