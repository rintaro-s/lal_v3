import torch
import time
import threading
import queue
import os
import logging
import gc
import json
from typing import List, Dict, Optional, Callable
from transformers import AutoTokenizer

from brain_model import BrainModel
from memory_system import MemorySystem
from real_time_thoughts import ThoughtGenerator, RealTimeOutputManager

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StreamingCallback:
    """ストリーミング出力用コールバック"""
    def __init__(self, output_func: Callable[[str], None] = print):
        self.output_func = output_func
    
    def __call__(self, text: str):
        self.output_func(text)

class InferenceEngine:
    """リアルタイム推論エンジン"""
    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        device: str = None,
        stream_output: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        self.stream_output = stream_output
        self.imouto_mode = False  # デフォルトはオフ
        
        # デバイスの設定
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Using device: {self.device}")
        
        # トークナイザーの読み込み - モデル読み込みの前に実行
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.logger.info(f"Tokenizer loaded successfully with vocab size: {len(self.tokenizer)}")
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer {tokenizer_name}: {e}")
        
        # モデルの読み込み - トークナイザーの後に実行
        self.logger.info(f"Loading model from {model_path}")
        try:
            if not self.load_model(model_path):
                raise RuntimeError("Model loading failed")
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
            
        # モデルの語彙サイズを設定 (必要な場合)
        if hasattr(self.model, 'resize_token_embeddings'):
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        # メモリシステムの初期化
        try:
            self.memory_system = MemorySystem(embedding_dim=self.model.hidden_size)
        except Exception as e:
            self.logger.error(f"Error initializing memory system: {e}")
            self.logger.warning("Will continue without memory system")
            self.memory_system = None
        
        # 思考生成器と出力マネージャー
        self.thought_generator = ThoughtGenerator(self.model, self.tokenizer, self.device)
        self.output_manager = RealTimeOutputManager(self.thought_generator)
        
        # 推論状態
        self.is_generating = False
        self.output_thread = None
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()
    
    def load_model(self, model_path: str):
        """モデルを読み込む関数"""
        try:
            # 新しいBrainModel.from_pretrainedメソッドを使用
            self.logger.info(f"Loading model from {model_path}")
            # トークナイザーの語彙サイズを明示的に渡す
            vocab_size = len(self.tokenizer) if hasattr(self, 'tokenizer') and self.tokenizer is not None else None
            
            # 語彙サイズが不一致でもロードできるように
            self.model = BrainModel.from_pretrained(
                model_path,
                vocab_size=vocab_size
            )
            
            self.model.to(self.device)
            self.model.eval()
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def _output_worker(self, callback: StreamingCallback):
        """出力ワーカースレッド"""
        while not self.stop_event.is_set() or not self.output_queue.empty():
            try:
                # キューから次のテキストチャンクを取得
                if not self.output_queue.empty():
                    text_chunk = self.output_queue.get(block=False)
                    # コールバックで出力
                    callback(text_chunk)
                    self.output_queue.task_done()
                else:
                    # キューが空の場合は短い待機
                    time.sleep(0.01)
            except queue.Empty:
                # キューが空の場合も短い待機
                time.sleep(0.01)
    
    def generate_response(self, input_text: str, callback: Optional[StreamingCallback] = None):
        """入力テキストに対する応答を生成"""
        if callback is None:
            callback = StreamingCallback()
        
        # 妹口調モードが有効なら、プロンプトを適宜修正
        if self.imouto_mode and "お兄ちゃん" not in input_text:
            # ユーザー入力の場合のみ修正（チャット履歴には適用しない）
            if input_text.startswith("質問:") or not ("あなた:" in input_text or "AI:" in input_text):
                input_text = f"質問: {input_text}\n回答（妹口調で、お兄ちゃんと呼んで）: "
            # チャット履歴の場合は末尾だけを修正
            elif input_text.endswith("AI: "):
                input_text = input_text.rstrip("AI: ") + "妹AI: "
        
        # すでに生成中の場合は終了
        if self.is_generating:
            logger.warning("Already generating a response")
            return
        
        self.is_generating = True
        self.stop_event.clear()
        
        # 出力スレッドの開始
        if self.stream_output:
            self.output_thread = threading.Thread(
                target=self._output_worker,
                args=(callback,)
            )
            self.output_thread.start()
        
        # 思考生成プロセスを開始
        self.thought_generator.start_thinking(input_text)
        
        # 最初の「考え中」表示
        if self.stream_output:
            thinking_indicator = self.output_manager.add_thinking_indicator()
            self.output_queue.put(thinking_indicator + " ")
        
        # リアルタイム出力ループ
        full_response = []
        try:
            # リアルタイム思考と出力
            while self.thought_generator.is_still_thinking():
                # 次の思考チャンクを出力に変換
                output_chunk = self.output_manager.process_thoughts_to_output()
                
                if output_chunk:
                    full_response.append(output_chunk)
                    
                    # ストリーミング出力
                    if self.stream_output:
                        self.output_queue.put(output_chunk + " ")
                
                # 遅延（人間らしいタイミング）
                time.sleep(0.05)
                
                # ランダムに「考え中」表示を挿入
                if self.stream_output and self.thought_generator.is_still_thinking() and np.random.random() < 0.03:
                    thinking_indicator = self.output_manager.add_thinking_indicator()
                    self.output_queue.put("\n" + thinking_indicator + " ")
        
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
        
        finally:
            # 最終レスポンスを取得
            final_response = " ".join(full_response)
            
            # 記憶システムに保存
            input_embedding = self._get_embedding(input_text)
            output_embedding = self._get_embedding(final_response)
            combined_embedding = (input_embedding + output_embedding) / 2
            
            importance = 0.7  # デフォルトの重要度
            if self.memory_system:
                self.memory_system.process_input(combined_embedding, importance, tag="conversation")
            
            # 終了処理
            if self.stream_output:
                self.stop_event.set()
                self.output_thread.join()
            
            self.is_generating = False
            
            return final_response
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """テキストの埋め込みを取得"""
        with torch.no_grad():
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            embedded = self.model.embedding(input_ids)
            return embedded.mean(dim=1).detach()
    
    def generate_response_sync(self, input_text: str) -> str:
        """同期的に応答を生成（ストリーミングなし）"""
        # 一時的にストリーミングを無効化
        original_stream = self.stream_output
        self.stream_output = False
        
        # 応答を生成
        response = self.generate_response(input_text)
        
        # 元の設定に戻す
        self.stream_output = original_stream
        
        return response
    
    def maintenance(self):
        """メンテナンス処理 - メモリ解放など"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
