import torch
import time
import threading
import queue
import os
import logging
import gc
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
        
        # デバイスの設定
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Using device: {self.device}")
        
        # モデルの読み込み
        self.logger.info(f"Loading model from {model_path}")
        try:
            # Hugging Faceモデルの場合
            if not os.path.exists(model_path) and '/' in model_path:
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(repo_id=model_path)
                self.logger.info(f"Downloaded model from Hugging Face to {model_path}")
            
            if os.path.isdir(model_path):
                # Hugging Face形式のモデル
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path)
                self.model = BrainModel(
                    vocab_size=config.vocab_size,
                    embedding_dim=config.hidden_size,
                    hidden_dim=config.intermediate_size,
                    memory_size=1000  # デフォルト値
                )
                self.model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
            else:
                # 通常のPyTorch形式モデル
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # モデルの初期化に必要な情報を取得
                self.model = BrainModel(
                    vocab_size=None,  # トークナイザーから取得するため後で設定
                    embedding_dim=768,
                    hidden_dim=1024,
                    memory_size=1000
                )
                
                # モデルの状態を読み込む
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")
        
        # トークナイザーの読み込み
        self.logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # モデルの語彙サイズを設定 (必要な場合)
            if hasattr(self.model, 'resize_token_embeddings'):
                self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {e}")
            raise RuntimeError(f"Failed to load tokenizer {tokenizer_name}: {e}")
        
        # モデルをデバイスに移動
        self.model.to(self.device)
        self.model.eval()
        
        # メモリシステムの初期化
        try:
            self.memory_system = MemorySystem(embedding_dim=768)
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
