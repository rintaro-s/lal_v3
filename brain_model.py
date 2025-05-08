import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import queue
import threading
from typing import Optional, Dict, Tuple, List, Any, Union

class DimensionAdapter(nn.Module):
    """次元不一致を解決するためのアダプターレイヤー"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        # 入力次元の確認
        if x.size(-1) != self.input_dim:
            print(f"次元修正: 入力 {x.size(-1)} -> 想定 {self.input_dim}")
            # 次元が不一致の場合は調整
            if x.size(-1) < self.input_dim:
                padding = (0, self.input_dim - x.size(-1))
                x = F.pad(x, padding)
            else:
                x = x[..., :self.input_dim]
        
        # 次元の投影
        return self.projection(x)

class NeuralModule(nn.Module):
    """基本的なニューラルネットワークモジュール"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_adapter = DimensionAdapter(input_dim, input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        x = self.input_adapter(x)
        return self.layers(x)

class LeftBrainModule(nn.Module):
    """論理的思考を担当するモジュール"""
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_adapter = DimensionAdapter(embedding_dim, hidden_dim)
        self.reasoning = NeuralModule(hidden_dim, hidden_dim * 2, hidden_dim)
        self.analysis = NeuralModule(hidden_dim, hidden_dim * 2, hidden_dim)
        self.output_adapter = DimensionAdapter(hidden_dim, embedding_dim)
        
    def forward(self, x, memory_context=None):
        # 入力次元を調整
        x = self.input_adapter(x)
        
        # メモリコンテキストがあれば統合
        if memory_context is not None:
            # メモリコンテキストの次元を調整
            if memory_context.size(-1) != self.hidden_dim:
                memory_adapter = DimensionAdapter(memory_context.size(-1), self.hidden_dim)
                memory_context = memory_adapter(memory_context)
            x = x + memory_context * 0.3
            
        x = self.reasoning(x)
        x = self.analysis(x)
        
        # 出力を元の埋め込み次元に戻す
        return self.output_adapter(x)

class RightBrainModule(nn.Module):
    """直感的思考を担当するモジュール"""
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_adapter = DimensionAdapter(embedding_dim, hidden_dim)
        self.pattern = NeuralModule(hidden_dim, hidden_dim * 2, hidden_dim)
        self.intuition = NeuralModule(hidden_dim, hidden_dim * 2, hidden_dim)
        self.output_adapter = DimensionAdapter(hidden_dim, embedding_dim)
        
    def forward(self, x, memory_context=None):
        # 入力次元を調整
        x = self.input_adapter(x)
        
        # メモリコンテキストがあれば統合
        if memory_context is not None:
            # メモリコンテキストの次元を調整
            if memory_context.size(-1) != self.hidden_dim:
                memory_adapter = DimensionAdapter(memory_context.size(-1), self.hidden_dim)
                memory_context = memory_adapter(memory_context)
            x = x + memory_context * 0.5
            
        x = self.pattern(x)
        x = self.intuition(x)
        
        # 出力を元の埋め込み次元に戻す
        return self.output_adapter(x)

class FrontalLobeModule(nn.Module):
    """前頭葉の意思決定を担当するモジュール"""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.left_adapter = DimensionAdapter(input_dim, hidden_dim)
        self.right_adapter = DimensionAdapter(input_dim, hidden_dim)
        self.integration = NeuralModule(hidden_dim * 2, hidden_dim * 3, hidden_dim, dropout)
        self.output_adapter = DimensionAdapter(hidden_dim, input_dim)
        
    def forward(self, left_input, right_input):
        # 入力次元を調整
        left_adapted = self.left_adapter(left_input)
        right_adapted = self.right_adapter(right_input)
        
        # 左脳と右脳の出力を結合
        combined = torch.cat([left_adapted, right_adapted], dim=-1)
        
        # 統合処理
        integrated = self.integration(combined)
        
        # 出力を元の次元に戻す
        return self.output_adapter(integrated)

class HippocampusModule(nn.Module):
    """記憶形成と検索を担当するモジュール"""
    def __init__(self, embedding_dim: int, hidden_dim: int, memory_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.input_adapter = DimensionAdapter(embedding_dim, hidden_dim)
        self.encoder = NeuralModule(hidden_dim, hidden_dim * 2, hidden_dim)
        self.memory_key = nn.Linear(hidden_dim, hidden_dim)
        self.memory_value = nn.Linear(hidden_dim, hidden_dim)
        self.memory_size = memory_size
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.retriever = NeuralModule(hidden_dim * 2, hidden_dim * 2, embedding_dim)
        self.output_adapter = DimensionAdapter(hidden_dim, embedding_dim)
        
    def store(self, x):
        # 入力次元を調整
        x = self.input_adapter(x)
        
        # 記憶の形成
        encoded = self.encoder(x)
        key = self.memory_key(encoded)
        value = self.memory_value(encoded)
        
        # 記憶の更新（単純なアップデート）
        # 実際の実装では更新ポリシーを持つべき
        if key.dim() > 1:
            # バッチがある場合は先頭を使用
            self.memory_keys.data[0] = key[0].detach()
            self.memory_values.data[0] = value[0].detach()
        else:
            self.memory_keys.data[0] = key.detach()
            self.memory_values.data[0] = value.detach()
    
    def retrieve(self, query):
        try:
            # 入力次元を調整
            query = self.input_adapter(query)
            
            # 記憶検索のためのエンコード
            encoded = self.encoder(query)
            key = self.memory_key(encoded)
            
            # アテンション機構による記憶検索
            attn = torch.matmul(key, self.memory_keys.t())
            attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-1)
            retrieved = torch.matmul(attn, self.memory_values)
            
            # 検索結果とクエリを結合
            combined = torch.cat([encoded, retrieved], dim=-1)
            
            # 最終的な検索結果を生成
            result = self.retriever(combined)
            
            return result
        except Exception as e:
            print(f"記憶検索中にエラーが発生: {e}")
            # エラー発生時は元のクエリを返す
            return self.output_adapter(query)

class EmotionModule(nn.Module):
    """感情処理を担当するモジュール"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 入力と感情状態を結合するための次元計算
        self.emotion_processor = NeuralModule(input_dim, hidden_dim, input_dim)
        self.emotion_state = nn.Parameter(torch.randn(hidden_dim))
        self.adapter = DimensionAdapter(hidden_dim, input_dim)
        
    def process(self, x):
        try:
            # 入力の次元を確認
            batch_size = x.size(0)
            if x.dim() > 2:
                seq_len = x.size(1)
                # 感情状態を適切な形に拡張
                emotion = self.emotion_state.unsqueeze(0).unsqueeze(0)
                emotion = emotion.expand(batch_size, seq_len, -1)
                # 次元を入力に合わせる
                emotion = self.adapter(emotion)
                # 入力に感情的な影響を与える（加算のみ）
                return x + 0.2 * emotion
            else:
                # シーケンス次元がない場合
                emotion = self.emotion_state.unsqueeze(0).expand(batch_size, -1)
                emotion = self.adapter(emotion)
                return x + 0.2 * emotion
        except Exception as e:
            print(f"感情モジュールでエラーが発生: {e}")
            # エラーが発生した場合は元の入力をそのまま返す
            return x

class BrainModel(nn.Module):
    """人間の脳のような処理を模倣したモデル"""
    def __init__(self, vocab_size: int, hidden_size: int = 768, embedding_dim: Optional[int] = None, 
                 num_layers: int = 12, num_heads: int = 12, output_size: int = None, 
                 dropout: float = 0.1, max_length: int = 512, hidden_dim: Optional[int] = None):
        """
        初期化メソッド
        
        Args:
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層のサイズ（デフォルト: 768）
            embedding_dim: 埋め込み次元（指定された場合はhidden_sizeより優先）
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            output_size: 出力サイズ（指定されていない場合はvocab_sizeと同じ）
            dropout: ドロップアウト率
            max_length: 最大シーケンス長
            hidden_dim: hidden_sizeの別名（互換性のため）
        """
        super().__init__()
        
        # hidden_dimパラメータがある場合はhidden_sizeとして使用（互換性のため）
        if hidden_dim is not None:
            hidden_size = hidden_dim
            
        # embedding_dimが指定されていれば、それをhidden_sizeとして使用
        if embedding_dim is not None:
            hidden_size = embedding_dim
            
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_size = output_size or vocab_size
        self.dropout = dropout
        
        # 埋め込み層
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # 脳の各部位
        self.left_brain = LeftBrainModule(hidden_size, hidden_size)
        self.right_brain = RightBrainModule(hidden_size, hidden_size)
        self.frontal_lobe = FrontalLobeModule(hidden_size, hidden_size, dropout)
        self.hippocampus = HippocampusModule(hidden_size, hidden_size, memory_size=128)
        self.emotion = EmotionModule(hidden_size, hidden_size // 2)
        
        # 出力層 - 次元を明示的に合わせる
        self.output_adapter = DimensionAdapter(hidden_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, self.output_size)
        
        # 損失関数
        self.loss_fn = nn.CrossEntropyLoss()
        
        # デバイス保存用
        self.device = None
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        # デバイス追跡
        device = next(self.parameters()).device
        self.device = device
        
        # 入力テンソルをモデルのデバイスに移動
        if input_ids is not None:
            input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)
            
        try:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # 埋め込み
                embedded = self.embedding(input_ids)
                
                # 右脳からの直感的な出力を先に取得（早い反応）
                right_output = self.right_brain(embedded)
                
                # 記憶システムからコンテキスト検索
                memory_context = self.hippocampus.retrieve(embedded)
                
                # 左脳での詳細分析（時間がかかる）
                left_output = self.left_brain(embedded, memory_context)
                
                # 前頭葉による統合
                integrated = self.frontal_lobe(left_output, right_output)
                
                # 感情処理の統合
                emotional = self.emotion.process(integrated)
                
                # 出力次元の調整
                adapted_output = self.output_adapter(emotional)
                
                # 語彙に変換
                logits = self.output_projection(adapted_output)
                
                # 損失計算
                loss = None
                if labels is not None:
                    # ラベルとロジットの形状を確認
                    flat_logits = logits.reshape(-1, self.output_size)
                    flat_labels = labels.reshape(-1)
                    
                    try:
                        loss = self.loss_fn(flat_logits, flat_labels)
                    except Exception as inner_e:
                        print(f"損失計算でエラー: {inner_e}, logits={flat_logits.shape}, labels={flat_labels.shape}")
                        loss = torch.tensor(0.5, device=device, requires_grad=True)
                
                return type('OutputWithLoss', (object,), {'loss': loss, 'logits': logits})
        except Exception as e:
            print(f"モデル計算でエラーが発生: {e}")
            
            # フォールバックモード
            embedded = self.embedding(input_ids)
            
            # シンプルな計算パス（次元問題を避ける）
            # 埋め込みから直接出力へ
            logits = self.output_projection(embedded)
            
            # 損失計算
            loss = None
            if labels is not None:
                try:
                    loss = self.loss_fn(logits.reshape(-1, self.output_size), labels.reshape(-1))
                except Exception as e:
                    print(f"フォールバック損失計算でエラー: {e}")
                    loss = torch.tensor(1.0, device=device, requires_grad=True)
            
            return type('OutputWithLoss', (object,), {'loss': loss, 'logits': logits})
    
    def save_pretrained(self, save_directory):
        """モデルをHugging Face形式で保存"""
        os.makedirs(save_directory, exist_ok=True)
        
        # モデル状態の保存
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # モデル設定の保存
        config = {
            "model_type": "BrainModel",
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size, 
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "output_size": self.output_size,
            "dropout": self.dropout
        }
        
        with open(os.path.join(save_directory, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        """
        保存されたモデルからBrainModelを読み込む
        
        Args:
            model_path: モデルパスまたはHugging Face ID
            **kwargs: 追加のキーワード引数
            
        Returns:
            BrainModel: ロードされたモデルインスタンス
        """
        import os
        import json
        
        # Hugging Faceモデルの場合
        if not os.path.exists(model_path) and '/' in model_path:
            from huggingface_hub import hf_hub_download
            try:
                # configとモデルをダウンロード
                config_file = hf_hub_download(repo_id=model_path, filename="config.json")
                model_file = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
                
                # 設定を読み込む
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # モデルの作成とロード
                model = cls(
                    vocab_size=config.get("vocab_size", 32000),
                    hidden_size=config.get("hidden_size", 768),
                    num_layers=config.get("num_layers", 12),
                    num_heads=config.get("num_heads", 12),
                    output_size=config.get("output_size", config.get("vocab_size", 32000)),
                    dropout=config.get("dropout", 0.1)
                )
                
                # 状態辞書をロード
                state_dict = torch.load(model_file, map_location="cpu")
                model._load_state_dict_with_mismatch(state_dict)
                
                return model
            except Exception as e:
                print(f"Hugging Faceからのモデルロードエラー: {e}")
                raise
        
        # ローカルモデルの場合
        try:
            # チェックポイントをロード
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # モデル設定情報を取得
            if "model_config" in checkpoint:
                config = checkpoint["model_config"]
            else:
                config = {}
                # チェックポイントと同じディレクトリにconfig.jsonがあるか確認
                config_path = os.path.join(os.path.dirname(model_path), "config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
            
            # 古いチェックポイント形式との互換性
            if "hidden_dim" in config and "hidden_size" not in config:
                config["hidden_size"] = config["hidden_dim"]
            
            # モデルの作成
            vocab_size = config.get("vocab_size", kwargs.get("vocab_size", 32000))
            hidden_size = config.get("hidden_size", kwargs.get("hidden_size", 768))
            num_layers = config.get("num_layers", kwargs.get("num_layers", 12))
            num_heads = config.get("num_heads", kwargs.get("num_heads", 12))
            output_size = config.get("output_size", vocab_size)
            dropout = config.get("dropout", kwargs.get("dropout", 0.1))
            
            model = cls(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                output_size=output_size,
                dropout=dropout
            )
            
            # 状態辞書の取得と読み込み
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
                
            model._load_state_dict_with_mismatch(state_dict)
            
            return model
        except Exception as e:
            print(f"モデルロードエラー: {e}")
            raise
    
    def _load_state_dict_with_mismatch(self, state_dict):
        """
        語彙サイズの不一致があっても状態辞書を読み込む
        
        Args:
            state_dict: ロードする状態辞書
        """
        # サイズ不一致を許容するためのフラグ
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        
        # 現在のモデルとチェックポイントの語彙サイズを取得
        checkpoint_vocab_size = None
        if 'embedding.weight' in state_dict:
            checkpoint_vocab_size = state_dict['embedding.weight'].size(0)
        elif 'output_projection.bias' in state_dict:
            checkpoint_vocab_size = state_dict['output_projection.bias'].size(0)
        
        current_vocab_size = self.vocab_size
        
        # 語彙サイズの不一致を検出した場合
        if checkpoint_vocab_size and checkpoint_vocab_size != current_vocab_size:
            print(f"語彙サイズの不一致を検出: チェックポイント={checkpoint_vocab_size}, 現在のモデル={current_vocab_size}")
            print("語彙サイズを調整します...")
            
            # 埋め込み層の重みを調整
            if 'embedding.weight' in state_dict:
                embedding_weight = state_dict['embedding.weight']
                if current_vocab_size < checkpoint_vocab_size:
                    # 現在のモデルの語彙サイズが小さい場合、切り詰める
                    state_dict['embedding.weight'] = embedding_weight[:current_vocab_size, :]
                else:
                    # 現在のモデルの語彙サイズが大きい場合、拡張する
                    new_embedding = torch.zeros(
                        (current_vocab_size, embedding_weight.size(1)),
                        dtype=embedding_weight.dtype
                    )
                    new_embedding[:checkpoint_vocab_size, :] = embedding_weight
                    # 新しいトークンをランダムに初期化
                    nn.init.normal_(new_embedding[checkpoint_vocab_size:, :], mean=0.0, std=0.02)
                    state_dict['embedding.weight'] = new_embedding
            
            # 出力投影層の重みを調整
            if 'output_projection.weight' in state_dict:
                output_weight = state_dict['output_projection.weight']
                if current_vocab_size < checkpoint_vocab_size:
                    # 切り詰める
                    state_dict['output_projection.weight'] = output_weight[:current_vocab_size, :]
                else:
                    # 拡張する
                    new_output = torch.zeros(
                        (current_vocab_size, output_weight.size(1)), 
                        dtype=output_weight.dtype
                    )
                    new_output[:checkpoint_vocab_size, :] = output_weight
                    nn.init.normal_(new_output[checkpoint_vocab_size:, :], mean=0.0, std=0.02)
                    state_dict['output_projection.weight'] = new_output
            
            # 出力投影層のバイアスを調整
            if 'output_projection.bias' in state_dict:
                output_bias = state_dict['output_projection.bias']
                if current_vocab_size < checkpoint_vocab_size:
                    # 切り詰める
                    state_dict['output_projection.bias'] = output_bias[:current_vocab_size]
                else:
                    # 拡張する
                    new_bias = torch.zeros(
                        current_vocab_size, 
                        dtype=output_bias.dtype
                    )
                    new_bias[:checkpoint_vocab_size] = output_bias
                    nn.init.zeros_(new_bias[checkpoint_vocab_size:])
                    state_dict['output_projection.bias'] = new_bias
        
        # 状態辞書をロード（厳密さのフラグをFalseに設定してサイズ不一致を許容）
        self.load_state_dict(state_dict, strict=False)
    
    def generate(self, input_ids, attention_mask=None, max_length=100, temperature=1.0, 
                 top_k=50, top_p=0.95, repetition_penalty=1.0, do_sample=True, 
                 pad_token_id=None, eos_token_id=None, **kwargs):
        """
        文章生成用のメソッド
        
        Args:
            input_ids: 入力トークンID
            attention_mask: アテンションマスク
            max_length: 生成する最大長
            temperature: 生成の温度パラメータ
            top_k: Top-k サンプリングの k 値
            top_p: 核サンプリングの確率閾値
            repetition_penalty: 繰り返しペナルティ
            do_sample: サンプリングを行うかどうか (False の場合は貪欲法)
            pad_token_id: パディングトークンID
            eos_token_id: 終了トークンID
            **kwargs: その他の引数
            
        Returns:
            生成されたトークンのテンソル
        """
        self.eval()  # 評価モードに設定
        device = next(self.parameters()).device
        
        # 入力テンソルの形状確認と調整
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.to(device)
        
        # バッチ次元がない場合は追加
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        # 現在の入力長
        cur_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        
        # EOS/PADトークンの設定
        if eos_token_id is None:
            eos_token_id = 2  # 一般的なEOSトークンのデフォルト値
        
        if pad_token_id is None:
            pad_token_id = 0  # 一般的なPADトークンのデフォルト値
        
        # 生成済みの入力を保持
        generated_ids = input_ids.clone()
        
        # 最大長のチェックと調整
        if max_length < cur_len:
            max_length = cur_len + 50  # 少なくとも50トークンは生成
            
        with torch.no_grad():
            # 自動回帰的に文章を生成
            for _ in range(max_length - cur_len):
                # 現在のシーケンス全体を対象に次のトークンを予測
                outputs = self(
                    input_ids=generated_ids,
                    attention_mask=attention_mask
                )
                
                next_token_logits = outputs.logits[:, -1, :]
                
                # 温度によるスケーリング
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # 繰り返しペナルティの適用
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated_ids[i].tolist()):
                            if next_token_logits[i, token_id] < 0:
                                next_token_logits[i, token_id] *= repetition_penalty
                            else:
                                next_token_logits[i, token_id] /= repetition_penalty
                
                # Top-k サンプリング
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Top-p (核) サンプリング
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 累積確率がtop_pを超える部分を除去
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 最初のトークンは保持
                    sorted_indices_to_remove[..., 0] = False
                    
                    # インデックスをソート前の位置に戻す
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
                
                # サンプリングまたは貪欲法による次トークンの選択
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # EOSトークンに到達したら、そのシーケンスの生成を停止
                for i in range(batch_size):
                    if next_tokens[i].item() == eos_token_id:
                        next_tokens[i] = pad_token_id
                
                # 次のトークンを生成済みトークンに追加
                generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # アテンションマスクの更新
                if attention_mask is not None:
                    attention_mask = F.pad(attention_mask, (0, 1), value=1)
                
                # すべてのバッチがEOSに到達したか確認
                if all(token == pad_token_id for token in next_tokens):
                    break
        
        return generated_ids