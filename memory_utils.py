"""
メモリ使用状況の監視と次元調整のためのユーティリティ関数
"""

import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import logging
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

def log_memory_usage():
    """システムとGPUのメモリ使用状況をログ出力"""
    # システムメモリ
    process = psutil.Process(os.getpid())
    ram = psutil.virtual_memory()
    process_memory = process.memory_info().rss / (1024 * 1024)  # MB単位
    
    logger.info(f"プロセスメモリ使用量: {process_memory:.1f}MB")
    logger.info(f"システムメモリ: 使用中 {ram.used/(1024**3):.1f}GB / 合計 {ram.total/(1024**3):.1f}GB ({ram.percent}%)")
    
    # GPUメモリ
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                       f"割当 {allocated_mem:.1f}GB / 確保 {reserved_mem:.1f}GB / 合計 {total_mem:.1f}GB")
    else:
        logger.info("GPUは使用できません")

def optimize_memory():
    """メモリの最適化を行う"""
    # Pythonのガベージコレクションを明示的に実行
    gc.collect()
    
    # GPUキャッシュのクリア
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPUキャッシュをクリアしました")

def check_tensor_dims(x: torch.Tensor, name: str = "tensor"):
    """テンソルの次元情報を出力"""
    logger.info(f"{name} - 形状: {x.shape}, デバイス: {x.device}, 型: {x.dtype}")
    if torch.isnan(x).any():
        logger.warning(f"{name} にNaN値が含まれています")
    if torch.isinf(x).any():
        logger.warning(f"{name} にInf値が含まれています")
    return x.shape

def ensure_matching_dims(a: torch.Tensor, b: torch.Tensor):
    """2つのテンソルが乗算可能な次元になっているか確認"""
    if a.dim() < 2 or b.dim() < 2:
        logger.warning(f"少なくとも一方のテンソルは2次元未満です: a={a.dim()}, b={b.dim()}")
        return False
    
    if a.size(-1) != b.size(-2):
        logger.warning(f"テンソル次元が乗算不可: a{a.shape} @ b{b.shape}, "
                      f"a_last={a.size(-1)} != b_first={b.size(-2)}")
        return False
    
    return True

def estimate_memory_requirement(model: nn.Module, batch_size: int, seq_len: int, fp16: bool = True):
    """モデルの推定メモリ要件を計算"""
    # パラメータ数をカウント
    params = sum(p.nelement() for p in model.parameters())
    param_size = params * (2 if fp16 else 4)  # バイト単位
    
    # 仮想バッチでの中間表現を計算
    sample_input = torch.zeros((batch_size, seq_len), dtype=torch.long)
    sample_attn = torch.ones((batch_size, seq_len), dtype=torch.long)
    
    # 推論モード
    model.eval()
    
    # 中間表現が使うメモリを概算
    intermediate_memory = 0
    try:
        # フックを使用して中間表現のサイズを取得
        activations = []
        hooks = []
        
        def hook_fn(m, i, o):
            if isinstance(o, torch.Tensor):
                bytes_per_elem = 2 if fp16 else 4
                activation_size = o.nelement() * bytes_per_elem
                activations.append(activation_size)
            
        # モジュールにフックを登録
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # 順伝播を実行
        with torch.no_grad():
            _ = model(sample_input, attention_mask=sample_attn)
        
        # フックを削除
        for h in hooks:
            h.remove()
        
        # 中間表現メモリを合計
        intermediate_memory = sum(activations)
    except Exception as e:
        logger.warning(f"中間表現メモリ推定中にエラー: {e}")
        # フォールバック: バッチサイズ、シーケンス長、隠れ層をもとに概算
        hidden_size = getattr(model, 'hidden_size', 768)
        intermediate_memory = batch_size * seq_len * hidden_size * (2 if fp16 else 4) * 12
    
    # 勾配用のメモリ（訓練時は概算で2倍）
    gradient_memory = param_size
    
    # オプティマイザ状態用メモリ（AdamWを想定）
    optimizer_memory = param_size * 2  # モーメンタムと分散
    
    # 合計メモリ使用量
    total_memory_bytes = param_size + intermediate_memory + gradient_memory + optimizer_memory
    total_memory_gb = total_memory_bytes / (1024**3)
    
    return {
        "params": params,
        "param_memory_bytes": param_size,
        "intermediate_memory_bytes": intermediate_memory,
        "gradient_memory_bytes": gradient_memory, 
        "optimizer_memory_bytes": optimizer_memory,
        "total_memory_bytes": total_memory_bytes,
        "total_memory_gb": total_memory_gb
    }

def verify_model_dimensions(model):
    """モデルの次元整合性を検証"""
    issues = []
    
    # 埋め込み層と出力層の次元が一致しているか確認
    if hasattr(model, 'embedding') and hasattr(model, 'output_projection'):
        embedding_out_dim = model.embedding.weight.size(1)
        output_in_dim = model.output_projection.weight.size(1)
        
        if embedding_out_dim != output_in_dim:
            issues.append(f"埋め込み出力次元 ({embedding_out_dim}) と出力層入力次元 ({output_in_dim}) が一致していません")
    
    # 左脳と右脳の次元が一致しているか確認
    if hasattr(model, 'left_brain') and hasattr(model, 'right_brain'):
        left_dim = getattr(model.left_brain, 'hidden_dim', None)
        right_dim = getattr(model.right_brain, 'hidden_dim', None)
        
        if left_dim is not None and right_dim is not None and left_dim != right_dim:
            issues.append(f"左脳次元 ({left_dim}) と右脳次元 ({right_dim}) が一致していません")
    
    # 出力層の次元が語彙サイズと一致しているか確認
    if hasattr(model, 'output_projection') and hasattr(model, 'vocab_size'):
        output_dim = model.output_projection.weight.size(0)
        vocab_size = model.vocab_size
        
        if output_dim != vocab_size:
            issues.append(f"出力層の次元 ({output_dim}) が語彙サイズ ({vocab_size}) と一致していません")
    
    # 結果を返す
    if issues:
        return False, issues
    return True, []

def validate_model_config(args):
    """コマンドライン引数のモデル設定を検証して補正"""
    # 基本的なモデル次元パラメータのデフォルト値を設定
    if not hasattr(args, 'hidden_size'):
        args.hidden_size = 768
        logger.info(f"hidden_sizeが指定されていません。デフォルト値を使用: {args.hidden_size}")
        
    if not hasattr(args, 'model_dim'):
        args.model_dim = args.hidden_size
        logger.info(f"model_dimが指定されていません。hidden_sizeと同じ値を使用: {args.model_dim}")
    
    if not hasattr(args, 'num_layers'):
        args.num_layers = 12
        logger.info(f"num_layersが指定されていません。デフォルト値を使用: {args.num_layers}")
    
    if not hasattr(args, 'num_heads'):
        args.num_heads = 12
        logger.info(f"num_headsが指定されていません。デフォルト値を使用: {args.num_heads}")
    
    if not hasattr(args, 'dropout'):
        args.dropout = 0.1
        logger.info(f"dropoutが指定されていません。デフォルト値を使用: {args.dropout}")
    
    # 次元の整合性チェック
    if hasattr(args, 'embedding_dim') and args.embedding_dim is not None:
        if args.embedding_dim != args.hidden_size:
            logger.warning(f"embedding_dim ({args.embedding_dim}) とhidden_size ({args.hidden_size}) が一致しません。embedding_dimが優先されます。")
    
    return args
