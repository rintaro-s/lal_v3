import os
import gc
import torch
import psutil
import logging
import numpy as np
from typing import Optional, Dict, List
import time

def log_memory_usage(logger: logging.Logger):
    """現在のメモリ使用状況をログに記録"""
    # CPU メモリ
    ram = psutil.virtual_memory()
    logger.info(f"RAM: 使用中 {ram.used/1024**3:.1f}GB / {ram.total/1024**3:.1f}GB ({ram.percent}%)")
    
    # GPU メモリ (利用可能な場合)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            gpu_mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            gpu_mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): 確保 {gpu_mem_alloc:.1f}GB / 予約 {gpu_mem_reserved:.1f}GB / 合計 {gpu_mem_total:.1f}GB")

def clear_gpu_memory():
    """GPUメモリを解放"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def optimize_memory_usage(min_free_gb: float = 2.0):
    """メモリ使用量を最適化"""
    # 不要なPythonオブジェクトを解放
    gc.collect()
    
    # GPUメモリを解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # CPU RAMの空き状況を確認
    ram = psutil.virtual_memory()
    
    # 空きメモリが指定閾値以下の場合、追加のメモリ確保を試みる
    if ram.available / 1024**3 < min_free_gb:
        # 積極的なGCを実行
        for _ in range(3):
            gc.collect()
        
        # 環境変数を介してPythonのGCを調整
        os.environ['PYTHONMALLOC'] = 'malloc'
        os.environ['PYTHONGC'] = 'aggressive'

class MemoryTracker:
    """メモリ使用量追跡クラス"""
    def __init__(self, log_interval: int = 60):
        self.log_interval = log_interval  # ログ記録間隔（秒）
        self.last_log_time = time.time()
        self.peak_gpu_memory = 0.0
        self.peak_ram = 0.0
        self.gpu_history = []
        self.ram_history = []
        
    def update(self, force_log: bool = False, logger: Optional[logging.Logger] = None):
        """メモリ使用状況を更新"""
        current_time = time.time()
        
        # RAM情報を取得
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / 1024**3
        self.peak_ram = max(self.peak_ram, ram_used_gb)
        self.ram_history.append(ram_used_gb)
        
        # GPU情報を取得（利用可能な場合）
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.memory_allocated(0) / 1024**3
            self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem_gb)
            self.gpu_history.append(gpu_mem_gb)
        else:
            gpu_mem_gb = 0.0
            self.gpu_history.append(0.0)
        
        # ログ記録間隔を過ぎているか、強制ログが要求された場合
        if force_log or (current_time - self.last_log_time) >= self.log_interval:
            message = (f"メモリ使用: RAM {ram_used_gb:.1f}GB (ピーク: {self.peak_ram:.1f}GB), "
                      f"GPU {gpu_mem_gb:.1f}GB (ピーク: {self.peak_gpu_memory:.1f}GB)")
            
            if logger:
                logger.info(message)
            else:
                print(message)
            
            self.last_log_time = current_time
    
    def get_stats(self) -> Dict:
        """メモリ統計情報を取得"""
        return {
            "peak_ram_gb": self.peak_ram,
            "peak_gpu_gb": self.peak_gpu_memory,
            "ram_history": self.ram_history,
            "gpu_history": self.gpu_history,
            "last_ram_gb": self.ram_history[-1] if self.ram_history else 0.0,
            "last_gpu_gb": self.gpu_history[-1] if self.gpu_history else 0.0
        }
    
    def plot_history(self, filename: str = "memory_usage.png"):
        """メモリ使用履歴をプロット"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # RAM使用履歴
            plt.plot(self.ram_history, label='RAM使用量 (GB)', color='blue')
            plt.axhline(y=self.peak_ram, linestyle='--', color='darkblue', alpha=0.7, label=f'RAM最大値: {self.peak_ram:.2f}GB')
            
            # GPU使用履歴（利用可能な場合）
            if any(self.gpu_history):
                plt.plot(self.gpu_history, label='GPU使用量 (GB)', color='red')
                plt.axhline(y=self.peak_gpu_memory, linestyle='--', color='darkred', alpha=0.7, label=f'GPU最大値: {self.peak_gpu_memory:.2f}GB')
            
            # グラフの設定
            plt.title('メモリ使用履歴')
            plt.xlabel('測定ポイント')
            plt.ylabel('使用量 (GB)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # ファイルに保存
            plt.savefig(filename)
            plt.close()
            return True
            
        except ImportError:
            print("matplotlibがインストールされていないため、グラフを作成できません。")
            return False

class RAMDataCache:
    """大規模システムRAMを活用したデータキャッシュ"""
    def __init__(self, max_size_gb: float = 50.0):
        self.max_size_bytes = max_size_gb * 1024**3
        self.current_size_bytes = 0
        self.cache = {}
        
    def add(self, key: str, data: any) -> bool:
        """キャッシュにデータを追加"""
        # データサイズの見積もり
        import sys
        data_size = sys.getsizeof(data)
        
        # 再帰的にデータサイズを推定（リストやディクショナリなど）
        if isinstance(data, dict):
            for k, v in data.items():
                data_size += sys.getsizeof(k)
                data_size += self._estimate_size(v)
        elif isinstance(data, (list, tuple)):
            for item in data:
                data_size += self._estimate_size(item)
                
        # キャッシュに追加するスペースがあるか確認
        if self.current_size_bytes + data_size > self.max_size_bytes:
            return False
        
        self.cache[key] = data
        self.current_size_bytes += data_size
        return True
    
    def get(self, key: str, default=None):
        """キャッシュからデータを取得"""
        return self.cache.get(key, default)
    
    def clear(self):
        """キャッシュをクリア"""
        self.cache.clear()
        self.current_size_bytes = 0
        gc.collect()
    
    def _estimate_size(self, obj: any) -> int:
        """オブジェクトのサイズを再帰的に見積もる"""
        import sys
        
        if obj is None:
            return 0
            
        size = sys.getsizeof(obj)
        
        if isinstance(obj, dict):
            size += sum(self._estimate_size(v) for v in obj.values())
            size += sum(sys.getsizeof(k) for k in obj.keys())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(self._estimate_size(i) for i in obj)
        
        return size
