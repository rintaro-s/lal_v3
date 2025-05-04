import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass

@dataclass
class MemoryItem:
    """記憶アイテムの構造"""
    content: torch.Tensor
    importance: float
    timestamp: float
    tag: str = ""
    
    @property
    def age(self):
        """記憶の経過時間（秒）"""
        return time.time() - self.timestamp

class WorkingMemory:
    """ワーキングメモリ層：現在の会話コンテキスト保持"""
    def __init__(self, capacity: int = 10, embedding_dim: int = 768):
        self.capacity = capacity
        self.buffer = []
        self.embedding_dim = embedding_dim
    
    def add(self, item: torch.Tensor, importance: float = 1.0, tag: str = ""):
        """新しいアイテムを追加"""
        memory_item = MemoryItem(
            content=item.detach().clone(),
            importance=importance,
            timestamp=time.time(),
            tag=tag
        )
        
        self.buffer.append(memory_item)
        
        # 容量を超えた場合、重要度が最も低いアイテムを削除
        if len(self.buffer) > self.capacity:
            # 重要度でソートして最も低いものを削除
            self.buffer.sort(key=lambda x: x.importance)
            self.buffer.pop(0)
    
    def retrieve(self, query: torch.Tensor, top_k: int = 3) -> List[torch.Tensor]:
        """クエリに関連するアイテムを取得"""
        if not self.buffer:
            return []
        
        similarities = []
        for item in self.buffer:
            # コサイン類似度を計算
            sim = torch.cosine_similarity(query.flatten(), item.content.flatten(), dim=0)
            similarities.append((sim.item(), item))
        
        # 類似度でソート
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 上位k個を返す
        return [item.content for _, item in similarities[:top_k]]
    
    def update_importance(self, indices: List[int], new_importance: List[float]):
        """特定のアイテムの重要度を更新"""
        for idx, importance in zip(indices, new_importance):
            if 0 <= idx < len(self.buffer):
                self.buffer[idx].importance = importance
    
    def clear(self):
        """メモリをクリア"""
        self.buffer = []
    
    def get_all(self) -> List[torch.Tensor]:
        """すべてのアイテムを取得"""
        return [item.content for item in self.buffer]

class ShortTermMemory:
    """短期記憶層：最近の会話履歴とトピック"""
    def __init__(self, capacity: int = 100, decay_rate: float = 0.05, embedding_dim: int = 768):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.buffer = []
        self.embedding_dim = embedding_dim
        self.last_decay_time = time.time()
    
    def add(self, item: torch.Tensor, importance: float = 1.0, tag: str = ""):
        """新しいアイテムを追加"""
        memory_item = MemoryItem(
            content=item.detach().clone(),
            importance=importance,
            timestamp=time.time(),
            tag=tag
        )
        
        self.buffer.append(memory_item)
        
        # 容量を超えた場合、最も古いアイテムを削除
        if len(self.buffer) > self.capacity:
            self.buffer.sort(key=lambda x: x.timestamp)
            self.buffer.pop(0)
        
        # 定期的に重要度の減衰を適用
        self._apply_decay()
    
    def _apply_decay(self):
        """時間経過に基づいて記憶の重要度を減衰させる"""
        current_time = time.time()
        
        # 前回の減衰から一定時間経過した場合のみ減衰を適用
        if current_time - self.last_decay_time > 60:  # 1分ごとに減衰
            for item in self.buffer:
                # 経過時間に基づいて重要度を減衰
                time_factor = (current_time - item.timestamp) / 3600  # 時間単位
                decay = np.exp(-self.decay_rate * time_factor)
                item.importance *= decay
            
            # Miller's Law（7±2）を考慮した整理
            if len(self.buffer) > 9:
                # 重要度でソート
                self.buffer.sort(key=lambda x: x.importance, reverse=True)
                # 上位7-9個を保持
                keep_count = np.random.randint(7, 10)
                self.buffer = self.buffer[:keep_count]
            
            self.last_decay_time = current_time
    
    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> List[torch.Tensor]:
        """クエリに関連するアイテムを取得"""
        if not self.buffer:
            return []
        
        # 重要度が閾値以下のアイテムをフィルタリング
        filtered_buffer = [item for item in self.buffer if item.importance > 0.2]
        
        if not filtered_buffer:
            return []
        
        similarities = []
        for item in filtered_buffer:
            # コサイン類似度を計算
            sim = torch.cosine_similarity(query.flatten(), item.content.flatten(), dim=0)
            # 重要度と新しさも考慮
            recency_factor = np.exp(-0.1 * item.age / 3600)  # 時間経過による減衰
            adjusted_sim = sim.item() * item.importance * recency_factor
            similarities.append((adjusted_sim, item))
        
        # 調整済み類似度でソート
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 上位k個を返す
        return [item.content for _, item in similarities[:top_k]]
    
    def forget_old_memories(self, threshold_hours: float = 24.0):
        """一定時間以上経過した記憶を忘れる"""
        current_time = time.time()
        threshold_seconds = threshold_hours * 3600
        
        self.buffer = [item for item in self.buffer 
                      if (current_time - item.timestamp) < threshold_seconds]

class LongTermMemory:
    """長期記憶層：事前学習知識と重要な記憶"""
    def __init__(self, capacity: int = 10000, embedding_dim: int = 768):
        self.capacity = capacity
        self.semantic_memory = []  # 意味記憶（事実、概念）
        self.episodic_memory = []  # エピソード記憶（経験、出来事）
        self.embedding_dim = embedding_dim
        
        # 記憶の簡略化・物語化のための変換器
        self.summarizer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
    
    def add_semantic(self, item: torch.Tensor, importance: float = 1.0, tag: str = ""):
        """意味記憶に追加"""
        # 簡略化して保存（記憶の抽象化）
        summarized = self.summarizer(item)
        
        memory_item = MemoryItem(
            content=summarized.detach().clone(),
            importance=importance,
            timestamp=time.time(),
            tag=tag
        )
        
        self.semantic_memory.append(memory_item)
        
        # 容量を超えた場合、重要度が最も低いアイテムを削除
        if len(self.semantic_memory) > self.capacity // 2:
            self.semantic_memory.sort(key=lambda x: x.importance)
            self.semantic_memory.pop(0)
    
    def add_episodic(self, item: torch.Tensor, importance: float = 1.0, tag: str = ""):
        """エピソード記憶に追加"""
        memory_item = MemoryItem(
            content=item.detach().clone(),
            importance=importance,
            timestamp=time.time(),
            tag=tag
        )
        
        self.episodic_memory.append(memory_item)
        
        # 容量を超えた場合、重要度が最も低いアイテムを削除
        if len(self.episodic_memory) > self.capacity // 2:
            self.episodic_memory.sort(key=lambda x: x.importance)
            self.episodic_memory.pop(0)
    
    def retrieve(self, query: torch.Tensor, memory_type: str = "both", top_k: int = 5) -> List[Tuple[torch.Tensor, str]]:
        """クエリに関連する記憶を取得"""
        results = []
        
        # 検索対象のメモリを決定
        if memory_type == "semantic" or memory_type == "both":
            for item in self.semantic_memory:
                sim = torch.cosine_similarity(query.flatten(), item.content.flatten(), dim=0)
                results.append((sim.item(), item, "semantic"))
        
        if memory_type == "episodic" or memory_type == "both":
            for item in self.episodic_memory:
                sim = torch.cosine_similarity(query.flatten(), item.content.flatten(), dim=0)
                results.append((sim.item(), item, "episodic"))
        
        # 類似度でソート
        results.sort(key=lambda x: x[0] * x[1].importance, reverse=True)
        
        # 上位k個を返す
        return [(item.content, mem_type) for _, item, mem_type in results[:top_k]]
    
    def consolidate_memories(self):
        """記憶の整理・統合処理（定期的に実行）"""
        if not self.episodic_memory:
            return
        
        # 類似したエピソード記憶をグループ化
        groups = self._cluster_similar_memories()
        
        for group in groups:
            if len(group) > 1:
                # グループ内の記憶を統合
                consolidated = torch.stack([item.content for item in group]).mean(dim=0)
                importance = max(item.importance for item in group)
                
                # 統合した記憶を意味記憶に追加
                self.add_semantic(consolidated, importance, "consolidated")
                
                # 元のエピソード記憶を削除（オプション）
                for item in group:
                    if item in self.episodic_memory:
                        self.episodic_memory.remove(item)
    
    def _cluster_similar_memories(self, threshold: float = 0.8):
        """類似した記憶をクラスタリング"""
        groups = []
        remaining = self.episodic_memory.copy()
        
        while remaining:
            current = remaining.pop(0)
            group = [current]
            
            i = 0
            while i < len(remaining):
                item = remaining[i]
                sim = torch.cosine_similarity(
                    current.content.flatten(), 
                    item.content.flatten(), 
                    dim=0
                ).item()
                
                if sim > threshold:
                    group.append(item)
                    remaining.pop(i)
                else:
                    i += 1
            
            groups.append(group)
        
        return groups

class ProceduralMemory:
    """体の記憶（手続き記憶）シミュレーション"""
    def __init__(self, pattern_dim: int = 768, max_patterns: int = 100):
        self.patterns = {}  # パターン名: (パターン表現, 使用回数)
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
    
    def register_pattern(self, name: str, pattern: torch.Tensor):
        """新しい行動パターンを登録"""
        self.patterns[name] = (pattern.detach().clone(), 1)
        
        # パターン数が上限を超えた場合、最も使用頻度の低いものを削除
        if len(self.patterns) > self.max_patterns:
            min_name = min(self.patterns.items(), key=lambda x: x[1][1])[0]
            del self.patterns[min_name]
    
    def recognize_pattern(self, input_pattern: torch.Tensor, threshold: float = 0.85) -> Optional[str]:
        """入力パターンに一致する登録済みパターンを認識"""
        best_match = None
        best_similarity = threshold  # この閾値を超えるもののみ一致と見なす
        
        for name, (pattern, _) in self.patterns.items():
            sim = torch.cosine_similarity(
                input_pattern.flatten(), 
                pattern.flatten(), 
                dim=0
            ).item()
            
            if sim > best_similarity:
                best_similarity = sim
                best_match = name
        
        if best_match:
            # 使用回数を増やす
            pattern, count = self.patterns[best_match]
            self.patterns[best_match] = (pattern, count + 1)
        
        return best_match
    
    def get_pattern(self, name: str) -> Optional[torch.Tensor]:
        """名前でパターンを取得"""
        if name in self.patterns:
            return self.patterns[name][0]
        return None

class MemorySystem:
    """統合記憶システム"""
    def __init__(self, embedding_dim: int = 768):
        self.working_memory = WorkingMemory(capacity=10, embedding_dim=embedding_dim)
        self.short_term_memory = ShortTermMemory(capacity=100, embedding_dim=embedding_dim)
        self.long_term_memory = LongTermMemory(capacity=10000, embedding_dim=embedding_dim)
        self.procedural_memory = ProceduralMemory(pattern_dim=embedding_dim)
        self.embedding_dim = embedding_dim
    
    def process_input(self, input_embedding: torch.Tensor, importance: float = 1.0, tag: str = ""):
        """入力を記憶システムに処理"""
        # ワーキングメモリに追加（現在の会話コンテキスト）
        self.working_memory.add(input_embedding, importance, tag)
        
        # 重要度に応じて短期記憶にも追加
        if importance > 0.5:
            self.short_term_memory.add(input_embedding, importance, tag)
        
        # 非常に重要な情報は長期記憶にも追加
        if importance > 0.8:
            self.long_term_memory.add_episodic(input_embedding, importance, tag)
        
        # 手続き記憶のパターン認識を試みる
        pattern_name = self.procedural_memory.recognize_pattern(input_embedding)
        
        return pattern_name
    
    def retrieve_relevant_context(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """クエリに関連するコンテキストを記憶から取得"""
        # 各記憶層から関連情報を取得
        working_memories = self.working_memory.retrieve(query_embedding, top_k=3)
        short_term_memories = self.short_term_memory.retrieve(query_embedding, top_k=3)
        long_term_memories = [m for m, _ in self.long_term_memory.retrieve(query_embedding, top_k=3)]
        
        all_memories = []
        
        # 各記憶を重み付きで統合
        if working_memories:
            working_avg = torch.stack(working_memories).mean(dim=0)
            all_memories.append((working_avg, 3.0))  # ワーキングメモリは最も重要
        
        if short_term_memories:
            short_term_avg = torch.stack(short_term_memories).mean(dim=0)
            all_memories.append((short_term_avg, 2.0))  # 短期記憶は次に重要
        
        if long_term_memories:
            long_term_avg = torch.stack(long_term_memories).mean(dim=0)
            all_memories.append((long_term_avg, 1.0))  # 長期記憶は補助的
        
        if not all_memories:
            # メモリが空の場合はゼロベクトルを返す
            return torch.zeros(self.embedding_dim)
        
        # 重み付き平均を計算
        weighted_sum = torch.zeros(self.embedding_dim)
        total_weight = 0.0
        
        for memory, weight in all_memories:
            weighted_sum += memory * weight
            total_weight += weight
        
        return weighted_sum / total_weight
    
    def maintenance(self):
        """記憶システムのメンテナンス処理（定期的に実行）"""
        # 短期記憶の古いアイテムを忘却
        self.short_term_memory.forget_old_memories()
        
        # 長期記憶の整理・統合
        self.long_term_memory.consolidate_memories()
