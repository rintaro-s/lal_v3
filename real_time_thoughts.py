import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

@dataclass
class ThoughtChunk:
    """思考の断片（チャンク）"""
    text: str
    confidence: float
    type: str  # "intuition", "logical", "emotional", "uncertain", "insight"
    timestamp: float
    
    @property
    def age(self):
        return time.time() - self.timestamp

class ThoughtGenerator:
    """思考生成器"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature_scheduler = TemperatureScheduler()
        self.uncertainty_detector = UncertaintyDetector()
        
        # 思考状態の追跡
        self.current_thinking_state = "initial"  # "initial", "exploring", "analyzing", "concluding"
        
        # 思考チャンクキュー
        self.thought_queue = queue.Queue()
        self.is_thinking = False
        self.thinking_thread = None
    
    def start_thinking(self, input_text: str):
        """思考プロセスを開始"""
        self.is_thinking = True
        self.thinking_thread = threading.Thread(
            target=self._background_thinking,
            args=(input_text,)
        )
        self.thinking_thread.start()
    
    def _background_thinking(self, input_text: str):
        """バックグラウンドでの思考プロセス"""
        try:
            # 入力のトークン化
            input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
            
            # 1. 直感的な最初の反応（高温度、短時間）
            self.current_thinking_state = "initial"
            intuition = self._generate_intuitive_response(input_ids)
            self._enqueue_thought_chunks(intuition, "intuition", 0.7)
            
            # 2. 探索的思考（中温度、やや長め）
            self.current_thinking_state = "exploring"
            exploration = self._generate_exploratory_thoughts(input_ids)
            self._enqueue_thought_chunks(exploration, "logical", 0.8)
            
            # 3. 分析的思考（低温度、時間をかける）
            self.current_thinking_state = "analyzing"
            analysis = self._generate_analytical_thoughts(input_ids, intuition)
            self._enqueue_thought_chunks(analysis, "logical", 0.9)
            
            # 4. ひらめき生成（変動温度、ランダム性導入）
            insight = self._generate_insight(input_ids, intuition + analysis)
            if insight:
                self._enqueue_thought_chunks(insight, "insight", 0.95)
            
            # 5. 結論（低温度、確信度高め）
            self.current_thinking_state = "concluding"
            conclusion = self._generate_conclusion(input_ids, intuition + analysis + (insight or ""))
            self._enqueue_thought_chunks(conclusion, "logical", 1.0)
        
        except Exception as e:
            print(f"Error in thought generation: {e}")
        finally:
            self.is_thinking = False
    
    def _generate_intuitive_response(self, input_ids):
        """直感的な最初の反応を生成"""
        # 高温度で短いレスポンスを生成
        temp = self.temperature_scheduler.get_temperature("intuitive")
        outputs = self.model.generate(
            input_ids,
            max_length=50,
            temperature=temp,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.9
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 思考状態を表す表現を追加
        thinking_expressions = [
            "えっと、", "まず思いついたのは、", "直感的には、", 
            "最初に思ったのは、", "パッと見た感じだと、"
        ]
        prefix = np.random.choice(thinking_expressions)
        
        return prefix + response
    
    def _generate_exploratory_thoughts(self, input_ids):
        """探索的思考を生成"""
        temp = self.temperature_scheduler.get_temperature("exploratory")
        
        # 元の入力と「もう少し考えてみると...」のようなプロンプトを結合
        exploratory_prompt = self.tokenizer.encode("もう少し考えてみると...", return_tensors='pt').to(self.device)
        combined_input = torch.cat([input_ids, exploratory_prompt], dim=1)
        
        outputs = self.model.generate(
            combined_input,
            max_length=100,
            temperature=temp,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.85
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 思考状態を表す表現を追加
        transition_expressions = [
            "\nでも、考えてみると、", "\nただ、別の視点から見ると、",
            "\n一方で、", "\nもう少し深く考えると、"
        ]
        transition = np.random.choice(transition_expressions)
        
        return transition + response
    
    def _generate_analytical_thoughts(self, input_ids, previous_thoughts):
        """分析的思考を生成"""
        temp = self.temperature_scheduler.get_temperature("analytical")
        
        # 前の思考と分析プロンプトを結合
        analytical_prompt = self.tokenizer.encode(
            previous_thoughts + "\n\nより詳しく分析すると...", 
            return_tensors='pt'
        ).to(self.device)
        
        outputs = self.model.generate(
            analytical_prompt,
            max_length=150,
            temperature=temp,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.2
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 不確実性を検出し、適切な表現を挿入
        uncertainty = self.uncertainty_detector.detect_uncertainty(response)
        if uncertainty > 0.6:
            hedges = [
                "\nただし、ここには不確実な要素があって、", 
                "\n確実ではないですが、", 
                "\nこの点については注意が必要で、"
            ]
            response += np.random.choice(hedges)
        
        return "\n\n" + response
    
    def _generate_insight(self, input_ids, previous_thoughts):
        """ひらめきを生成（確率的に発生）"""
        # ひらめきは75%の確率で生成
        if np.random.random() > 0.25:
            temp = self.temperature_scheduler.get_temperature("insight")
            
            # 前の思考とひらめきプロンプトを結合
            insight_prompt = self.tokenizer.encode(
                previous_thoughts + "\n\nあ！そうか！", 
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model.generate(
                insight_prompt,
                max_length=50,
                temperature=temp,
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            insight_expressions = [
                "\n\nあ！ひらめいた！", "\n\nそうか！", 
                "\n\nなるほど！実は", "\n\n待てよ、もしかして"
            ]
            prefix = np.random.choice(insight_expressions)
            
            return prefix + response
        
        return None
    
    def _generate_conclusion(self, input_ids, previous_thoughts):
        """結論を生成"""
        temp = self.temperature_scheduler.get_temperature("conclusion")
        
        # 前の思考と結論プロンプトを結合
        conclusion_prompt = self.tokenizer.encode(
            previous_thoughts + "\n\n結論として、", 
            return_tensors='pt'
        ).to(self.device)
        
        outputs = self.model.generate(
            conclusion_prompt,
            max_length=100,
            temperature=temp,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.7
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        conclusion_expressions = [
            "\n\n結論としては、", "\n\nまとめると、", 
            "\n\n総合すると、", "\n\nつまり、"
        ]
        prefix = np.random.choice(conclusion_expressions)
        
        return prefix + response
    
    def _enqueue_thought_chunks(self, text: str, thought_type: str, base_confidence: float):
        """テキストを思考チャンクに分割してキューに入れる"""
        # テキストをセンテンス単位に分割
        sentences = text.split('。')
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # 文末に句点を追加（最後の文以外）
            if i < len(sentences) - 1:
                sentence += '。'
            
            # 確信度を計算（徐々に上昇、多少のノイズあり）
            confidence = base_confidence * (0.8 + 0.2 * (i / len(sentences)))
            confidence *= (0.95 + 0.1 * np.random.random())  # 少しのノイズを加える
            confidence = min(max(confidence, 0.5), 1.0)  # 0.5〜1.0の範囲に収める
            
            chunk = ThoughtChunk(
                text=sentence,
                confidence=confidence,
                type=thought_type,
                timestamp=time.time()
            )
            
            self.thought_queue.put(chunk)
            
            # 思考チャンク間に短い遅延を追加（人間らしさのため）
            time.sleep(0.1 + 0.2 * np.random.random())
    
    def get_next_thought_chunk(self) -> Optional[ThoughtChunk]:
        """次の思考チャンクを取得（非ブロッキング）"""
        if not self.thought_queue.empty():
            return self.thought_queue.get()
        return None
    
    def is_still_thinking(self) -> bool:
        """思考プロセスが進行中かどうか"""
        return self.is_thinking or not self.thought_queue.empty()
    
    def get_thinking_state(self) -> str:
        """現在の思考状態を取得"""
        return self.current_thinking_state

class TemperatureScheduler:
    """生成温度のスケジューリング"""
    def __init__(self):
        self.base_temperatures = {
            "intuitive": 0.8,    # 直感的思考は多様性高め
            "exploratory": 0.7,  # 探索的思考は適度な多様性
            "analytical": 0.5,   # 分析的思考は収束気味
            "insight": 0.9,      # ひらめきは高い多様性
            "conclusion": 0.4    # 結論は収束重視
        }
        self.fluctuation = 0.1    # 温度のゆらぎ範囲
        
    def get_temperature(self, thinking_mode: str) -> float:
        """現在のモードに適した温度を取得（揺らぎあり）"""
        base_temp = self.base_temperatures.get(thinking_mode, 0.7)
        # ランダムな揺らぎを加える
        fluctuation = (np.random.random() * 2 - 1) * self.fluctuation
        return max(0.1, base_temp + fluctuation)

class UncertaintyDetector:
    """不確実性の検出"""
    def __init__(self):
        # 不確実性を示す表現のリスト
        self.uncertainty_phrases = [
            "かもしれない", "可能性がある", "だろう", "だろうか", 
            "思われる", "考えられる", "推測", "推定", "おそらく", 
            "たぶん", "でしょう", "かな", "かしら", "だといいな", 
            "だったら", "もし", "仮に", "必ずしも", "とは限らない"
        ]
    
    def detect_uncertainty(self, text: str) -> float:
        """テキスト内の不確実性レベルを検出（0.0〜1.0）"""
        text_lower = text.lower()
        count = 0
        
        for phrase in self.uncertainty_phrases:
            if phrase in text_lower:
                count += 1
        
        # 不確実性の表現の数に基づいて0.0〜1.0のスコアを計算
        # 長いテキストでは表現が増えるので、テキスト長で正規化
        normalized_count = count / max(1, len(text) / 100)
        uncertainty = min(1.0, normalized_count)
        
        return uncertainty

class RealTimeOutputManager:
    """リアルタイム出力管理"""
    def __init__(self, thought_generator: ThoughtGenerator):
        self.thought_generator = thought_generator
        self.output_buffer = []
        self.last_output_time = 0
        
        # 出力のスタイル設定
        self.output_style = {
            "intuition": {
                "prefix": ["えっと、", "まず、", "最初に思ったのは、"],
                "suffix": ["", "かな。", "と思います。"]
            },
            "logical": {
                "prefix": ["", "考えてみると、", "分析すると、"],
                "suffix": ["", "です。", "と考えられます。"]
            },
            "emotional": {
                "prefix": ["感覚的には、", "なんとなく、", ""],
                "suffix": ["気がします。", "感じがします。", "かもしれません。"]
            },
            "uncertain": {
                "prefix": ["もしかすると、", "たぶん、", "おそらく、"],
                "suffix": ["かもしれません。", "だと思います。", "可能性があります。"]
            },
            "insight": {
                "prefix": ["あ！", "そうか！", "なるほど！"],
                "suffix": ["!", "!!!", "ですね！"]
            }
        }
        
        # 人間らしいフィラー表現
        self.fillers = [
            "えーっと", "あのー", "そうですね", "うーん", 
            "そうそう", "なんていうか", "そういえば"
        ]
        
        # 接続表現
        self.connectors = [
            "それで", "そして", "ただ", "しかし", 
            "一方で", "また", "さらに", "ところで"
        ]
    
    def process_thoughts_to_output(self) -> Optional[str]:
        """思考チャンクを処理して出力文字列を生成"""
        current_time = time.time()
        
        # 出力の間隔を調整（人間らしいリズム）
        if current_time - self.last_output_time < 0.3:
            return None
        
        thought_chunk = self.thought_generator.get_next_thought_chunk()
        if not thought_chunk:
            return None
        
        # 出力のスタイルを設定
        style = self.output_style.get(thought_chunk.type, self.output_style["logical"])
        
        # 出力テキスト作成
        output_text = thought_chunk.text
        
        # 前の出力があれば、自然な接続を検討
        if self.output_buffer:
            # 低確率で接続表現やフィラーを挿入
            if np.random.random() < 0.15:
                output_text = np.random.choice(self.connectors) + "、" + output_text
            elif np.random.random() < 0.1:
                output_text = np.random.choice(self.fillers) + "、" + output_text
        
        # 確信度が低い場合は不確実性を表現
        if thought_chunk.confidence < 0.7:
            style = self.output_style["uncertain"]
            
        # 20%の確率でスタイルを適用（多様性のため）
        if np.random.random() < 0.2:
            prefix = np.random.choice(style["prefix"])
            if prefix:
                output_text = prefix + output_text
        
        if np.random.random() < 0.2:
            suffix = np.random.choice(style["suffix"])
            if suffix and not output_text.endswith(suffix):
                output_text = output_text + suffix
        
        # バッファに追加
        self.output_buffer.append(output_text)
        self.last_output_time = current_time
        
        return output_text
    
    def get_full_output(self) -> str:
        """現在までの出力全体を取得"""
        return " ".join(self.output_buffer)
    
    def add_thinking_indicator(self) -> str:
        """思考中の表現を生成"""
        thinking_state = self.thought_generator.get_thinking_state()
        
        indicators = {
            "initial": ["考え中...", "ちょっと考えます...", "..."],
            "exploring": ["もう少し考えています...", "別の視点から...", "他の可能性は..."],
            "analyzing": ["詳しく分析中...", "掘り下げています...", "より深く考えると..."],
            "concluding": ["まとめています...", "結論としては...", "総合すると..."]
        }
        
        state_indicators = indicators.get(thinking_state, ["考え中..."])
        return np.random.choice(state_indicators)
