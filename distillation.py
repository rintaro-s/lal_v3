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
        use_cpu_only: bool = False
    ):
        self.logger = logging.getLogger(__name__)
        self.teacher_model_name = teacher_model_name
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
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

    def filter_by_subjects(self, questions: List[str], subjects: List[str]) -> List[str]:
        """特定の分野に関連する質問をフィルタリング"""
        # 分野ごとのキーワード
        subject_keywords = {
            "highschool": ["高校", "学校", "勉強", "数学", "国語", "英語", "理科", "社会", "歴史", "地理", 
                        "物理", "化学", "生物", "地学", "現代社会", "倫理", "高校数学", "受験", "積分", "微分",
                        "漸化式", "方程式", "確率", "統計", "ベクトル", "行列", "座標", "指数", "対数", "三角関数"],
            "electronics": ["電気", "電子", "回路", "半導体", "抵抗", "コンデンサ", "インダクタ", "トランジスタ", 
                          "ダイオード", "CPU", "マイコン", "電圧", "電流", "電力", "オームの法則", "キルヒホッフ",
                          "論理ゲート", "電磁誘導", "アナログ", "デジタル"],
            "it": ["プログラミング", "コンピュータ", "アルゴリズム", "データ構造", "ウェブ", "インターネット",
                 "セキュリティ", "ネットワーク", "クラウド", "サーバー", "データベース", "Arduino", "HTML", "マイコン",
                 "JavaScript", "Python", "Java", "C++", "C#", "AI", "人工知能", "機械学習", "深層学習", "SQL",
                 "Git", "API"]
        }
        
        # 指定された分野に関連するキーワードのリストを作成
        target_keywords = []
        for subject in subjects:
            if subject in subject_keywords:
                target_keywords.extend(subject_keywords[subject])
        
        self.logger.info(f"フィルタリングに使用するキーワード数: {len(target_keywords)}")
        
        if not target_keywords:
            self.logger.warning("有効なキーワードが見つかりません。フィルタリングをスキップします。")
            return questions
        
        # キーワードを含む質問をフィルタリング
        filtered_questions = []
        for question in questions:
            if any(keyword in question for keyword in target_keywords):
                filtered_questions.append(question)
        
        # フィルタリング後、質問が少なすぎる場合はランダムに追加
        if len(filtered_questions) < 0.5 * len(questions) and len(filtered_questions) > 0:
            self.logger.warning(f"フィルタリング後の質問が少なすぎます: {len(filtered_questions)}件。ランダムに追加します。")
            remaining = [q for q in questions if q not in filtered_questions]
            additional = random.sample(remaining, min(len(remaining), len(questions) - len(filtered_questions)))
            filtered_questions.extend(additional)
        
        # フィルタリング結果が空の場合は元の質問を使用
        if not filtered_questions:
            self.logger.warning("フィルタリング結果が空です。元の質問を使用します。")
            return questions
        
        self.logger.info(f"Filtered questions: {len(filtered_questions)} out of {len(questions)} original questions")
        return filtered_questions

    def generate_questions_if_needed(self, questions_file: str, num_samples: int) -> str:
        """質問ファイルが存在しない場合、自動生成する"""
        self.logger.info("自動的に質問データを生成します")
        
        try:
            # 質問リストの基本テーマ
            question_themes = [
                # 高校教科: 数学を強化
                "数学の微分・積分の基本概念",
                "数学の極限の考え方",
                "数学における三角関数の応用",
                "数列の漸化式と一般項",
                "ベクトルの内積と外積の違い",
                "確率統計の基本定理",
                "複素数平面の図形的意味",
                "二次方程式と判別式の関係",
                "指数関数と対数関数の性質",
                "空間座標系と立体図形",
                "行列の固有値と固有ベクトル",
                
                #高校物理
                "力学における運動の法則",
                "エネルギー保存の法則",
                "波動の性質と干渉",
                "電磁気学におけるクーロンの法則",
                "熱力学の基本法則",
                "光の屈折と全反射",
                "原子核の構造と放射線",
                "電流と磁場の関係",
                "運動量保存の法則",
                "振動と波の関係",


                # その他の高校教科
                "英語の関係代名詞の使い方",
                "日本史における江戸時代の特徴",
                "化学の酸化と還元反応",
                "物理学における運動方程式",
                "古典文学「源氏物語」の魅力",
                "世界史における産業革命の影響",
                "生物の細胞分裂の過程",
                "地理学における気候帯の分類",
                "現代社会における憲法の役割",
                
                # 電子工学
                "トランジスタの基本的な仕組み",
                "オームの法則",
                "電子回路の並列接続と直列接続の違い",
                "半導体の種類と特性",
                "アナログ回路とデジタル回路の違い",
                "マイクロコントローラの役割と機能",
                "電圧と電流の関係性",
                "ダイオードの整流作用",
                "論理ゲートの種類と機能",
                "電磁誘導の原理",
                
                # IT・プログラミング
                "プログラミング言語Pythonの特徴",
                "データベースのSQLクエリの基本構文",
                "オブジェクト指向プログラミングの概念",
                "人工知能の機械学習とディープラーニングの違い",
                "ウェブセキュリティにおけるXSSとは",
                "クラウドコンピューティングのメリット",
                "アルゴリズムの計算量とビッグO記法",
                "ネットワークのTCP/IPプロトコル",
                "Gitのような分散型バージョン管理システムの利点",
                "APIの役割と活用方法"
            ]
            
            self.logger.info(f"基本的な質問テーマ数: {len(question_themes)}")
            
            # 質問の生成パターン (自然な質問文になるように設計)
            question_patterns = [
                "{}について教えてください。",
                "{}を詳しく説明してもらえますか？",
                "{}とは何ですか？",
                "{}の基本概念を教えてください。",
                "{}について知りたいです。",
                "{}の仕組みはどうなっていますか？"
            ]
            
            # 妹口調用の追加パターン
            imouto_patterns = [
                "{}について教えて欲しいな、お兄ちゃん。",
                "お兄ちゃん、{}ってどういうこと？",
                "{}のこと、わかりやすく説明してほしいな。",
                "お兄ちゃん、{}について教えてくれる？",
                "{}って難しそうだけど、お兄ちゃんなら分かるよね？"
            ]
            
            self.logger.info(f"質問生成パターン数: {len(question_patterns)} (通常) + {len(imouto_patterns)} (妹口調)")
            
            # 質問を生成
            questions = []
            self.logger.info("質問の生成を開始します...")
            
            # 進捗状況の表示準備
            total_themes = len(question_themes)
            
            # 通常の質問を生成
            for i, theme in enumerate(question_themes):
                # コンソール上に進捗バーを表示
                progress = (i + 1) / total_themes
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '=' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout.write(f"\r生成中: [{bar}] {progress*100:.1f}% ({i+1}/{total_themes})")
                sys.stdout.flush()
                
                # このテーマに対して複数のパターンを適用
                theme_questions = []
                for pattern in question_patterns:
                    theme_questions.append(pattern.format(theme))
                
                # ランダムに最大2つを選んで追加
                selected = random.sample(theme_questions, min(2, len(theme_questions)))
                questions.extend(selected)
                
                # 妹口調テーマも追加
                if self.imouto_mode and random.random() < 0.3:  # 30%の確率で妹口調の質問も追加
                    imouto_pattern = random.choice(imouto_patterns)
                    questions.append(imouto_pattern.format(theme))
                
                if len(questions) >= num_samples:
                    break
            
            print()  # 進捗バーの後に改行
            
            # 必要数に達しない場合、テーマをバリエーション
            if len(questions) < num_samples:
                self.logger.info(f"追加の質問を生成します。現在: {len(questions)}/{num_samples}")
                
                # テーマのバリエーションを作成
                variations = []
                for theme in question_themes:
                    # テーマの応用的なバリエーション
                    variations.append(f"{theme}の応用例")
                    variations.append(f"{theme}の歴史")
                    variations.append(f"{theme}の重要性")
                    variations.append(f"{theme}の最新動向")
                    variations.append(f"{theme}の学習方法")
                
                # バリエーションからランダムに選択して質問を生成
                random.shuffle(variations)
                for theme in variations:
                    pattern = random.choice(question_patterns)
                    new_q = pattern.format(theme)
                    
                    if new_q not in questions:
                        questions.append(new_q)
                    
                    if len(questions) >= num_samples:
                        break
            
            # 質問をランダムに並び替え
            random.shuffle(questions)
            
            # 質問数を最終確認
            self.logger.info(f"生成された質問数: {len(questions)}")
            if len(questions) < num_samples:
                self.logger.warning(f"要求された {num_samples} 個の質問を生成できませんでしたが、{len(questions)}個生成しました")
            
            # 質問が少なすぎる場合はデフォルト質問を追加
            if len(questions) < 3:
                self.logger.warning("生成された質問が少なすぎます。デフォルト質問を追加します")
                default_questions = [
                    "数学の基本概念について教えてください。",
                    "物理学の法則について説明してください。",
                    "プログラミングの入門方法を教えてください。",
                    "電子工学の基礎を説明してください。",
                    "コンピュータの仕組みを教えてください。"
                ]
                questions.extend(default_questions)
            
            # 質問をファイルに保存
            output_path = questions_file
            dir_path = os.path.dirname(output_path)
            if dir_path and not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path)
                except Exception as e:
                    self.logger.error(f"ディレクトリ作成エラー: {e}")
                    # 絶対パスの代わりにカレントディレクトリの相対パスを使用
                    output_path = os.path.basename(questions_file)
            
            # ファイル書き込みを処理
            self.logger.info(f"質問をファイルに書き込み中: {output_path}")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for q in questions[:min(len(questions), num_samples)]:
                        f.write(q + '\n')
                self.logger.info(f"質問ファイルの書き込みが完了しました: {output_path}")
                return output_path  # 正常なパスを返す
            except Exception as e:
                self.logger.error(f"ファイル書き込み中にエラーが発生: {e}")
                # 緊急用の代替ファイルに書き込み
                fallback_path = "questions_emergency.txt"
                try:
                    with open(fallback_path, 'w', encoding='utf-8') as f:
                        for q in questions[:min(len(questions), num_samples)]:
                            f.write(q + '\n')
                    self.logger.info(f"代替ファイルに保存しました: {fallback_path}")
                    return fallback_path  # 代替パスを返す
                except Exception as e2:
                    self.logger.critical(f"代替ファイル保存も失敗: {e2}")
                    # 最後の手段として、カレントディレクトリに保存
                    last_resort_path = "last_resort_questions.txt"
                    with open(last_resort_path, 'w', encoding='utf-8') as f:
                        f.write("緊急質問です。\n")
                    return last_resort_path
                
        except Exception as e:
            self.logger.error(f"質問生成中に予期しないエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # エラー回復: 最低限の質問セットを生成して返す
            emergency_questions = [
                "数学について教えてください",
                "英語学習のコツは何ですか",
                "プログラミングの基礎を説明してください",
                "電子工学とは何ですか",
                "人工知能の仕組みを教えてください"
            ]
            
            # 緊急用ファイルに書き込み
            emergency_path = "questions_error_recovery.txt"
            try:
                with open(emergency_path, 'w', encoding='utf-8') as f:
                    for q in emergency_questions:
                        f.write(q + '\n')
                
                self.logger.info(f"エラー回復: 緊急質問ファイルを生成しました: {emergency_path}")
                return emergency_path  # 緊急パスを返す
            except Exception as e2:
                self.logger.critical(f"緊急ファイル保存も失敗: {e2}")
                # 最終手段として、直接文字列を返す代わりにハードコードされたファイルを作成
                final_emergency_path = "final_emergency_questions.txt"
                with open(final_emergency_path, 'w', encoding='utf-8') as f:
                    f.write("最終緊急質問です。\n")
                return final_emergency_path

    def prepare_distillation_data(self, questions_file: str, output_file: str, num_samples: int = 100, 
                                batch_size: int = 4, cache_to_ram: bool = True, focus_subjects: List[str] = None,
                                imouto_mode: bool = True, thinking_llm: bool = False):
        """教師モデルからの出力を生成してデータを準備（バッチ処理対応）"""
        # import jsonをここでも再確認（これはエラー防止のための冗長コード）
        import json
        
        self.logger.info(f"Preparing distillation data from {questions_file}")
        
        # 質問ファイルが存在しない場合は自動生成
        if not questions_file or not os.path.exists(questions_file):
            self.logger.info(f"Questions file {questions_file} not found, generating automatically")
            questions_file = self.generate_questions_if_needed(questions_file, num_samples)
            
            # 質問ファイルが正常に生成されたか確認
            if not questions_file or not os.path.exists(questions_file):
                self.logger.error(f"質問ファイルの生成に失敗しました")
                # 緊急対応として空のファイルを作成
                questions_file = "emergency_questions.txt"
                with open(questions_file, "w", encoding="utf-8") as f:
                    f.write("数学について教えてください\n")
                    f.write("英語の基本文法を説明してください\n")
                    f.write("プログラミングの始め方について教えてください\n")
                self.logger.info(f"緊急質問ファイルを作成しました: {questions_file}")
        
        # 質問を読み込む
        questions = []
        try:
            if os.path.exists(questions_file):
                with open(questions_file, 'r', encoding='utf-8') as f:
                    questions = [line.strip() for line in f if line.strip()]  # 空行を除外
            else:
                self.logger.error(f"質問ファイルが存在しません: {questions_file}")
                
            if not questions:  # 空のリストの場合の対応
                self.logger.warning(f"質問ファイル {questions_file} は空か読み込めません。デフォルトの質問を使用します。")
                questions = [
                    "数学の基本概念について教えてください。",
                    "物理学の法則について説明してください。",
                    "プログラミングの入門方法を教えてください。"
                ]
        except Exception as e:
            self.logger.error(f"質問ファイル読み込み中にエラーが発生: {e}")
            # エラー回復のためのデフォルト質問を設定
            questions = [
                "数学の基本概念について教えてください。",
                "物理学の法則について説明してください。",
                "プログラミングの入門方法を教えてください。"
            ]
            self.logger.info(f"エラー回復: デフォルトの質問 {len(questions)}件を使用します")
        
        self.logger.info(f"読み込んだ質問数: {len(questions)}")
        
        # 重点分野にフィルタリング - Noneチェック追加
        if focus_subjects and isinstance(focus_subjects, list) and questions:
            try:
                filtered_questions = self.filter_by_subjects(questions, focus_subjects)
                if filtered_questions:
                    questions = filtered_questions
                    self.logger.info(f"フィルタリング後の質問数: {len(questions)}")
                else:
                    self.logger.warning(f"フィルタリングの結果が空になりました。元の質問を使用します。")
            except Exception as e:
                self.logger.error(f"フィルタリング中にエラーが発生: {e}")
                # エラーが発生した場合は元の質問を使用
        
        # サンプル数を制限（型チェック追加）
        if questions and isinstance(num_samples, int) and num_samples < len(questions):
            try:
                questions = random.sample(questions, num_samples)
                self.logger.info(f"サンプリング後の質問数: {len(questions)}")
            except Exception as e:
                self.logger.error(f"サンプリング中にエラーが発生: {e}")
                # エラー時は全ての質問を使用
        
        # 質問が最低1つあることを保証
        if not questions:
            self.logger.critical("有効な質問が見つかりませんでした。緊急質問セットを使用します。")
            # 最小限の緊急質問セットを設定
            questions = [
                "基本的な質問に答えてください。",
                "数学について教えてください。",
                "科学の基礎概念を説明してください。"
            ]
            self.logger.info(f"緊急質問セット {len(questions)}件を使用します。")
        
        distillation_data = []
        
        # バッチ処理で効率化
        total_batches = math.ceil(len(questions) / batch_size)
        start_time = time.time()
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            current_batch = i // batch_size + 1
            
            # 進捗状況の表示
            progress = current_batch / total_batches
            bar_length = 30
            filled_length = int(bar_length * progress)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            
            # 経過時間と推定残り時間を計算
            elapsed = time.time() - start_time
            if current_batch > 1:  # 1バッチ目は推定に使えない
                estimated_total = elapsed / (current_batch - 1) * total_batches
                remaining = max(0, estimated_total - elapsed)
                # フォーマット関数を使用して読みやすく表示
                remaining_str = self._format_time_display(remaining)
                elapsed_str = self._format_time_display(elapsed)
            else:
                remaining_str = "計算中..."
                elapsed_str = "0:00:00"
                
            # 表示行末にスペースを追加して、ログとの区切りを明確に
            progress_message = f"\r進捗: [{bar}] {progress*100:.1f}% - バッチ {current_batch}/{total_batches} - 経過: {elapsed_str} - 残り: {remaining_str}    "
            sys.stdout.write(progress_message)
            sys.stdout.flush()
            
            self.logger.info(f"Processing batch {current_batch}/{total_batches}")
            
            # 妹モードの場合はプロンプトテンプレートを変更
            template = self.imouto_template if imouto_mode else self.thinking_template
            batch_prompts = [template.format(question=q) for q in batch_questions]
            
            if self.use_gguf:
                # ...existing code...
                pass
            elif self.use_lmstudio:
                # ...existing code...
                pass
            else:
                # 通常の教師モデルを使用する場合
                # 教師モデルのチェック
                if self.teacher_model is None:
                    self.logger.error("Teacher model is not initialized")
                    raise ValueError("Teacher model is not initialized. Please check your configuration.")
                
                try:
                    # パディングサイドが適切に設定されていることを確認
                    if hasattr(self.tokenizer, 'padding_side') and self.tokenizer.padding_side != 'left':
                        self.logger.info("トークナイザーのパディング方向を左側に設定します（バッチ処理時）")
                        self.tokenizer.padding_side = 'left'
                    
                    batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
                    
                    with torch.no_grad():
                        # 教師モデルから出力を生成
                        outputs = self.teacher_model.generate(
                            input_ids=batch_inputs.input_ids,
                            attention_mask=batch_inputs.attention_mask,
                            max_new_tokens=512,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                        
                        # 出力をデコード
                        for j, output in enumerate(outputs):
                            self.logger.info(f"Processing output {j+1}/{len(outputs)} in batch {i//batch_size + 1}")
                            teacher_output = self.tokenizer.decode(output, skip_special_tokens=True)
                            
                            # 入力プロンプトと出力から質問部分を取得
                            original_question = batch_questions[j % len(batch_questions)]
                            
                            distillation_data.append({
                                "input": original_question,
                                "output": teacher_output
                            })
                    
                    # バッチ処理後にメモリを解放
                    del batch_inputs, outputs
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                except Exception as e:
                    self.logger.error(f"Error generating teacher model outputs: {e}")
                    import traceback
                    self.logger.error(f"詳細なエラー: {traceback.format_exc()}")
                    # エラーが発生しても次のバッチに進む
                    continue
            
            # 定期的に進捗を保存（万が一のクラッシュに備える）
            if (i // batch_size) % 10 == 0 and i > 0:
                temp_file = f"{output_file}.temp_{i}"
                try:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(distillation_data, f, ensure_ascii=False, indent=2)
                    self.logger.info(f"Saved progress to temporary file: {temp_file}")
                except Exception as e:
                    self.logger.error(f"一時ファイル保存エラー: {e}")
        
        print()  # 進捗バー後の改行
        
        # 出力ディレクトリの確保
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self.logger.info(f"出力ディレクトリを作成しました: {output_dir}")
            except Exception as e:
                self.logger.error(f"ディレクトリ作成エラー: {e}")
                # 出力先をカレントディレクトリに変更
                output_file = os.path.basename(output_file)
                self.logger.info(f"出力先をカレントディレクトリに変更: {output_file}")
        
        # 結果を保存
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(distillation_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"結果ファイル保存エラー: {e}")
            # バックアップファイルを試す
            backup_file = f"{os.path.basename(output_file)}.backup"
            try:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(distillation_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"バックアップファイルに保存: {backup_file}")
                output_file = backup_file  # 出力ファイルパスを更新
            except Exception as e2:
                self.logger.critical(f"バックアップファイルも保存失敗: {e2}")
                raise
        
        # 完了メッセージと合計時間
        total_time = time.time() - start_time
        total_time_str = self._format_time_display(total_time)
        self.logger.info(f"Distillation data saved to {output_file} with {len(distillation_data)} examples")
        self.logger.info(f"Total processing time: {total_time_str}")
        
        # 検証用データも作成
        val_size = min(int(len(distillation_data) * 0.1), 100)  # 全体の10%か100件のいずれか小さい方
        if val_size > 0:
            val_data = random.sample(distillation_data, val_size)
            val_file = output_file.replace('.json', '_val.json')
            try:
                with open(val_file, 'w', encoding='utf-8') as f:
                    json.dump(val_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Validation data saved to {val_file} with {len(val_data)} examples")
            except Exception as e:
                self.logger.error(f"検証データ保存エラー: {e}")
        
        return output_file

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
                    # バッチを学習する
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # フォワードパス
                    outputs = self.student_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    
                    loss = outputs.loss / gradient_accumulation_steps
                    loss.backward()
                    
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
                        
                        val_loss = val_outputs.loss
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
            
            # モデルとトークナイザーの保存
            self.student_model.save_pretrained(output_dir)
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
