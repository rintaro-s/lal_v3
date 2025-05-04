import math
import torch
import numpy as np
from transformers import AutoTokenizer
import psutil

def estimate_distillation_time(
    num_examples: int = 1000,
    teacher_model_name: str = "elyza/ELYZA-Thinking-1.0-Qwen-32B",
    batch_size: int = 4,
    num_epochs: int = 3,
    max_length: int = 512,
    gpu_name: str = "RTX 5070Ti",
    quantize: bool = True,
    cpu_offload: bool = True,
    gradient_accumulation_steps: int = 8
):
    """知識蒸留にかかる時間をより現実的に見積もる"""
    print(f"=== 知識蒸留時間の見積もり ({gpu_name}、量子化={quantize}、CPU Offload={cpu_offload}) ===")
    
    # RAM情報を取得
    ram = psutil.virtual_memory()
    print(f"システムRAM: 合計 {ram.total/1024**3:.1f}GB、空き {ram.available/1024**3:.1f}GB")
    
    # GPU情報を取得（可能な場合）
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} (メモリ: {gpu_mem:.1f}GB)")
    
    # データ生成にかかる時間の見積もり
    # 量子化と大型モデルに基づく調整
    if quantize:
        # 量子化によってややスピードアップ
        example_gen_time_per_item = 12  # 秒
    else:
        # VRAM制限に起因するスワッピングで遅くなる可能性
        example_gen_time_per_item = 20  # 秒
    
    # バッチ処理の効果を考慮
    example_gen_time_per_item /= math.sqrt(batch_size)
    
    # CPU offloadの効果を考慮
    if cpu_offload:
        # CPU offloadは性能を下げるがOOM回避に必要
        example_gen_time_per_item *= 1.5
    
    total_example_gen_time = num_examples * example_gen_time_per_item
    hours, remainder = divmod(total_example_gen_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"1. 蒸留データの生成時間: {int(hours)}時間 {int(minutes)}分 {int(seconds)}秒")
    
    # トークナイザーの取得（ボキャブラリサイズ推定用）
    try:
        tokenizer = AutoTokenizer.from_pretrained("elyza/elyza-japanese-llama-2-7b")
        vocab_size = len(tokenizer)
    except:
        vocab_size = 32000  # 推定値
    
    # モデルサイズの計算（パラメータ数）
    embedding_dim = 768
    hidden_dim = 1024
    memory_size = 1000
    
    # 主要なパラメータ数を計算
    embedding_params = vocab_size * embedding_dim
    neuron_module_params = 6 * (embedding_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * embedding_dim)
    memory_params = memory_size * hidden_dim * 2
    output_params = embedding_dim * vocab_size
    
    total_params = embedding_params + neuron_module_params + memory_params + output_params
    print(f"生成するモデルのパラメータ数: 約{total_params / 1e6:.2f}M")
    
    # 1エポックあたりの学習時間を見積もる
    # RTX 5070Ti/3080/4090などGPUの性能を考慮
    if "3090" in gpu_name or "4090" in gpu_name:
        gpu_factor = 0.5  # より高速
    elif "3080" in gpu_name or "3070" in gpu_name:
        gpu_factor = 1.0  # 標準
    else:
        gpu_factor = 1.2  # やや遅い
    
    # 勾配累積の効果（効率性向上）
    ga_efficiency = 1.0 - math.log(gradient_accumulation_steps) * 0.1
    
    # バッチあたり処理時間の再計算（より現実的な値）
    examples_per_second = 1.0 / (batch_size * 0.5 * gpu_factor * ga_efficiency)
    seconds_per_epoch = num_examples / examples_per_second
    
    epoch_hours, remainder = divmod(seconds_per_epoch, 3600)
    epoch_minutes, epoch_seconds = divmod(remainder, 60)
    print(f"2. 1エポックあたりの学習時間: {int(epoch_hours)}時間 {int(epoch_minutes)}分 {int(epoch_seconds)}秒")
    
    # 全エポックの学習時間
    total_training_time = seconds_per_epoch * num_epochs
    training_hours, remainder = divmod(total_training_time, 3600)
    training_minutes, training_seconds = divmod(remainder, 60)
    print(f"3. 全エポック({num_epochs}エポック)の学習時間: {int(training_hours)}時間 {int(training_minutes)}分 {int(training_seconds)}秒")
    
    # モデル保存、データ読み込み、チェックポイントなどのオーバーヘッド
    overhead_hours = (hours + training_hours) * 0.05  # 全体の約5%
    overhead_minutes = int((overhead_hours - int(overhead_hours)) * 60)
    print(f"4. その他オーバーヘッド: 約{int(overhead_hours)}時間 {overhead_minutes}分")
    
    # 合計時間
    total_time = total_example_gen_time + total_training_time + (overhead_hours * 3600)
    total_hours, remainder = divmod(total_time, 3600)
    total_minutes, total_seconds = divmod(remainder, 60)
    print(f"5. 合計見積もり時間: {int(total_hours)}時間 {int(total_minutes)}分 {int(total_seconds)}秒")
    
    # 日数に換算
    days, hours = divmod(total_hours, 24)
    print(f"   (約 {int(days)}日 {int(hours)}時間)")
    
    # 最悪のケースの見積もり（設定にもよる）
    worst_case_factor = 1.5
    worst_case_hours = total_hours * worst_case_factor
    worst_case_days, worst_case_hours_remainder = divmod(worst_case_hours, 24)
    print(f"6. 最悪ケース見積もり: 約 {int(worst_case_days)}日 {int(worst_case_hours_remainder)}時間")
    
    # ELYZA-Thinkingのスペックを十分に継承するための推奨設定
    if num_examples < 3000 or num_epochs < 3:
        print("\n[推奨] ELYZA-Thinkingの能力をより継承するには:")
        print(f"  - サンプル数を3000～5000に増やす (現在: {num_examples})")
        print(f"  - エポック数を3～5に設定する (現在: {num_epochs})")
        
        # 推奨設定での見積もり時間
        recommended_examples = max(num_examples, 3000)
        recommended_epochs = max(num_epochs, 3)
        rec_total_time = (recommended_examples / num_examples) * (recommended_epochs / num_epochs) * total_time
        rec_days, rec_hours = divmod(rec_total_time / 3600, 24)
        rec_hours, rec_minutes = divmod(rec_hours * 60, 60)
        print(f"  - 推奨設定での予想所要時間: 約 {int(rec_days)}日 {int(rec_hours)}時間 {int(rec_minutes)}分")
    
    # 実行環境に応じた注意事項
    ram_warning_threshold = 32  # GB
    if ram.available / 1024**3 < ram_warning_threshold:
        print("\n[警告] 空きRAMが少ないため、処理速度が低下する可能性があります。")
        print("  - 不要なアプリケーションを終了するか、スワップ領域を増やしてください。")
    
    if torch.cuda.is_available() and gpu_mem < 16 and not quantize:
        print("\n[警告] GPUメモリが16GB未満で量子化を使用していません。")
        print("  - --quantize オプションを有効にすることを強くお勧めします。")
    
    return {
        "data_generation_hours": total_example_gen_time / 3600,
        "training_hours": total_training_time / 3600,
        "overhead_hours": overhead_hours,
        "total_hours": total_time / 3600,
        "days": days + (hours / 24),
        "worst_case_days": worst_case_days + (worst_case_hours_remainder / 24)
    }

if __name__ == "__main__":
    # デフォルト設定での見積もり
    default_estimate = estimate_distillation_time()
    print("\n")
    
    # より少ないサンプル数での見積もり
    small_estimate = estimate_distillation_time(num_examples=500, num_epochs=2)
    print("\n")
    
    # より多いサンプル数での見積もり
    large_estimate = estimate_distillation_time(num_examples=5000, num_epochs=5)
    print("\n")
    
    # ELYZA-Thinkingのスペックを最大限継承するための設定
    optimal_estimate = estimate_distillation_time(num_examples=5000, num_epochs=5, batch_size=2, gradient_accumulation_steps=16)
