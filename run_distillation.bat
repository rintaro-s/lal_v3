@echo off
echo GGUFモデルを使用した知識蒸留を開始します（GPU使用）
python main.py distill ^
  --use_gguf ^
  --gguf_model_path "C:\Users\s-rin\.lmstudio\models\mmnga\cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-gguf\cyberagent-DeepSeek-R1-Distill-Qwen-14B-Japanese-Q6_K.gguf" ^
  --thinking_llm ^
  --focus_subjects highschool,electronics,it ^
  --imouto_mode ^
  --num_examples 5000 ^
  --num_epochs 5 ^
  --batch_size 2 ^
  --gradient_accumulation 16 ^
  --save_hf_format ^
  --hf_model_name "lorinta/lal_v3" ^
  --gguf_gpu_layers -1 ^
  --gguf_n_gpu 1 ^
  --gguf_n_batch 512

if %errorlevel% neq 0 (
  echo エラーが発生しました。ログを確認してください。
  pause
) else (
  echo 処理が正常に完了しました。
)
