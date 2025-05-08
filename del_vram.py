import torch

def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("VRAM cleared.")
    else:
        print("CUDA is not available.")

# 実行例
clear_vram()