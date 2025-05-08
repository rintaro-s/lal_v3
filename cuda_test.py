import torch
import torch.nn as nn
import os
import gc

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Support: {torch.backends.cudnn.version()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"Current GPU Device: {torch.cuda.current_device()}")
        print(f"Device Count: {torch.cuda.device_count()}")
        
        # モデル張量位置テスト
        test_tensor = torch.rand(3, 3)
        test_tensor_gpu = test_tensor.cuda()
        print(f"CPU Tensor Device: {test_tensor.device}")
        print(f"GPU Tensor Device: {test_tensor_gpu.device}")
    else:
        print("CUDA is not available.")
        print("Using CPU only.")

def test_matrix_multiplication():
    """行列乗算の次元テスト"""
    print("\n===== Matrix Multiplication Test =====")
    
    # 適合する次元
    a = torch.rand(4, 8)
    b = torch.rand(8, 16)
    try:
        c = a @ b
        print(f"Success: {a.size()} @ {b.size()} -> {c.size()}")
    except RuntimeError as e:
        print(f"Error: {a.size()} @ {b.size()} - {e}")
    
    # 不適合な次元
    a = torch.rand(4, 8)
    b = torch.rand(16, 32)
    try:
        c = a @ b
        print(f"Success: {a.size()} @ {b.size()} -> {c.size()}")
    except RuntimeError as e:
        print(f"Error: {a.size()} @ {b.size()} - {e}")

    # 不適合だが修正可能
    a = torch.rand(4, 1024)
    b = torch.rand(2048, 32)
    try:
        # 次元を適応させる
        adapter = nn.Linear(1024, 2048)
        a_adapted = adapter(a)
        c = a_adapted @ b
        print(f"Adapted: {a.size()} -> {a_adapted.size()} @ {b.size()} -> {c.size()}")
    except RuntimeError as e:
        print(f"Adaptation error: {e}")

def test_model_dimensions(hidden_sizes=[768, 1024, 2048]):
    """モデルの次元互換性テスト"""
    print("\n===== Model Dimensions Test =====")
    
    for hidden_size in hidden_sizes:
        print(f"\nTesting hidden_size={hidden_size}")
        
        # シンプルなネットワーク定義
        class SimpleNetwork(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.embed = nn.Embedding(1000, hidden_size)
                self.layer1 = nn.Linear(hidden_size, hidden_size*2)
                self.layer2 = nn.Linear(hidden_size*2, hidden_size)
                self.output = nn.Linear(hidden_size, 1000)
                
            def forward(self, x):
                x = self.embed(x)
                x = self.layer1(x)
                x = torch.relu(x)
                x = self.layer2(x)
                return self.output(x)
        
        # モデルを作成
        try:
            model = SimpleNetwork(hidden_size)
            input_ids = torch.randint(0, 1000, (2, 10))
            output = model(input_ids)
            print(f"Model works: input={input_ids.size()}, output={output.size()}")
        except Exception as e:
            print(f"Model error: {e}")

def check_memory_usage():
    """メモリ使用状況の詳細チェック"""
    print("\n===== Memory Usage Check =====")
    
    # システムメモリ
    import psutil
    ram = psutil.virtual_memory()
    print(f"System RAM: {ram.used/1024**3:.2f}GB used / {ram.total/1024**3:.2f}GB total ({ram.percent}%)")
    
    # GPUメモリ
    if torch.cuda.is_available():
        # CUDA初期化
        torch.cuda.synchronize()
        
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            print(f"GPU {i}: {allocated_mem:.2f}GB allocated / {reserved_mem:.2f}GB reserved / {total_mem:.2f}GB total")
            
        # メモリ使用テスト
        print("\nTesting GPU memory allocation...")
        try:
            tensors = []
            for i in range(5):
                size_gb = 0.1 * (i + 1)  # 増分でテスト (0.1GB, 0.2GB, ...)
                size_bytes = int(size_gb * 1024**3 / 4)  # float32のサイズを考慮
                dim = int(size_bytes ** 0.5)  # 2D次元計算
                
                print(f"Allocating {size_gb:.1f}GB tensor ({dim}x{dim})...")
                tensor = torch.zeros(dim, dim, device='cuda')
                tensors.append(tensor)
                
                allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
                print(f"  Current allocated: {allocated_mem:.2f}GB")
        except RuntimeError as e:
            print(f"Memory allocation failed: {e}")
        finally:
            # クリーンアップ
            tensors = None
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    check_cuda()
    
    # デバイス間転送テスト
    print("\n===== Device Transfer Test =====")
    x = torch.rand(3, 3)
    print(f"Original tensor device: {x.device}")
    
    if torch.cuda.is_available():
        y = x.to("cuda")
        print(f"Transferred tensor device: {y.device}")
        
        # CPUに戻す
        z = y.to("cpu")
        print(f"Transferred back to CPU: {z.device}")
    
    # 追加テスト実行
    test_matrix_multiplication()
    test_model_dimensions()
    check_memory_usage()
    
    print("\nTest completed successfully!")