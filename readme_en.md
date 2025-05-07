# LLM System with Human-like Thought Processes

This project implements a language model system that mimics human-like thought processes. Its main features include the ability to generate responses while thinking in real-time and a parallel processing system modeled after multiple brain functions.

## Key Features

1. Multi-Process Brain Model
   - Parallel processing of left brain (logical thinking) and right brain (intuitive thinking)
   - Executive function and decision-making by the frontal lobe
   - Memory management by the hippocampus module
   - Emotional expression through emotion processing module

2. Multi-layer Memory System
   - Hierarchical structure of working memory, short-term memory, and long-term memory
   - Memory management based on Miller's Law (7±2)
   - Pattern recognition through procedural memory

3. Real-time Thinking and Output Mechanism
   - Output begins without waiting for thought completion
   - Natural expression of uncertainty
   - Thought streaming mechanism

## Usage

### Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

PyTorch nightly version configuration:
```bash
# When using PyTorch nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

Dependencies for maximizing GPU functionality in Windows environment:
```bash
pip install xformers>=0.0.20 accelerate>=0.20.0 bitsandbytes>=0.40.0
```

Experimental use of triton in Windows environment (advanced functionality):
```bash
pip install triton-windows>=2.0.0 accelerate>=0.20.0 bitsandbytes>=0.40.0
```

Configuration for direct GPU access:
```bash
pip install accelerate>=0.20.0 bitsandbytes>=0.40.0 safetensors>=0.3.0
```

Additional installations possible in Linux environment:
```bash
pip install triton>=2.0.0 flash-attn>=2.3.0  # Linux only
```

These packages are only supported in specific GPU (CUDA) environments. If incompatible, use CPU fallback mode.

Installation in development mode:
```bash
pip install -e .
```

### Running Knowledge Distillation

Basic usage:
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B
```

Recommended settings to better inherit ELYZA-Thinking performance:
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --num_examples 5000 --num_epochs 5 --batch_size 2 --gradient_accumulation 16 --max_length 768
```

For environments without GPU support, lighter alternative models can be used:
```bash
python main.py distill --teacher_model elyza/elyza-japanese-llama-2-7b --use_cpu_only
```

Saving the model in Hugging Face format (can be uploaded to model hub):
```bash
python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --save_hf_format --hf_model_name "your-username/lal-brain-model"
```

### Model Selection Guide

| Model Name | Size | Features | Required GPU RAM | PyTorch Nightly Support |
|---------|--------|------|-------------|-------------|
| elyza/ELYZA-Thinking-1.0-Qwen-32B | 32B | Best quality & thinking ability | 12GB+ | Requires direct GPU config |
| elyza/Llama-3-ELYZA-JP-8B | 8B | High performance & Japanese specialized | 8GB+ | Full support |
| microsoft/Phi-4-reasoning | 4B | Efficient & reasoning focused | 6GB+ | Full support |
| elyza/ELYZA-japanese-Llama-2-13b-instruct | 13B | High quality & Japanese | 10GB+ | Full support |
| elyza/elyza-japanese-llama-2-7b | 7B | Balanced & lightweight | 8GB+ | Full support |
| stabilityai/stablelm-base-alpha-7b | 7B | Lightweight & English | 8GB+ | Full support |
| cyberagent/calm2-7b | 7B | Japanese specialized & lightweight | 8GB+ | Full support |

### Parameter Details

| Parameter | Description | Default Value |
|------------|------|------------|
| `--num_examples` | Number of distillation data to generate | 5000 |
| `--batch_size` | Batch size | 2 |
| `--num_epochs` | Number of epochs | 5 |
| `--gradient_accumulation` | Gradient accumulation steps | 16 |
| `--quantize` | 4-bit quantization of teacher model | Enabled |
| `--cpu_offload` | Offload part of the model to CPU | Enabled |
| `--max_length` | Maximum sequence length | 768 |
| `--save_hf_format` | Save in Hugging Face format | Enabled |
| `--hf_model_name` | Model name for Hugging Face | "lal-brain-model" |

### Known Issues and Solutions

- **Tokenizers version error**: If you see `tokenizers>=0.13.3 is required` error, upgrade the tokenizers package:
  ```bash
  pip install --upgrade tokenizers>=0.13.3
  ```

- **Triton module error (Windows environment)**: If you see `No module named 'triton'` error:
  ```bash
  # Normal Windows optimization mode (recommended)
  python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --windows_mode
  
  # Experimental: when using triton-windows
  pip install triton-windows>=2.0.0
  python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --use_triton_windows
  ```
  Or use a fully Windows-compatible model:
  ```bash
  python main.py distill --teacher_model elyza/elyza-japanese-llama-2-7b
  ```

- **GPU access error (when using PyTorch nightly)**: With PyTorch nightly versions, some optimization libraries (triton, xformers) might not be available. Instead, configure direct access to the GPU:
  ```bash
  # Command line option
  python main.py distill --teacher_model elyza/ELYZA-Thinking-1.0-Qwen-32B --use_direct_gpu
  
  # Or alternative model
  python main.py distill --teacher_model elyza/elyza-japanese-llama-2-7b
  ```

- **CUDA errors**: If CUDA-related errors occur, use a PyTorch version compatible with your system's CUDA version:
  ```bash
  # Install PyTorch compatible with CUDA
  pip install torch==1.13.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
  ```

- **CUDA compatibility error**: If your GPU/CUDA version is not supported:
  ```bash
  python main.py distill --use_cpu_only --teacher_model elyza/elyza-japanese-llama-2-7b
  ```

- **Out of memory**: If GPU memory out of error occurs, adjust the following parameters:
  ```bash
  python main.py distill --batch_size 1 --gradient_accumulation 32 --quantize
  ```

- **Out of memory error**: If you run out of memory when loading the GTP model:
  ```bash
  python main.py distill --batch_size 1 --gradient_accumulation 32 --quantize
  ```

### Uploading Hugging Face Models

How to upload the trained model to Hugging Face Hub:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import HfApi; api = HfApi(); api.upload_folder(folder_path='./models/lal-brain-model', repo_id='YOUR_USERNAME/lal-brain-model')"
```

### Chat Mode

```bash
python main.py chat --model_path ./models/brain_model_best.pt
```

When loading from Hugging Face:
```bash
python main.py chat --model_path YOUR_USERNAME/lal-brain-model
```

## System Structure

- `brain_model.py`: Brain model implementation
- `memory_system.py`: Multi-layer memory system implementation
- `real_time_thoughts.py`: Real-time thinking and output mechanism
- `distillation.py`: Knowledge distillation process
- `inference.py`: Inference engine
- `main.py`: Command line interface

## Technical Details

### Multi-Process Brain Model

The system has multiple parallel processing modules, each with different roles:

- Left brain module: Deep analysis and logical thinking
- Right brain module: Intuitive responses and creativity
- Frontal lobe module: Executive function and final decision making
- Hippocampus module: Memory formation and retrieval
- Emotional module: Tracking and expressing emotional states

### Thought Process

1. Intuitive first impression (right brain)
2. Exploratory thinking (retrieving related information from memory)
3. Analytical thinking (left brain)
4. Insight generation (occurs probabilistically)
5. Conclusion formation (frontal lobe)

Each stage progresses asynchronously, and by streaming the intermediate progress of thought, it achieves the effect of "talking while thinking" like humans.

### Optimization of Distillation Process

To optimize knowledge distillation from ELYZA models, we've implemented the following techniques:

- Efficient training through gradient accumulation
- Efficient use of large models through 4-bit quantization
- Dynamic allocation of CPU/GPU memory for resource optimization
- Learning with longer sequence lengths for complete knowledge inheritance
- Optimization of temperature parameters to ensure diversity
