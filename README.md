# ai-benchmark-suite

## run 
preperations
```bash
poetry install  # install dependencies
poetry shell  # activate virtual environment
```

first you need to generate your device information file to gather information about your device
```bash
python -m src.ai_benchmark_suite.device_information
```

then you can run the benchmark suite
```bash
python -m src.ai_benchmark_suite
```

## results

### MNIST
![MNIST with cpu](results/mnist_cpu_average_training_time_per_epoch.png)
![MNIST with cuda](results/mnist_cuda_average_training_time_per_epoch.png)
![MNIST with mps](results/mnist_mps_average_training_time_per_epoch.png)

### QWEN2 
#### 1.5B
![QWEN2 with cpu](results/qwen2-1_5B_cpu_num_generated_tokens_per_second.png)
![QWEN2 with cuda](results/qwen2-1_5B_cuda_num_generated_tokens_per_second.png)
![QWEN2 with mps](results/qwen2-1_5B_mps_num_generated_tokens_per_second.png)

#### 7B
![QWEN2 with cpu](results/qwen2-7B_cpu_num_generated_tokens_per_second.png)
![QWEN2 with cuda](results/qwen2-7B_cuda_num_generated_tokens_per_second.png)
![QWEN2 with mps](results/qwen2-7B_mps_num_generated_tokens_per_second.png)