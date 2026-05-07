# vLLM Benchmark Project

This repository contains the code, prompt files, Slurm scripts, and plotting utilities used to compare Hugging Face `generate()` against vLLM for large language model serving. The project benchmarks request throughput under controlled batch-size, sequence-length, and concurrency sweeps.

The project was designed to run on UCF's Newton HPC cluster, but it can also be installed in a normal Python environment if the required GPU, CUDA, and Python dependencies are available.

---

## Repository Structure

```text
vllm-benchmark/
├── prompts/
│   ├── prompts_128_revised.jsonl
│   ├── prompts_256_revised.jsonl
│   ├── prompts_512_revised.jsonl
│   ├── prompts_1024_revised.jsonl
│   └── prompts_2048_revised.jsonl
├── python/
│   ├── benchmark_hf_generate.py
│   ├── benchmark_vllm_openai.py
│   ├── plot_sweep_results.py
│   └── sample_hf_outputs_with_metrics.py
├── scripts/
│   ├── setup_env.sbatch
│   ├── run_hf_batch_sweep.sbatch
│   ├── run_hf_sequence_sweep.sbatch
│   ├── run_hf_concurrency_sweep.sbatch
│   ├── run_vllm_batch_sweep.sbatch
│   ├── run_vllm_sequence_sweep.sbatch
│   ├── run_vllm_concurrency_sweep.sbatch
│   └── run_sample_hf_outputs.sbatch
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 1. Clone the Repository

Clone the repository with:

```bash
git clone https://github.com/ViniciusVR/vllm-benchmark.git
cd vllm-benchmark
```

On some clusters, Git may fail because the login node limits thread creation. If that happens, use:

```bash
git -c pack.threads=1 clone https://github.com/ViniciusVR/vllm-benchmark.git
cd vllm-benchmark
```

---

## 2. Hugging Face Token Requirement

This project uses Meta's Llama 2 model:

```text
meta-llama/Llama-2-7b-chat-hf
```

This is a gated Hugging Face model. Before running the benchmark, you must:

1. Have a Hugging Face account.
2. Request and receive access to the Llama 2 model on Hugging Face.
3. Create a Hugging Face access token.
4. Save that token in a file called `.hf_token` in the project root.

Create the token file like this:

```bash
nano .hf_token
```

Paste your Hugging Face token into the file, save, and exit.

The file should look like this:

```text
hf_your_token_here
```

Do not commit `.hf_token` to GitHub. It is already included in `.gitignore`.

The Slurm scripts automatically load the token using:

```bash
export HF_TOKEN="$(cat .hf_token)"
```

If you are running manually, you can also export it yourself:

```bash
export HF_TOKEN="$(cat .hf_token)"
```

---

## 3. Slurm Setup

The easiest setup on a Slurm cluster is to use the provided setup script:

```bash
sbatch scripts/setup_env.sbatch
```

This script creates a virtual environment named:

```text
venv/
```

and installs the required packages, including:

```text
torch
transformers==4.46.3
tokenizers<0.21
vllm==0.6.4.post1
pandas
matplotlib
aiohttp
openai
```

The pinned versions are important because newer versions of `transformers` and `tokenizers` may cause compatibility issues with `vllm==0.6.4.post1`.

After submitting the setup job, monitor it with:

```bash
squeue -u $USER
```

You can follow the setup log with:

```bash
tail -f scripts/logs/setup_env_<jobid>.out
```

or check the error log with:

```bash
tail -f scripts/logs/setup_env_<jobid>.err
```

Replace `<jobid>` with the Slurm job ID.

---

## 4. Non-Slurm Setup

If you are not using Slurm, create a virtual environment manually:

```bash
python3 -m venv venv
source venv/bin/activate
```

Upgrade basic packaging tools:

```bash
pip install --upgrade pip setuptools wheel
```

Install PyTorch. For CUDA 12.1 systems, use:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If vLLM or Transformers compatibility issues occur, reinstall the pinned versions manually:

```bash
pip uninstall -y transformers tokenizers
pip install "transformers==4.46.3" "tokenizers<0.21"

pip uninstall -y vllm
pip install "vllm==0.6.4.post1"
```

Then verify the install:

```bash
python - <<'PY'
import torch
import transformers
import tokenizers
import vllm
import pandas
import matplotlib

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("Tokenizers:", tokenizers.__version__)
print("vLLM:", vllm.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
PY
```

---

## 5. Running the Hugging Face Benchmarks

The Hugging Face baseline uses `AutoModelForCausalLM.generate()`.

Run the three sweeps with:

```bash
sbatch scripts/run_hf_batch_sweep.sbatch
sbatch scripts/run_hf_sequence_sweep.sbatch
sbatch scripts/run_hf_concurrency_sweep.sbatch
```

These create:

```text
results/hf_batch_sweep.csv
results/hf_sequence_sweep.csv
results/hf_concurrency_sweep.csv
```

### Hugging Face Sweep Definitions

Batch sweep:

```text
Prompt length: 512 tokens
Output length: 512 tokens
Batch sizes: 1, 2, 4, 8, 16, 32
Concurrency: 1
```

Sequence sweep:

```text
Prompt lengths: 128, 256, 512, 1024, 2048 tokens
Output length: matched to prompt target length
Batch size: 1
Concurrency: 1
```

Concurrency sweep:

```text
Prompt length: 512 tokens
Output length: 512 tokens
Batch size: 1
Concurrency levels: 1, 2, 4, 8, 16, 32
```

For Hugging Face, concurrency is simulated using Python threads because `generate()` is a local inference API, not a serving system with a built-in request scheduler.

---

## 6. Running the vLLM Benchmarks

The vLLM benchmarks launch a local OpenAI-compatible vLLM server inside each Slurm job. The benchmark client sends requests to that server.

Run the three sweeps with:

```bash
sbatch scripts/run_vllm_batch_sweep.sbatch
sbatch scripts/run_vllm_sequence_sweep.sbatch
sbatch scripts/run_vllm_concurrency_sweep.sbatch
```

These create:

```text
results/vllm_batch_sweep.csv
results/vllm_sequence_sweep.csv
results/vllm_concurrency_sweep.csv
```

### vLLM Sweep Definitions

The vLLM sweeps use the same prompt files, output lengths, and workload settings as the Hugging Face sweeps.

For vLLM:

```text
batch_size_requested
```

means the number of prompts sent in one HTTP completion request.

```text
concurrency
```

means the number of HTTP completion requests in flight.

vLLM may dynamically batch and schedule requests internally.

---

## 7. Running a Small Output Sample

To generate qualitative example outputs with throughput information:

```bash
sbatch scripts/run_sample_hf_outputs.sbatch
```

This creates:

```text
results/sample_outputs_512_with_metrics.csv
```

The CSV includes:

```text
prompt
generated_output
prompt_tokens_model
generated_tokens
total_tokens
batch_latency_sec
batch_tokens_per_sec
batch_generated_tokens_per_sec
batch_requests_per_sec
prompt_tokens_per_sec
prompt_generated_tokens_per_sec
prompt_requests_per_sec
```

This file is useful for including example prompts and outputs in the report.

---

## 8. Plotting Figures

After the HF and vLLM sweeps have finished, create comparison figures with:

```bash
python python/plot_sweep_results.py \
    --results-dir results \
    --output-dir figures \
    --metric requests_per_sec
```

This creates:

```text
figures/batch_sweep_requests_per_sec.png
figures/sequence_sweep_requests_per_sec.png
figures/concurrency_sweep_requests_per_sec.png
```

Other available metrics include:

```bash
python python/plot_sweep_results.py --results-dir results --output-dir figures --metric generated_tokens_per_sec
python python/plot_sweep_results.py --results-dir results --output-dir figures --metric tokens_per_sec
python python/plot_sweep_results.py --results-dir results --output-dir figures --metric total_time_sec
```

---

## 9. Output CSV Columns

The main benchmark CSV files include columns such as:

```text
status
backend
sweep_name
model_name
prompt_file
prompt_target_tokens
batch_size_requested
concurrency
max_new_tokens
num_prompts
num_requests
prompt_tokens
generated_tokens
total_tokens
total_time_sec
tokens_per_sec
generated_tokens_per_sec
requests_per_sec
```

The most important metric used in the report is:

```text
requests_per_sec
```

Other useful metrics are:

```text
generated_tokens_per_sec
tokens_per_sec
total_time_sec
```

---

## 10. Changing the Model

The scripts are written for:

```text
meta-llama/Llama-2-7b-chat-hf
```

To test a different model, either edit the `MODEL_NAME` line in the Slurm script or modify the scripts to read the model from an environment variable.

Example command if the scripts support environment-variable override:

```bash
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2" \
MODEL_TAG="mistral_7b" \
sbatch scripts/run_hf_batch_sweep.sbatch
```

If using a gated model, make sure your `.hf_token` has access to that model.

---

## 11. Notes and Troubleshooting

### vLLM Compatibility

This project used:

```text
vllm==0.6.4.post1
transformers==4.46.3
tokenizers<0.21
```

If vLLM fails with tokenizer-related errors, reinstall these pinned versions:

```bash
source venv/bin/activate

pip uninstall -y transformers tokenizers
pip install "transformers==4.46.3" "tokenizers<0.21"

pip uninstall -y vllm
pip install "vllm==0.6.4.post1"
```

### OpenBLAS or Thread Creation Errors

If you see errors like:

```text
OpenBLAS blas_thread_init: pthread_create failed
```

limit CPU threads:

```bash
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

The Slurm scripts already set these variables.

### Git Clone Thread Error on Cluster

If cloning fails with:

```text
fatal: unable to create thread: Resource temporarily unavailable
fatal: fetch-pack: invalid index-pack output
```

use:

```bash
git -c pack.threads=1 clone --depth 1 https://github.com/ViniciusVR/vllm-benchmark.git
```

### Figures Do Not Include One Backend

If a figure only shows HF or only shows vLLM, check that both CSV files exist in `results/` and contain successful rows:

```bash
ls results/
head results/hf_batch_sweep.csv
head results/vllm_batch_sweep.csv
```

The plotting script prints which backend rows were detected for each sweep.

---

## 12. Recommended Full Reproduction Order

On Slurm:

```bash
git -c pack.threads=1 clone --depth 1 https://github.com/ViniciusVR/vllm-benchmark.git
cd vllm-benchmark

nano .hf_token

sbatch scripts/setup_env.sbatch

sbatch scripts/run_hf_batch_sweep.sbatch
sbatch scripts/run_hf_sequence_sweep.sbatch
sbatch scripts/run_hf_concurrency_sweep.sbatch

sbatch scripts/run_vllm_batch_sweep.sbatch
sbatch scripts/run_vllm_sequence_sweep.sbatch
sbatch scripts/run_vllm_concurrency_sweep.sbatch

python python/plot_sweep_results.py \
    --results-dir results \
    --output-dir figures \
    --metric requests_per_sec
```

For a normal non-Slurm environment:

```bash
git clone https://github.com/ViniciusVR/vllm-benchmark.git
cd vllm-benchmark

nano .hf_token

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Then run the Python benchmark scripts manually or adapt the Slurm commands into shell commands for the local environment.

---

## 13. Project Summary

This benchmark compares direct Hugging Face generation against vLLM serving under controlled workload pressure. The goal is to evaluate how serving-oriented features such as dynamic request scheduling and PagedAttention-based KV-cache management affect throughput as batch size, sequence length, and concurrency increase.
