# vLLM Benchmark Project

This repository contains the code, prompt files, Slurm scripts, and plotting utilities used to compare Hugging Face `generate()` against vLLM for large language model serving. The project benchmarks request throughput under controlled batch-size, sequence-length, and concurrency sweeps.

The project was designed to run on UCF's Newton HPC cluster, but it can also be installed in a normal Python environment if the required GPU, CUDA, and Python dependencies are available.

---

## Repository Structure

```text
vllm-benchmark/
├── prompts/
│   ├── prompts_128.jsonl
│   ├── prompts_256.jsonl
│   ├── prompts_512.jsonl
│   ├── prompts_1024.jsonl
│   └── prompts_2048.jsonl
├── python/
│   ├── benchmark_hf_generate.py
│   ├── benchmark_vllm_openai.py
│   ├── plot_sweep_results.py
│   └── sample_hf_outputs_with_metrics.py
├── scripts/
│   ├── setup_env.sh
│   ├── run_hf_batch_sweep.sh
│   ├── run_hf_sequence_sweep.sh
│   ├── run_hf_concurrency_sweep.sh
│   ├── run_vllm_batch_sweep.sh
│   ├── run_vllm_sequence_sweep.sh
│   ├── run_vllm_concurrency_sweep.sh
│   ├── show_results.sh
│   └── run_sample_hf_outputs.sh
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

The Slurm scripts automatically load the token using:

```bash
export HF_TOKEN="$(cat .hf_token)"
```

If you are running manually, you can also export it yourself. From the project root:

```bash
export HF_TOKEN="$(cat .hf_token)"
```

This export only lasts for the current terminal session. If you close the terminal, you need to run it again.

---

## 3. Slurm Setup

The easiest setup on a Slurm cluster is to use the provided setup script:

```bash
sbatch scripts/setup_env.sbatch
```

This script creates a virtual environment and installs the required packages.

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

## 10. Changing the Model

As mentioned before, scripts are written for:

```text
meta-llama/Llama-2-7b-chat-hf
```

To test a different model, edit the `MODEL_NAME` line in the Slurm script.

If using a gated model, make sure your `.hf_token` has access to that model.

---
evaluate how serving-oriented features such as dynamic request scheduling and PagedAttention-based KV-cache management affect throughput as batch size, sequence length, and concurrency increase.
