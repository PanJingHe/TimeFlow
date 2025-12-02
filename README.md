# TimeFlow


## Diffusion Experiments for Time Series

This directory contains the diffusion/flow-based time series experiments used in the project.
The notes below only describe how to set up the environment and run training / sampling.

---

### Environment setup

- Python 3.8+
- One CUDA‑capable GPU is recommended.

Create a new environment and install dependencies:

```bash
conda create -n timeflow python=3.8
conda activate timeflow
pip install -r requirements.txt
```

---

### Data preparation

Place all datasets under:

```text
Data/dataset/
```

The exact file names and splits are controlled by the YAML config files in `Config/`.

---

### Training

All experiments are launched with `main.py`.
The most important arguments are:

- `--name`: experiment name (used to create a subdirectory under `OUTPUT/`).
- `--config_file`: path to a YAML config in `Config/`.
- `--train`: enable training mode.

```bash
python main.py \
  --name {dataset} \
  --config_file Config/{dataset}.yaml \
  --train
```


During training, checkpoints and logs are saved to:

```text
OUTPUT/{name}/
```

---

### Sampling

After training, use `main.py` with `--sample` to generate sequences from a trained checkpoint.

**regular sampling**

```bash
python main.py \
  --name {dataset} \
  --config_file Config/dataset.yaml \
  --sample 0 \
  --milestone {checkpoint_number}
```

**irregular sampling **

```bash
python main.py \
  --name {dataset} \
  --config_file Config/energy_impute.yaml \
  --gpu 0 \
  --sample 1 \
  --mode infill \
  --missing_ratio {missing ratio} \
  --milestone {checkpoint_number}

Generated samples are saved as `.npy` files under `OUTPUT/{name}/` and can be directly used for evaluation and visualization.

