# ai-research-template

A modular template for deep learning research projects using PyTorch Lightning.

## Features

- Modular data loading with custom dataset and datamodule ([data.py](data.py))
- Flexible model configuration ([configs/model_config.yaml](configs/model_config.yaml))
- Training pipeline with logging and early stopping ([train.py](train.py))
- Inference script for model evaluation ([inference.py](inference.py))
- Utility functions for config loading, data splitting, and preprocessing ([utils.py](utils.py))
- Easily configurable via YAML files in [configs/](configs/)

## Project Structure

```
.
├── configs/
│   ├── general_config.yaml
│   ├── model_config.yaml
│   └── train_config.yaml
├── data.py
├── inference.py
├── model/
│   └── base.py
├── train.py
├── train_module.py
├── utils.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Getting Started

### Installation

1. Clone the repository.
2. Install dependencies:
    ```sh
    uv sync
    ```
    check [pyproject.toml](pyproject.toml) for more details about the project

Edit the YAML files in [configs/](configs/) to set up your experiment, model, and training parameters.

### Training

Run:
```sh
python train.py --config_dir ./configs
```

### Inference

After training, run:
```sh
python inference.py --model_name LSTM --exp_name onboarding --version_num 0
```

## Customization

- Add your own models in [model/](model/).
- Modify data processing in [data.py](data.py).
- Adjust training logic in [train_module.py](train_module.py).

## License

MIT License. See

