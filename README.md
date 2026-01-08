# SocialJax for uv
This project is a fork of [SocialJax](https://github.com/cooperativex/SocialJax)
While the original repository used **poetry** for Python dependency and package management, this implementation will use **uv** instead.

## Setting
Creating a Python 3.10 Environment
```
uv python install 3.10
uv venv --python 3.10
```
Dependency resolution and installation
```
uv lock
uv sync --no-install-project --group dev
```
> --no-install-project
→ Skips editable installation since this project will be sourced via PYTHONPATH
> --group dev
→ Installs all necessary dependencies for research/learning code

import test
```
PYTHONPATH=. uv run python -c "import socialjax; print(socialjax.__file__)"
```
If you see the path to socialjax/__init__.py in the project directory when running this, it confirms that the sources are being correctly referenced from the uv environment.
The CUDA-related warnings that appear only indicate CPU fallback and don't affect operation.

run test
```
PYTHONPATH=. uv run python algorithms/IPPO/ippo_cnn_coins.py;
```

If you want to use GPU, please install JAX's GPU-enabled version.
gpu test
```
uv run python -c "import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

## run
### config
Hyperparameters are managed by **hydra-core**. Please refer to the directories under algorithms/*/config/.

### result
We use **wandb** for tracking learning progress.
To view learning logs/gifs/pkl files, please check:
- /outputs
- /wandb
- /checkpoints
- /evaluation



---
This software includes modified code from SocialJax.
Changes were made by romanohu.
