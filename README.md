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
→ Skips editable installation since this project is run from source at the repo root
> --group dev
→ Installs all necessary dependencies for research/learning code

import test
```
uv run python -c "import socialjax; print(socialjax.__file__)"
```
If you see the path to socialjax/__init__.py in the project directory when running this, it confirms that the sources are being correctly referenced from the uv environment.
The CUDA-related warnings that appear only indicate CPU fallback and don't affect operation.

run test
```
uv run algorithms/IPPO/ippo_cnn_coins.py
```

If you want to use GPU, please install JAX's GPU-enabled version.
gpu test
```
uv run python -c "import jax; print('backend:', jax.default_backend()); print('devices:', jax.devices())"
```

## run
if you use **pueue**, set `cwd` to the repo root so imports resolve correctly.
### config
Hyperparameters are managed by **hydra-core**. Please refer to the directories under algorithms/*/config/.
New component-based runs use configs under:
- scripts/config/algorithm/
- scripts/config/env/
- scripts/config/wandb/

### runner
We provide a reusable training entrypoint under scripts/ backed by shared components for IPPO/MAPPO/SVO.
```
uv run scripts/train.py algorithm=ippo env=clean_up
```
Switch independent policy/reward
```
uv run scripts/train.py algorithm=ippo env=clean_up independent_policy=true independent_reward=true
```
Override config values
```
uv run scripts/train.py algorithm.LR=0.0003 env.env_kwargs.num_agents=5
```
Disable actual training (config check only)
```
uv run scripts/train.py dry_run=true
```

Checkpoint output (component runner)
```
uv run scripts/train.py algorithm.CHECKPOINT_DIR=checkpoints/components/ippo algorithm.CHECKPOINT_EVERY=10
```

Evaluation (GIF rendering)
```
uv run scripts/eval.py algorithm=ippo env=clean_up checkpoint_dir=checkpoints/components/ippo
```

### Agent Color
Color Palette List
This order is preserved in self.PLAYER_COLOURS and used for drawing.
| Preview | RGB Value | HEX Code | Color Name / Description |
| :---: | :--- | :--- | :--- |
| <img src="https://singlecolorimage.com/get/CC2828/50x20" > | `(204, 40, 40)` | `#CC2828` | Vivid Red |
| <img src="https://singlecolorimage.com/get/CCB428/50x20" > | `(204, 180, 40)` | `#CCB428` | Golden Yellow |
| <img src="https://singlecolorimage.com/get/57CC28/50x20" > | `(87, 204, 40)` | `#57CC28` | Lime Green |
| <img src="https://singlecolorimage.com/get/28CC86/50x20" > | `(40, 204, 134)` | `#28CC86` | Emerald Green |
| <img src="https://singlecolorimage.com/get/2886CC/50x20" > | `(40, 134, 204)` | `#2886CC` | Sky Blue |
| <img src="https://singlecolorimage.com/get/5728CC/50x20" > | `(87, 40, 204)` | `#5728CC` | Deep Violet |
| <img src="https://singlecolorimage.com/get/CC28B4/50x20" > | `(204, 40, 180)` | `#CC28B4` | Magenta / Rose |


### result
We use **wandb** for tracking learning progress.
To view learning artifacts, please check:
- /runs/YYYY-MM-DD_HH-MM-SS (training runs, checkpoints)
- /wandb
- /runs/YYYY-MM-DD_HH-MM-SS/checkpoints/<algorithm>/evaluation (evaluation GIFs)

Component runner notes:
- It logs env info metrics (e.g. cleanup's cleaned_water) to wandb automatically.
- It saves checkpoints under runs/YYYY-MM-DD_HH-MM-SS/<algorithm> by default.

---
This software includes modified code from SocialJax.
Changes were made by romanohu.
