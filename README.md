# PPO Self-Driving Mini Car

A 2D top-down self-driving car simulation trained end-to-end with
**Proximal Policy Optimization (PPO)** from Stable-Baselines3.
Runs entirely on CPU — no GPU required.

<video width="700" controls autoplay loop muted>
  <source src="./RL.mkv" type="video/mkv">
  Your browser does not support the video tag.
</video>

```
self_driving_car/
├── car.py        — Car physics + raycasting sensors
├── track.py      — Track geometry, checkpoints, rendering
├── env.py        — Custom Gymnasium environment
├── train.py      — PPO training script
├── test.py       — Run a trained model with rendering
├── utils.py      — HUD, reward logger, Matplotlib plots
├── models/       — Saved model checkpoints
├── logs/         — CSV reward logs + TensorBoard logs
├── plots/        — Reward-curve PNG files
└── requirements.txt
```

---

## 1. Installation

```bash
# create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# CPU-only PyTorch (lighter download, no CUDA drivers needed):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 2. Training

```bash
# default: 200 000 timesteps
python train.py

# longer run
python train.py --timesteps 500000

# watch training (much slower — rendering adds overhead)
python train.py --timesteps 200000 --render

# continue from an existing model or checkpoint
python train.py --resume models/checkpoints/ppo_car_50000_steps.zip

# resume from the latest saved model or checkpoint
python train.py --resume-latest

# randomise spawn position each episode (harder, more robust)
python train.py --randomise-spawn
```

Progress is printed every 10 episodes. Reward plots are saved to
`plots/reward_curve.png` every 20 000 timesteps.

---

## 3. Testing

```bash
# watch 5 episodes with default model
python test.py

# slow-motion (15 fps)
python test.py --slow-mo --episodes 3

# deterministic actions (no sampling)
python test.py --deterministic --episodes 5
```

Press **ESC** or close the window to exit.

---

## 4. Architecture

### Environment (`env.py`)
| Item | Detail |
|---|---|
| Observation | 5 floats: front/left/right sensors, speed, angle-to-centre |
| Action | 1 continuous float: steering ∈ [−1, 1] |
| Reward | +1 near centre, +0.2 forward, +5 checkpoint, −10 crash, −0.01/step |
| Episode end | Crash or 2 000 steps |

### Car (`car.py`)
Simple Euler-integration physics:
```
speed  += throttle × 0.15
speed   = min(speed, 4.0)
speed  *= 0.92   (friction)
angle  += steering × turn_rate × (speed / max_speed)
x      += speed × cos(angle)
y      += speed × sin(angle)
```
5 raycasting sensors at angles: −90°, −45°, 0°, +45°, +90°.

### Track (`track.py`)
A rounded-rectangle loop built from arc segments.
8 evenly-spaced checkpoints drive the progress reward.
Off-track detection uses point-in-polygon tests against outer/inner boundaries.

### PPO Hyperparameters
| Parameter | Value |
|---|---|
| `n_steps` | 2 048 |
| `batch_size` | 64 |
| `learning_rate` | 3e-4 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.01 |
| `device` | cpu |

---

## 5. Tips

* **Training time**: ~10–20 min for 200 k steps on a modern CPU.
  The car usually starts staying on track by ~50 k steps.
* **Reward tuning**: if the car learns to stop (maximises time penalty
  avoidance by not moving), increase `W_FORWARD` in `env.py`.
* **TensorBoard**: `tensorboard --logdir logs/` to visualise training curves.
* **Multiple tracks**: extend `track.py` with a second `Track` subclass and
  pass it to `CarEnv(track=...)`.

---

## 6. Controls (test.py)
| Key | Action |
|---|---|
| ESC | Quit |
| Window ✕ | Quit |
