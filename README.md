# Reinforcement Learning Control of a Quadcopter

A reinforcement-learning project that uses **Soft Actor-Critic (SAC)** to control a simulated quadcopter.

The project defines a custom continuous-control environment with translational and rotational dynamics, trains an SAC agent to stabilize the vehicle around a target position, and evaluates the learned controller under varying initial conditions.

The accompanying **Reinforcement Learning Based Control of a Quadcopter.pdf** describes the project and results in more detail.

## What is implemented

- custom quadcopter dynamics and state representation,
- continuous four-motor action space,
- position/orientation initialization across a range of starting conditions,
- running observation normalization,
- SAC actor/critic training,
- replay-buffer based learning,
- model evaluation and trajectory visualization.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Files

- `SAC.py` — environment definition and SAC training implementation
- `SAC_evaluate.py` — evaluation and visualization of a trained controller
- `Reinforcement Learning Based Control of a Quadcopter.pdf` — project report

## Run

Train the controller:

```bash
python SAC.py
```

Evaluate a trained controller:

```bash
python SAC_evaluate.py
```

The scripts may create local model/checkpoint files and plots depending on the training/evaluation path used.

## Status

This is a research/learning implementation intended to explore continuous-control reinforcement learning rather than a flight-certified physical controller.
