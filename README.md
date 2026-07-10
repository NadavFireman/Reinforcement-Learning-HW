# Reinforcement Learning - Random Walk Prediction

**Home Assignment (M.Sc. Data Science, HIT). Estimating the state-value function of the classic Random Walk environment with five methods implemented from scratch — First-visit Monte Carlo, TD(0), linear value-function approximation (one-hot and polynomial features) and a small PyTorch value network — all compared against the known closed-form solution.**

## Overview
The Random Walk is the classic testbed for prediction: a 7-state chain with terminal rewards of -1 and +1, an equiprobable left/right policy and γ=1, for which the true state values are known in closed form (linearly spaced between -1 and +1). That makes it ideal for comparing prediction algorithms — every estimate is measured against exact ground truth. The assignment covers three parts: Markov processes and the Bellman equation (theory and manually derived values, verified in simulation), model-free prediction (Monte Carlo vs TD(0)), and value-function approximation — plus a neural-network bonus.

## Key Features
- **Five-Method Comparison** (RMSE on non-terminal states, 1,000 episodes):
  - **First-visit Monte Carlo:** **0.037** — the winner in this run; unbiased averaging pays off in a small environment.
  - **Linear FA with one-hot features:** 0.087.
  - **Tabular TD(0):** 0.134.
  - **Linear FA with polynomial features:** 0.163 — global weight coupling adds variance.
  - **Neural network (PyTorch, bonus):** ~0.23 — in a simple linear environment, extra capacity doesn't pay.
- **From-Scratch Implementations:** Episode generation, Monte Carlo returns, online TD(0) updates and the semi-gradient TD rule for linear approximation — NumPy only, no RL libraries.
- **Bias–Variance Analysis:** Learning curves of the center-state value across 1,000 episodes, contrasting MC's stable convergence with TD's bootstrapping oscillations.
- **Feature-Representation Study:** One-hot (tabular-equivalent, no generalization) vs polynomial (continuous generalization, degree tradeoff), with a discussion of when function approximation becomes necessary and its stability risks ("the deadly triad").
- **Error Analysis:** Per-state absolute-error comparison across all methods, including the effect of terminal-state initialization and sampling noise.

## Tech Stack
- **Language:** Python
- **Core:** NumPy (all algorithms), PyTorch (bonus network)
- **Visualization:** Matplotlib
- **Environment:** Google Colab

## Repository Structure
- `RL_HW_RandomWalk_Notebook.ipynb`: Full solution notebook — theory answers, all five implementations, comparison graphs and analysis.
- `random_walk_env.py`: The Random Walk environment provided with the assignment (configurable chain, rewards and policies), closed-form true values and feature helpers.
- `plotting_utils.py`: Value-comparison and learning-progress plotting helpers.
- `RL_HW_RandomWalk_Instructions.md`: Assignment instructions (Markdown version).
- `Assignment_Instructions_1.pdf`: Original course assignment PDF.
