# Quant-RL Lab

High-performance reinforcement learning for systematic trading.  
This repo pairs a fast **C++ market simulator/backtester** with **Python RL agents** (Stable-Baselines3/PyTorch) and classic technical baselines for walk-forward evaluation on daily SPY data.

> Research code for education only — **not financial advice**.

---

## Highlights

- **Walk-forward evaluation (WFV)** with year-by-year folds (train → validate → test)  
- **C++ backtester core** (pybind11) for speed; Python for agents & orchestration  
- **Baselines:** moving-average crossover (fast/slow), short-on-bear flag, fee/slippage modeling  
- **RL agents:** PPO (plug in other Stable Baseline 3 algorithms easily)  
- **Reports:** Sharpe, Max Drawdown, equity curves, trade stats, and optional animation  

**Recent results (example run)**  
- PPO on SPY (2005–2024), 3-fold walk-forward: **Sharpe 0.8868 ± 0.1022**, **MaxDD 0.0818**  
- Speed benchmark: **~171k steps/s** for 1e6-step runs; micro-loop **~1.36M steps/s**

---
