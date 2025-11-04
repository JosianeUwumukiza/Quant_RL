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
- Speed benchmark: **≈ 171 k steps/s** for 1 M-step runs; micro-loop **≈ 1.36 M steps/s**

---

## Results & Figures

<p align="center">
  <img src="py/report/benchmark.gif" alt="Benchmark animation: PPO vs baselines with equity curve and drawdown" width="80%">
</p>

### Performance Overview
<table>
  <tr>
    <td><img src="py/report/equity.png" alt="Equity curve" width="100%"><br><sub><b>Equity:</b> Cumulative portfolio value</sub></td>
    <td><img src="py/report/drawdown.png" alt="Drawdown curve" width="100%"><br><sub><b>Drawdown:</b> Peak-to-trough losses</sub></td>
  </tr>
  <tr>
    <td><img src="py/report/rolling_sharpe.png" alt="Rolling Sharpe" width="100%"><br><sub><b>Rolling Sharpe:</b> Stability of risk-adjusted returns</sub></td>
    <td><img src="py/report/time_fps.png" alt="Throughput (steps/sec)" width="100%"><br><sub><b>Speed:</b> Simulator throughput (steps/s)</sub></td>
  </tr>
</table>

### Data Sanity & Market Context
<table>
  <tr>
    <td><img src="py/report/data_returns_hist.png" alt="Returns histogram" width="100%"><br><sub><b>Returns Histogram:</b> Fat tails vs normal</sub></td>
    <td><img src="py/report/regime.png" alt="Market regime chart" width="100%"><br><sub><b>Regime:</b> Simple bull/bear labeling used by baselines</sub></td>
  </tr>
</table>

### Training Dynamics
<p align="center">
  <img src="py/report/train_episode_return_rolling.png" alt="Rolling episode return during training" width="60%">
</p>

---




