# py/plot_training.py
import os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_xy(x, y, title, xlabel, ylabel, outpath):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def load_monitor_csv(monitor_path):
    # SB3 Monitor CSV starts with commented header lines
    steps, ep_rew = [], []
    if not os.path.exists(monitor_path):
        return np.array([]), np.array([])
    with open(monitor_path, "r", newline="") as f:
        reader = csv.reader(f)
        # skip comment lines
        for row in reader:
            if not row or row[0].startswith("#"):  # header
                continue
            # default monitor columns: r,l,t  (reward, length, time)
            # SB3 may extend; try to be robust
            try:
                r = float(row[0])
                l = int(row[1])
                # cumulative steps
                steps.append(steps[-1] + l if steps else l)
                ep_rew.append(r)
            except Exception:
                pass
    return np.array(steps), np.array(ep_rew)

def find_event_files(log_dir):
    # TensorBoard event files usually named events.out.tfevents.*
    return glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))

def load_tb_scalars(log_dir):
    try:
        # TensorBoard is optional
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}

    scalars = {}
    for ev in find_event_files(log_dir):
        try:
            ea = EventAccumulator(ev)
            ea.Reload()
            tags = ea.Tags().get("scalars", [])
            for t in tags:
                events = ea.Scalars(t)
                xs = [e.step for e in events]
                ys = [e.value for e in events]
                if t not in scalars:
                    scalars[t] = {"x": [], "y": []}
                scalars[t]["x"].extend(xs)
                scalars[t]["y"].extend(ys)
        except Exception:
            continue
    return scalars

def main():
    here = os.path.dirname(__file__)
    log_dir = os.path.join(here, "logs")
    out_dir = os.path.join(here, "report")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Episode reward convergence (from Monitor)
    monitor_csv = os.path.join(log_dir, "monitor.csv")
    steps, ep_rew = load_monitor_csv(monitor_csv)
    if steps.size > 0:
        plot_xy(steps, ep_rew, "Episode Return (convergence)", "Steps", "Episode reward",
                os.path.join(out_dir, "train_episode_return.png"))
        # rolling mean to show trend
        if ep_rew.size > 20:
            k = min(200, ep_rew.size // 10)
            roll = np.convolve(ep_rew, np.ones(k)/k, mode="valid")
            roll_x = steps[len(steps) - len(roll):]
            plot_xy(roll_x, roll, "Episode Return (rolling mean)", "Steps", f"Mean over {k} episodes",
                    os.path.join(out_dir, "train_episode_return_rolling.png"))
    else:
        print("[warn] monitor.csv not found; skip episode convergence plot.")

    # 2) PPO losses & stats (from TensorBoard)
    tb = load_tb_scalars(log_dir)
    wanted = {
        "train/policy_gradient_loss": "Policy Gradient Loss",
        "train/value_loss": "Value Function Loss",
        "train/entropy_loss": "Entropy",
        "train/explained_variance": "Explained Variance",
        "rollout/ep_len_mean": "Episode Length (mean)",
        "rollout/ep_rew_mean": "Episode Reward (mean)",
        "train/learning_rate": "Learning Rate",
        "time/fps": "Frames per second",
    }
    any_tb = False
    for tag, title in wanted.items():
        if tag in tb and len(tb[tag]["x"]) > 0:
            any_tb = True
            plot_xy(tb[tag]["x"], tb[tag]["y"], title, "Steps", title,
                    os.path.join(out_dir, f"{tag.replace('/','_')}.png"))
    if not any_tb:
        print("[info] No TensorBoard scalars parsed. Install tensorboard and ensure tensorboard_log is set.")

    print("✅ Training report artifacts saved to:", out_dir)

if __name__ == "__main__":
    main()
