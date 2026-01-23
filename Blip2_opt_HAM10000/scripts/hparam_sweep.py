# scripts/hparam_sweep.py
import argparse, copy, json, os, random, subprocess, sys, time
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

from pprint import pformat

import yaml
def _pretty_trial_params(trial_idx: int, params: dict) -> str:
    # stable, readable format
    return f"\n[SWEEP] Trial {trial_idx:03d} params:\n{pformat(params, sort_dicts=True)}\n"

def set_by_dotted_key(d, dotted_key, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def get_by_dotted_key(d, dotted_key):
    keys = dotted_key.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def sample_value(spec, rng):
    # spec can be: [a,b,c] OR ["uniform", lo, hi] OR ["loguniform", lo, hi]
    if isinstance(spec, list) and len(spec) >= 3 and isinstance(spec[0], str):
        kind = spec[0].lower()
        lo, hi = float(spec[1]), float(spec[2])
        if kind == "uniform":
            return lo + (hi - lo) * rng.random()
        if kind == "loguniform":
            import math
            loglo, loghi = math.log(lo), math.log(hi)
            return float(math.exp(loglo + (loghi - loglo) * rng.random()))
        raise ValueError(f"Unknown sampler: {spec[0]}")
    # categorical
    if isinstance(spec, list):
        return rng.choice(spec)
    return spec

def apply_constraints(trial_params, constraints, rng):
    # very simple constraint handler matching the YAML structure above
    # If "if" dict matches sampled values, then enforce "then" replacements.
    if not constraints:
        return trial_params
    for rule in constraints:
        cond = rule.get("if", {})
        ok = True
        for k, v in cond.items():
            if trial_params.get(k) != v:
                ok = False
                break
        if ok:
            then = rule.get("then", {})
            for k, v in then.items():
                # v can be list -> pick one
                if isinstance(v, list):
                    trial_params[k] = rng.choice(v)
                else:
                    trial_params[k] = v
    return trial_params

def parse_metric_from_json(run_dir, metric_key):
    # If your code already writes metrics.json, prefer that.
    p = Path(run_dir) / "metrics.json"
    if p.exists():
        try:
            data = json.loads(p.read_text())
            return float(data.get(metric_key))
        except Exception:
            return None
    return None

def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", required=True, help="configs/hparam_search_blip2_opt.yaml")
    ap.add_argument("--project_root", default=".", help="repo root where src/run.py exists")
    ap.add_argument("--outdir", default="hparam_runs", help="where to store trial configs/results")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    search_cfg = yaml.safe_load(Path(args.search).read_text())
    base_cfg_path = project_root / search_cfg["base_config"]
    base_cfg = yaml.safe_load(base_cfg_path.read_text())

    metric = search_cfg.get("metric", "f1_macro")
    maximize = bool(search_cfg.get("maximize", True))
    trials = int(search_cfg.get("trials", 20))
    seed = int(search_cfg.get("seed", 42))

    fast_tune = search_cfg.get("fast_tune", {}) or {}
    fast_enabled = bool(fast_tune.get("enabled", False))
    fast_epochs = int(fast_tune.get("epochs", 6))

    rng = random.Random(seed)

    search_space = search_cfg.get("search_space", {})
    constraints = search_cfg.get("constraints", [])

    best = {"trial": None, "metric": None, "params": None, "config_path": None}

    results_path = outdir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for t in range(trials):
        # sample trial params
        params = {}
        for k, spec in search_space.items():
            params[k] = sample_value(spec, rng)
        params = apply_constraints(params, constraints, rng)

        # build trial config by overriding base
        trial_cfg = copy.deepcopy(base_cfg)

        # Optional fast tuning: reduce epochs (config-only)
        if fast_enabled:
            if "train" not in trial_cfg:
                trial_cfg["train"] = {}
            trial_cfg["train"]["epochs"] = fast_epochs

        # Apply only keys that exist OR always set (safe for YAML-only extension)
        for dotted_key, value in params.items():
            set_by_dotted_key(trial_cfg, dotted_key, value)

        trial_id = f"trial_{t:03d}"
        trial_dir = outdir / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        trial_yaml = trial_dir / "config.yaml"
        trial_yaml.write_text(yaml.safe_dump(trial_cfg, sort_keys=False))

        # choose mode from search config

        mode = search_cfg.get("mode", "train_multiclass")

        # IMPORTANT: keep your core code unchanged, just call it
        cmd = [
            sys.executable, str(project_root / "src" / "run.py"),
            "-c", str(trial_yaml),
            "--mode", str(mode),
        ]

        print(f"\n=== {trial_id} ===")
        print("CMD:", " ".join(cmd))
        
        # ---- add: print + persist params BEFORE training starts ----
        print(_pretty_trial_params(t, params), flush=True)
        
        # save the sampled params so you can always match logs/results later
        (trial_dir / "params.json").write_text(json.dumps(params, indent=2))
        # -----------------------------------------------------------


        # You can pass an env var to separate outputs if your code supports it.
        env = os.environ.copy()
        env["HPARAM_TRIAL_DIR"] = str(trial_dir)

        start = time.time()
        p = subprocess.run(cmd, cwd=str(project_root), env=env)
        dur = time.time() - start

        # Try to read metric from a known location (adapt if your project writes elsewhere)
        mval = parse_metric_from_json(trial_dir, metric)

        record = {
            "trial": trial_id,
            "duration_sec": dur,
            "returncode": p.returncode,
            "metric": mval,
            "metric_name": metric,
            "params": params,
            "config": str(trial_yaml),
        }
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # update best if we got a metric
        if mval is not None:
            if best["metric"] is None:
                better = True
            else:
                better = (mval > best["metric"]) if maximize else (mval < best["metric"])
            if better:
                best.update({"trial": trial_id, "metric": mval, "params": params, "config_path": str(trial_yaml)})
                (outdir / "best_config.yaml").write_text(trial_yaml.read_text())

        print(f"Metric {metric} =", mval)
        print("Best so far =", best["metric"], "from", best["trial"])

    print("\n==== DONE ====")
    print("BEST:", best)
    if best["config_path"]:
        print("Saved best config to:", outdir / "best_config.yaml")

if __name__ == "__main__":
    main()
