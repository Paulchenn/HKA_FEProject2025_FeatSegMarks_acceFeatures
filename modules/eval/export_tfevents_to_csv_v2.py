#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export TensorBoard .tfevents scalars to CSV (robust v2).
Usage examples:
  python3 export_tfevents_to_csv_v2.py /path/to/events.out.tfevents.* --out ./tb_export
  python3 export_tfevents_to_csv_v2.py /path/to/logdir --out ./tb_export

Changes vs v1:
- Always cast paths to str for TensorBoard's EventAccumulator (fixes PosixPath .endswith error).
- If a directory or glob is provided, auto-pick the newest .tfevents file.
- tbparse fallback tolerates different column names (e.g., timestamp vs wall_time).
- Better error messages and a --list-only mode to just show available scalar tags.
Requires:
  pip install tensorboard
  # optional fallback:
  pip install tbparse
"""
import argparse
import os
from pathlib import Path
import sys
import csv
import math
import json
from datetime import datetime, timezone
import pandas as pd
from typing import Dict, Optional

def resolve_event_file(path_like: str) -> Optional[Path]:
    p = Path(path_like)
    if p.is_file():
        return p
    # Treat as directory or glob
    candidates = []
    if p.is_dir():
        candidates = sorted(p.rglob("events.out.tfevents.*"), key=lambda x: x.stat().st_mtime)
    else:
        # glob pattern
        candidates = sorted(Path().glob(path_like), key=lambda x: x.stat().st_mtime)
    return candidates[-1] if candidates else None

def try_tensorboard(event_file: Path, list_only=False):
    try:
        from tensorboard.backend.event_processing import event_accumulator as ea
    except Exception as e:
        return None, f"tensorboard not available: {e!r}"
    try:
        acc = ea.EventAccumulator(str(event_file), size_guidance={'scalars': 0})
        acc.Reload()
        tags = acc.Tags().get('scalars', [])
        if list_only:
            return {"__TAGS_ONLY__": tags}, None
        scalars = {}
        for tag in tags:
            events = acc.Scalars(tag)
            # make DataFrame per tag
            scalars[tag] = pd.DataFrame({
                "step": [e.step for e in events],
                "wall_time": [e.wall_time for e in events],
                "value": [e.value for e in events],
            })
        return scalars, None
    except Exception as e:
        return None, f"tensorboard failed to read: {e!r}"

def try_tbparse(event_file: Path, list_only=False):
    try:
        from tbparse import SummaryReader
    except Exception as e:
        return None, f"tbparse not available: {e!r}"
    try:
        # tbparse reads a directory, so use parent dir
        reader = SummaryReader(str(event_file.parent), pivot=False)
        df = reader.scalars  # columns: (varies by version) e.g. wall_time/timestamp, step, tag, value, run
        if df is None or df.empty:
            if list_only:
                return {"__TAGS_ONLY__": []}, None
            return {}, None
        # Normalize column names
        cols_lower = {c.lower(): c for c in df.columns}
        # Prefer 'wall_time', but accept 'timestamp' or 'time'
        if 'wall_time' not in df.columns:
            if 'timestamp' in cols_lower:
                df.rename(columns={cols_lower['timestamp']: 'wall_time'}, inplace=True)
            elif 'time' in cols_lower:
                df.rename(columns={cols_lower['time']: 'wall_time'}, inplace=True)
            else:
                # No wall_time-like column; synthesize NaN
                df['wall_time'] = pd.NA
        if 'step' not in df.columns:
            # tbparse should have step; otherwise try 'global_step' or 'iter'
            for alt in ['global_step', 'iter', 'iteration']:
                if alt in df.columns:
                    df.rename(columns={alt: 'step'}, inplace=True)
                    break
            if 'step' not in df.columns:
                df['step'] = range(len(df))
        if 'value' not in df.columns:
            # Some versions may use 'scalar' or 'data'
            for alt in ['scalar', 'data']:
                if alt in df.columns:
                    df.rename(columns={alt: 'value'}, inplace=True)
                    break
        # List-only mode: just return tags
        if list_only:
            tags = sorted(df['tag'].unique().tolist()) if 'tag' in df.columns else []
            return {"__TAGS_ONLY__": tags}, None

        scalars = {}
        if 'tag' not in df.columns:
            # No tag column; treat as single unnamed series
            tag = "scalar"
            sub = df[["step", "wall_time", "value"]].sort_values("step").reset_index(drop=True)
            scalars[tag] = sub
        else:
            for tag, group in df.groupby("tag"):
                sub = group[["step", "wall_time", "value"]].sort_values("step").reset_index(drop=True)
                scalars[tag] = sub
        return scalars, None
    except Exception as e:
        return None, f"tbparse failed to read: {e!r}"

def aggregate_wide(scalars_dict: Dict[str, pd.DataFrame]):
    # build wide DataFrame with 'step' as index
    if not scalars_dict:
        return pd.DataFrame(columns=['step']).set_index('step')
    all_steps = sorted(set().union(*[set(df["step"].tolist()) for df in scalars_dict.values()]))
    wide = pd.DataFrame({"step": all_steps}).set_index("step")
    for tag, df in scalars_dict.items():
        s = df.set_index("step")["value"]
        wide[tag] = s
    return wide

def summarize_stats(scalars_dict: Dict[str, pd.DataFrame]):
    rows = []
    for tag, df in scalars_dict.items():
        if df.empty:
            continue
        steps = df["step"].to_list()
        values = df["value"].to_list()
        rows.append({
            "tag": tag,
            "count": len(values),
            "first_step": steps[0],
            "last_step": steps[-1],
            "min": float(min(values)),
            "max": float(max(values)),
            "last": float(values[-1]),
        })
    return pd.DataFrame(rows).sort_values("tag")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to .tfevents file, directory, or glob pattern")
    ap.add_argument("--out", "-o", default="./tfevents_export", help="Output directory (will be created)")
    ap.add_argument("--list-only", action="store_true", help="Only list available scalar tags (no CSV export)")
    args = ap.parse_args()

    resolved = resolve_event_file(args.path)
    if not resolved:
        print(f"[ERROR] Could not find any .tfevents file from: {args.path}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_tag").mkdir(exist_ok=True)

    # Try tensorboard first
    scalars, err = try_tensorboard(resolved, list_only=args.list_only)
    used_backend = "tensorboard"
    if scalars is None:
        scalars, err2 = try_tbparse(resolved, list_only=args.list_only)
        used_backend = "tbparse"
        if scalars is None:
            print("[ERROR] Could not read the event file with either 'tensorboard' or 'tbparse'.", file=sys.stderr)
            print(" tensorboard error:", err, file=sys.stderr)
            print(" tbparse error:", err2, file=sys.stderr)
            sys.exit(2)

    # List-only mode
    if isinstance(scalars, dict) and "__TAGS_ONLY__" in scalars:
        tags = scalars["__TAGS_ONLY__"]
        print(f"[OK] Found {len(tags)} scalar tag(s) using backend={used_backend}:")
        for t in tags:
            print(" -", t)
        return

    # Save per-tag CSVs
    for tag, df in scalars.items():
        safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tag)
        df.to_csv(out_dir / "per_tag" / f"{safe_tag}.csv", index=False)

    # Wide CSV and summary
    wide = aggregate_wide(scalars)
    wide.to_csv(out_dir / "all_scalars_wide.csv", index=True)

    summary = summarize_stats(scalars)
    summary.to_csv(out_dir / "summary_stats.csv", index=False)

    # Write a small manifest
    manifest = {
        "input_arg": args.path,
        "resolved_event_file": str(resolved),
        "backend": used_backend,
        "num_tags": len(scalars),
        "tags": sorted(list(scalars.keys())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Exported {len(scalars)} scalar tags using backend={used_backend}")
    print(f" - {out_dir/'all_scalars_wide.csv'}")
    print(f" - {out_dir/'summary_stats.csv'}")
    print(f" - per-tag CSVs in {out_dir/'per_tag'}")

if __name__ == "__main__":
    main()
