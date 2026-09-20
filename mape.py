"""
MAPE computation for the energy model against empirical datasets.

Computes MAPE for:
- Prefill energy using validation/prefill.json
- Decode energy using:
  - validation/decode.json for short outputs (O < threshold), with decode-only = total - prefill
  - validation/decode_parameters.json for long outputs (O >= threshold), decode-only directly
- Total energy combining prefill + decode in both regimes

Usage (as a module):
    from mape import report_mape
    report_mape()

CLI:
    python -m mape
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple
from pathlib import Path

import energy_model
from loaders.analytic import get_config


def _safe_mape(y_true: List[float], y_pred: List[float], eps: float = 1e-2) -> float:
    if not y_true:
        return float('nan')
    acc = 0.0
    for t, p in zip(y_true, y_pred):
        denom = abs(t) if abs(t) > eps else eps
        acc += abs(p - t) / denom
    return (acc / max(1, len(y_true))) * 100.0


def _load_decode_params(cfg) -> Dict[str, List[dict]]:
    path = cfg.get_exponential_params_path()
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "_metadata"}


def report_mape() -> Dict[str, dict]:
    cfg = get_config()
    models = cfg.get_supported_models()
    exp_threshold = cfg.get_exponential_threshold()

    # Prefill and decode validations used by energy_model
    prefill_lookup_available = hasattr(energy_model, "_LOOKUP") and hasattr(energy_model, "_MODEL_KEY_MAP")

    decode_lookup_path = cfg.get_decode_validation_path()
    decode_validation = {}
    if decode_lookup_path.exists():
        with decode_lookup_path.open() as f:
            decode_validation = json.load(f)

    decode_params_all = _load_decode_params(cfg)

    summary: Dict[str, dict] = {}

    for model in models:
        # Buckets
        y_true_prefill: List[float] = []
        y_pred_prefill: List[float] = []
        y_true_dec_short: List[float] = []
        y_pred_dec_short: List[float] = []
        y_true_dec_long: List[float] = []
        y_pred_dec_long: List[float] = []
        y_true_tot_short: List[float] = []
        y_pred_tot_short: List[float] = []
        y_true_tot_long: List[float] = []
        y_pred_tot_long: List[float] = []

        # Prefill from prefill.json
        model_key = energy_model._MODEL_KEY_MAP.get(model, model) if prefill_lookup_available else model
        if prefill_lookup_available and model_key in energy_model._LOOKUP:
            tokens, values = energy_model._LOOKUP[model_key]
            for I, ept in zip(tokens, values):
                true_e = float(ept) * int(I)
                pre_pred, _, _ = energy_model.calculate_energy_from_power_time(model, int(I), 0)
                y_true_prefill.append(true_e)
                y_pred_prefill.append(pre_pred)

        # Decode short (robustly interpret JSON as total or decode-only)
        if model_key in decode_validation:
            d = decode_validation[model_key]
            # Handle new structure: data organized by input length
            for input_len_str, input_data in d.items():
                I = int(input_len_str)
                output_tokens = input_data.get('output_tokens', [])
                total_energies = input_data.get('total_energy_j', [])
                
                for O, T in zip(output_tokens, total_energies):
                    O = int(O); T = float(T)
                    if O <= 0 or O >= exp_threshold:
                        continue
                    pre_emp = energy_model._lookup_prefill_energy(model, I)
                    if pre_emp is None:
                        pre_emp = energy_model.prefill_energy_formula(model, I)
                    # If the JSON stores decode-only energy already, subtracting prefill produces near-zero.
                    # Heuristic: if (T - pre_emp) <= 0 or is <10% of T (when T is non-trivial), treat T as decode-only.
                    dec_candidate = T - pre_emp
                    if dec_candidate <= 0.0 or (T > 1.0 and dec_candidate < 0.1 * T):
                        dec_true = T
                        tot_true = pre_emp + T
                    else:
                        dec_true = dec_candidate
                        tot_true = T
                    pre_pred, dec_pred, tot_pred = energy_model.calculate_energy_from_power_time(model, I, O)
                    y_true_dec_short.append(dec_true)
                    y_pred_dec_short.append(dec_pred)
                    y_true_tot_short.append(tot_true)
                    y_pred_tot_short.append(tot_pred)

        # Decode long (decode_parameters.json)
        exp_model_key = cfg.get_decode_validation_object(model)
        if exp_model_key in decode_params_all:
            for cfg_item in decode_params_all[exp_model_key]:
                I = int(cfg_item.get('input_tokens', 0))
                O = int(cfg_item.get('output_tokens', 0))
                if O < exp_threshold:
                    continue
                dec_true = cfg_item.get('decode_energy')
                if dec_true is None:
                    continue
                dec_true = float(dec_true)
                pre_emp = energy_model._lookup_prefill_energy(model, I)
                if pre_emp is None:
                    pre_emp = energy_model.prefill_energy_formula(model, I)
                tot_true = pre_emp + dec_true
                pre_pred, dec_pred, tot_pred = energy_model.calculate_energy_from_power_time(model, I, O)
                y_true_dec_long.append(dec_true)
                y_pred_dec_long.append(dec_pred)
                y_true_tot_long.append(tot_true)
                y_pred_tot_long.append(tot_pred)

        # Compute MAPEs
        prefill_mape = _safe_mape(y_true_prefill, y_pred_prefill) if y_true_prefill else float('nan')
        dec_short_mape = _safe_mape(y_true_dec_short, y_pred_dec_short) if y_true_dec_short else float('nan')
        dec_long_mape = _safe_mape(y_true_dec_long, y_pred_dec_long) if y_true_dec_long else float('nan')
        tot_short_mape = _safe_mape(y_true_tot_short, y_pred_tot_short) if y_true_tot_short else float('nan')
        tot_long_mape = _safe_mape(y_true_tot_long, y_pred_tot_long) if y_true_tot_long else float('nan')

        # Combined
        dec_all_mape = _safe_mape(y_true_dec_short + y_true_dec_long, y_pred_dec_short + y_pred_dec_long) \
                       if (y_true_dec_short or y_true_dec_long) else float('nan')
        tot_all_mape = _safe_mape(y_true_tot_short + y_true_tot_long, y_pred_tot_short + y_pred_tot_long) \
                       if (y_true_tot_short or y_true_tot_long) else float('nan')

        summary[model] = {
            'prefill_mape': prefill_mape,
            'decode_mape_short': dec_short_mape,
            'decode_mape_long': dec_long_mape,
            'decode_mape_all': dec_all_mape,
            'total_mape_short': tot_short_mape,
            'total_mape_long': tot_long_mape,
            'total_mape_all': tot_all_mape,
            'counts': {
                'prefill': len(y_true_prefill),
                'dec_short': len(y_true_dec_short),
                'dec_long': len(y_true_dec_long),
                'tot_short': len(y_true_tot_short),
                'tot_long': len(y_true_tot_long),
            }
        }

    # Print report
    print("\n📊 MAPE vs empirical datasets")
    print(f"(threshold for long decode: O>={exp_threshold})\n")
    print(f"{'Model':<20} {'Prefill':>8} {'Dec<65':>8} {'Dec>=65':>8} {'Dec-All':>8} {'Tot<65':>8} {'Tot>=65':>8} {'Tot-All':>8}")
    print("-" * 90)
    for model, m in summary.items():
        def _fmt(x: float) -> str:
            return f"{x:>.1f}%" if x == x else "  n/a "
        print(
            f"{model:<20} "
            f"{_fmt(m['prefill_mape']):>8} "
            f"{_fmt(m['decode_mape_short']):>8} "
            f"{_fmt(m['decode_mape_long']):>8} "
            f"{_fmt(m['decode_mape_all']):>8} "
            f"{_fmt(m['total_mape_short']):>8} "
            f"{_fmt(m['total_mape_long']):>8} "
            f"{_fmt(m['total_mape_all']):>8}"
        )
    print()

    return summary


if __name__ == "__main__":
    report_mape()
