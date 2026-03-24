# python3 evaluate.py <pairs_file> <embeddings_file> <metrics_out> [options] -> metrics.json; shared pairs only

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm



def load_pairs(pairs_file: Path):
    """
    Required keys:
      - img1_paths
      - img2_paths
      - labels

    Optional keys from improved pairs.py:
      - repeat_ids
      - fold_ids
      - seed
      - dataset_hash
      - num_repeats
      - num_folds
    """
    data = np.load(pairs_file, allow_pickle=True)

    required = ["img1_paths", "img2_paths", "labels"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing '{key}' in pairs file: {pairs_file}")

    img1_paths = np.asarray(data["img1_paths"]).astype(str)
    img2_paths = np.asarray(data["img2_paths"]).astype(str)
    labels = np.asarray(data["labels"]).astype(np.int32)

    if not (len(img1_paths) == len(img2_paths) == len(labels)):
        raise ValueError("Pair file arrays must have equal length.")

    repeat_ids = (
        np.asarray(data["repeat_ids"]).astype(np.int32)
        if "repeat_ids" in data
        else np.zeros(len(labels), dtype=np.int32)
    )
    fold_ids = (
        np.asarray(data["fold_ids"]).astype(np.int32)
        if "fold_ids" in data
        else np.zeros(len(labels), dtype=np.int32)
    )

    if len(repeat_ids) != len(labels):
        raise ValueError("repeat_ids length does not match labels length.")
    if len(fold_ids) != len(labels):
        raise ValueError("fold_ids length does not match labels length.")

    meta = {
        "seed": int(data["seed"]) if "seed" in data else None,
        "dataset_hash": str(data["dataset_hash"]) if "dataset_hash" in data else None,
        "num_repeats": int(data["num_repeats"]) if "num_repeats" in data else int(len(np.unique(repeat_ids))),
        "num_folds": int(data["num_folds"]) if "num_folds" in data else int(len(np.unique(fold_ids))),
    }

    return img1_paths, img2_paths, labels, repeat_ids, fold_ids, meta


def load_embeddings(embeddings_file: Path):
    data = np.load(embeddings_file, allow_pickle=True)

    embeddings = None
    for key in ["embeddings", "embedding", "embs", "x", "features", "feats"]:
        if key in data.files:
            embeddings = np.asarray(data[key], dtype=np.float32)
            break

    if embeddings is None:
        raise KeyError(f"No embeddings key found. Keys: {list(data.files)}")

    image_paths = None
    for key in ["image_paths", "paths", "img_paths", "filenames", "files"]:
        if key in data.files:
            image_paths = np.asarray(data[key]).astype(str)
            break

    if image_paths is None:
        raise KeyError(
            "No image path key found in embeddings file. "
            "Expected one of: image_paths, paths, img_paths, filenames, files. "
            f"Keys: {list(data.files)}"
        )

    if len(embeddings) != len(image_paths):
        raise ValueError(
            f"Embedding count ({len(embeddings)}) does not match image path count ({len(image_paths)})"
        )

    return embeddings, image_paths



def l2_normalize(x, axis=1, eps=1e-12):
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(norms, eps, None)


def cosine_similarity(a, b):
    return np.sum(a * b, axis=1)


def build_path_to_index(image_paths):
    path_to_index = {}
    duplicates = 0

    for i, p in enumerate(image_paths):
        rp = str(Path(p).resolve())
        if rp in path_to_index:
            duplicates += 1
        path_to_index[rp] = i

    if duplicates > 0:
        print(f"[WARN] Duplicate resolved image paths found in embeddings: {duplicates}")

    return path_to_index


def map_pairs_to_indices(img1_paths, img2_paths, path_to_index, strict_missing=True):
    idx1 = []
    idx2 = []
    valid_mask = []
    missing_examples = []

    for p1, p2 in zip(img1_paths, img2_paths):
        rp1 = str(Path(p1).resolve())
        rp2 = str(Path(p2).resolve())

        i1 = path_to_index.get(rp1)
        i2 = path_to_index.get(rp2)

        if i1 is None or i2 is None:
            valid_mask.append(False)
            if len(missing_examples) < 5:
                missing_examples.append((rp1, rp2))
            continue

        idx1.append(i1)
        idx2.append(i2)
        valid_mask.append(True)

    idx1 = np.asarray(idx1, dtype=np.int64)
    idx2 = np.asarray(idx2, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    dropped = int((~valid_mask).sum())

    if dropped > 0:
        msg = (
            f"{dropped} pair entries could not be matched to embeddings. "
            f"Examples: {missing_examples}"
        )
        if strict_missing:
            raise KeyError(msg)
        print(f"[WARN] Dropped unmatched pairs: {dropped}")
        print(f"[WARN] Example dropped pairs: {missing_examples}")

    return idx1, idx2, valid_mask



def safe_div(a, b):
    return a / b if b != 0 else 0.0


def confusion_from_threshold(scores: np.ndarray, y_true: np.ndarray, threshold: float):
    pred = scores >= threshold

    tp = int(np.sum((y_true == 1) & pred))
    tn = int(np.sum((y_true == 0) & (~pred)))
    fp = int(np.sum((y_true == 0) & pred))
    fn = int(np.sum((y_true == 1) & (~pred)))

    return tp, tn, fp, fn


def metrics_from_confusion(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = safe_div(tp + tn, total)
    far = safe_div(fp, fp + tn)
    frr = safe_div(fn, fn + tp)
    tar = safe_div(tp, tp + fn)
    tnr = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    recall = tar
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "accuracy": float(acc),
        "far": float(far),
        "frr": float(frr),
        "tar": float(tar),
        "tnr": float(tnr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_curve(scores: np.ndarray, y_true: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int32)

    uniq = np.unique(scores)
    if len(uniq) == 1:
        thresholds = np.array([uniq[0] - 1e-6, uniq[0], uniq[0] + 1e-6], dtype=np.float64)
    else:
        thresholds = np.concatenate((
            [uniq[0] - 1e-6],
            uniq,
            [uniq[-1] + 1e-6],
        ))

    fars = np.empty(len(thresholds), dtype=np.float64)
    frrs = np.empty(len(thresholds), dtype=np.float64)
    tars = np.empty(len(thresholds), dtype=np.float64)
    accs = np.empty(len(thresholds), dtype=np.float64)

    best_idx = 0
    best_acc = -1.0

    for i, th in enumerate(thresholds):
        tp, tn, fp, fn = confusion_from_threshold(scores, y_true, float(th))
        m = metrics_from_confusion(tp, tn, fp, fn)

        fars[i] = m["far"]
        frrs[i] = m["frr"]
        tars[i] = m["tar"]
        accs[i] = m["accuracy"]

        if m["accuracy"] > best_acc:
            best_acc = m["accuracy"]
            best_idx = i

    return thresholds, fars, frrs, tars, accs, best_idx


def compute_auc_from_curve(fars: np.ndarray, tars: np.ndarray):
    order = np.argsort(fars)
    x = fars[order]
    y = tars[order]

    x_unique, idx = np.unique(x, return_index=True)
    y_unique = y[idx]

    if len(x_unique) < 2:
        return 0.0

    dx = np.diff(x_unique)
    auc = np.sum((y_unique[:-1] + y_unique[1:]) * 0.5 * dx)
    return float(auc)


def find_eer(thresholds, fars, frrs):
    gap = np.abs(fars - frrs)
    i = int(np.argmin(gap))
    eer = (fars[i] + frrs[i]) / 2.0
    return {
        "eer": float(eer),
        "eer_threshold": float(thresholds[i]),
        "far_at_eer": float(fars[i]),
        "frr_at_eer": float(frrs[i]),
    }


def tar_at_far(thresholds, fars, tars, target_far: float):
    valid = np.where(fars <= target_far)[0]
    if len(valid) == 0:
        return {
            "target_far": float(target_far),
            "tar": 0.0,
            "threshold": None,
            "actual_far": None,
        }

    best_idx = valid[np.argmax(tars[valid])]
    return {
        "target_far": float(target_far),
        "tar": float(tars[best_idx]),
        "threshold": float(thresholds[best_idx]),
        "actual_far": float(fars[best_idx]),
    }


def evaluate_scores(scores: np.ndarray, y_true: np.ndarray, far_targets=(1e-1, 1e-2, 1e-3)):
    thresholds, fars, frrs, tars, accs, best_idx = compute_curve(scores, y_true)

    best_th = float(thresholds[best_idx])
    tp, tn, fp, fn = confusion_from_threshold(scores, y_true, best_th)
    best_metrics = metrics_from_confusion(tp, tn, fp, fn)

    eer_metrics = find_eer(thresholds, fars, frrs)
    auc = compute_auc_from_curve(fars, tars)
    tar_far_metrics = {
        f"tar@far={target:g}": tar_at_far(thresholds, fars, tars, target)
        for target in far_targets
    }

    return {
        "num_pairs": int(len(y_true)),
        "num_genuine_pairs": int(np.sum(y_true == 1)),
        "num_impostor_pairs": int(np.sum(y_true == 0)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "best_threshold": best_th,
        "best_accuracy": best_metrics["accuracy"],
        "far": best_metrics["far"],
        "frr": best_metrics["frr"],
        "tar": best_metrics["tar"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "tp": best_metrics["tp"],
        "tn": best_metrics["tn"],
        "fp": best_metrics["fp"],
        "fn": best_metrics["fn"],
        "auc": float(auc),
        **eer_metrics,
        "tar_at_far": tar_far_metrics,
    }



def summarize_metric_dicts(rows):
    metric_keys = [
        "accuracy", "far", "frr", "tar", "precision", "recall", "f1",
        "auc", "eer"
    ]

    summary = {}
    for key in metric_keys:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if len(vals) == 0:
            summary[key] = {"mean": None, "std": None}
            continue
        vals = np.asarray(vals, dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    tar_far_keys = set()
    for r in rows:
        if "tar_at_far" in r:
            tar_far_keys.update(r["tar_at_far"].keys())

    tar_far_summary = {}
    for k in sorted(tar_far_keys):
        vals = []
        for r in rows:
            if "tar_at_far" in r and k in r["tar_at_far"]:
                vals.append(r["tar_at_far"][k]["tar"])
        if len(vals) == 0:
            tar_far_summary[k] = {"mean": None, "std": None}
            continue
        vals = np.asarray(vals, dtype=np.float64)
        tar_far_summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }

    summary["tar_at_far"] = tar_far_summary
    return summary


def evaluate_repeat_fold(scores, y_true, repeat_ids, fold_ids, far_targets):
    results = []
    unique_repeats = sorted(np.unique(repeat_ids).tolist())

    for rep in unique_repeats:
        rep_mask = (repeat_ids == rep)
        rep_folds = sorted(np.unique(fold_ids[rep_mask]).tolist())
        fold_results = []

        for fold in rep_folds:
            test_mask = rep_mask & (fold_ids == fold)
            train_mask = rep_mask & (fold_ids != fold)

            if not np.any(test_mask) or not np.any(train_mask):
                continue

            train_scores = scores[train_mask]
            train_labels = y_true[train_mask]
            test_scores = scores[test_mask]
            test_labels = y_true[test_mask]

            thresholds, fars, frrs, tars, accs, best_idx = compute_curve(train_scores, train_labels)
            chosen_th = float(thresholds[best_idx])

            tp, tn, fp, fn = confusion_from_threshold(test_scores, test_labels, chosen_th)
            m = metrics_from_confusion(tp, tn, fp, fn)

            test_curve_th, test_fars, test_frrs, test_tars, _, _ = compute_curve(test_scores, test_labels)
            auc = compute_auc_from_curve(test_fars, test_tars)
            eer = find_eer(test_curve_th, test_fars, test_frrs)
            tar_far_metrics = {
                f"tar@far={target:g}": tar_at_far(test_curve_th, test_fars, test_tars, target)
                for target in far_targets
            }

            fold_results.append({
                "repeat": int(rep),
                "fold": int(fold),
                "train_pairs": int(np.sum(train_mask)),
                "test_pairs": int(np.sum(test_mask)),
                "selected_threshold": chosen_th,
                "accuracy": m["accuracy"],
                "far": m["far"],
                "frr": m["frr"],
                "tar": m["tar"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": m["tp"],
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "auc": float(auc),
                **eer,
                "tar_at_far": tar_far_metrics,
            })

        if fold_results:
            results.append({
                "repeat": int(rep),
                "fold_results": fold_results,
                "summary": summarize_metric_dicts(fold_results),
            })

    return results



def bootstrap_confidence_intervals(
    scores: np.ndarray,
    y_true: np.ndarray,
    seed: int,
    n_bootstrap: int = 1000,
    far_targets=(1e-1, 1e-2, 1e-3),
):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    accs = []
    aucs = []
    eers = []
    fars = []
    frrs = []
    f1s = []
    tar_far_store = {f"tar@far={target:g}": [] for target in far_targets}

    for _ in tqdm(range(n_bootstrap), desc="Bootstrap", leave=False):
        idx = rng.integers(0, n, size=n)
        s = scores[idx]
        y = y_true[idx]

        if len(np.unique(y)) < 2:
            continue

        res = evaluate_scores(s, y, far_targets=far_targets)

        accs.append(res["best_accuracy"])
        aucs.append(res["auc"])
        eers.append(res["eer"])
        fars.append(res["far"])
        frrs.append(res["frr"])
        f1s.append(res["f1"])

        for k in tar_far_store:
            tar_far_store[k].append(res["tar_at_far"][k]["tar"])

    def ci(arr):
        if len(arr) == 0:
            return {"mean": None, "std": None, "ci95_low": None, "ci95_high": None}
        arr = np.asarray(arr, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
        }

    return {
        "n_bootstrap_valid": int(len(accs)),
        "best_accuracy": ci(accs),
        "auc": ci(aucs),
        "eer": ci(eers),
        "far": ci(fars),
        "frr": ci(frrs),
        "f1": ci(f1s),
        "tar_at_far": {k: ci(v) for k, v in tar_far_store.items()},
    }



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate InsightFace verification embeddings with reproducible metrics."
    )
    parser.add_argument("pairs_file", type=str, help="Shared pairs .npz")
    parser.add_argument("embeddings_file", type=str, help="Embeddings .npz")
    parser.add_argument("metrics_out", type=str, help="Output metrics .json")
    parser.add_argument("--model-name", type=str, default="InsightFace")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument(
        "--far-targets",
        type=float,
        nargs="*",
        default=[1e-1, 1e-2, 1e-3],
        help="FAR targets for TAR@FAR"
    )
    parser.add_argument(
        "--allow-missing-pairs",
        action="store_true",
        help="Drop unmatched pairs instead of raising an error"
    )
    args = parser.parse_args()

    pairs_file = Path(args.pairs_file).resolve()
    embeddings_file = Path(args.embeddings_file).resolve()
    metrics_out = Path(args.metrics_out).resolve()

    if not pairs_file.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")

    print(f"[INFO] Pairs file:       {pairs_file}")
    print(f"[INFO] Embeddings file:  {embeddings_file}")
    print(f"[INFO] Metrics out:      {metrics_out}")
    print(f"[INFO] Model name:       {args.model_name}")
    print(f"[INFO] Bootstrap iters:  {args.bootstrap}")
    print(f"[INFO] FAR targets:      {args.far_targets}")
    print(f"[INFO] Allow missing:    {args.allow_missing_pairs}")

    t0 = time.perf_counter()

    img1_paths, img2_paths, y_true, repeat_ids, fold_ids, pair_meta = load_pairs(pairs_file)
    print(f"[INFO] Loaded pairs: {len(y_true)}")

    embeddings, image_paths = load_embeddings(embeddings_file)
    embeddings = l2_normalize(embeddings, axis=1)

    print(f"[INFO] Loaded embeddings:  {embeddings.shape}")
    print(f"[INFO] Loaded image paths: {image_paths.shape}")
    print("[INFO] Applied L2 normalization")

    path_to_index = build_path_to_index(image_paths)
    original_pair_count = len(y_true)

    idx1, idx2, valid_mask = map_pairs_to_indices(
        img1_paths,
        img2_paths,
        path_to_index,
        strict_missing=not args.allow_missing_pairs,
    )

    if args.allow_missing_pairs:
        img1_paths = img1_paths[valid_mask]
        img2_paths = img2_paths[valid_mask]
        y_true = y_true[valid_mask]
        repeat_ids = repeat_ids[valid_mask]
        fold_ids = fold_ids[valid_mask]

        if len(y_true) == 0:
            raise RuntimeError("No valid pairs remain after filtering unmatched embedding paths.")

        print(f"[INFO] Valid pairs after filtering: {len(y_true)}")
        print(f"[INFO] Dropped pairs: {original_pair_count - len(y_true)}")

    num_genuine = int(np.sum(y_true == 1))
    num_impostor = int(np.sum(y_true == 0))

    print(f"[INFO] Genuine pairs:  {num_genuine}")
    print(f"[INFO] Impostor pairs: {num_impostor}")
    print(f"[INFO] Total pairs:    {len(y_true)}")
    print(f"[INFO] Repeats found:  {len(np.unique(repeat_ids))}")
    print(f"[INFO] Folds found:    {len(np.unique(fold_ids))}")

    emb_a = embeddings[idx1]
    emb_b = embeddings[idx2]
    scores = cosine_similarity(emb_a, emb_b)

    print(
        f"[INFO] Score range: min={scores.min():.6f}, "
        f"max={scores.max():.6f}, mean={scores.mean():.6f}, std={scores.std():.6f}"
    )

    pooled = evaluate_scores(scores, y_true, far_targets=args.far_targets)

    cv_results = evaluate_repeat_fold(
        scores=scores,
        y_true=y_true,
        repeat_ids=repeat_ids,
        fold_ids=fold_ids,
        far_targets=args.far_targets,
    )

    all_fold_rows = []
    for rep in cv_results:
        all_fold_rows.extend(rep["fold_results"])

    cv_summary = summarize_metric_dicts(all_fold_rows) if all_fold_rows else None

    bootstrap_seed = pair_meta["seed"] if pair_meta["seed"] is not None else 42
    bootstrap = bootstrap_confidence_intervals(
        scores=scores,
        y_true=y_true,
        seed=bootstrap_seed,
        n_bootstrap=args.bootstrap,
        far_targets=args.far_targets,
    )

    elapsed_sec = time.perf_counter() - t0

    modification_type = ""
    try:
        modification_type = metrics_out.parent.parent.name
    except Exception:
        modification_type = ""

    metrics = {
        "modification_type": modification_type,
        "model": args.model_name,
        "num_embeddings": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "num_pairs": int(len(y_true)),
        "num_genuine_pairs": num_genuine,
        "num_impostor_pairs": num_impostor,
        "pair_file": str(pairs_file),
        "embeddings_file": str(embeddings_file),
        "runtime_seconds": float(elapsed_sec),
        "reproducibility": {
            "pair_seed": pair_meta["seed"],
            "dataset_hash": pair_meta["dataset_hash"],
            "num_repeats": pair_meta["num_repeats"],
            "num_folds": pair_meta["num_folds"],
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "allow_missing_pairs": bool(args.allow_missing_pairs),
        },
        "pooled_metrics": pooled,
        "crossval": {
            "per_repeat": cv_results,
            "summary_over_all_test_folds": cv_summary,
        },
        "bootstrap_ci": bootstrap,
        "limitations": [
            "Thresholds are selected by accuracy on train folds unless otherwise changed.",
            "Bootstrap CIs are pair-level, not identity-level.",
            "AUC and EER are computed from observed score thresholds on this evaluation set.",
            "This script evaluates one embedding model on one pair file; cross-model transfer should be run separately.",
            "If --allow-missing-pairs is used, results depend on the filtered subset rather than the full intended protocol."
        ],
    }

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ VERIFICATION RESULTS ================")
    print(f"Best threshold:     {pooled['best_threshold']:.6f}")
    print(f"Best accuracy:      {pooled['best_accuracy']:.6f}")
    print(f"FAR:                {pooled['far']:.6f}")
    print(f"FRR:                {pooled['frr']:.6f}")
    print(f"TAR:                {pooled['tar']:.6f}")
    print(f"Precision:          {pooled['precision']:.6f}")
    print(f"Recall:             {pooled['recall']:.6f}")
    print(f"F1 score:           {pooled['f1']:.6f}")
    print(f"AUC:                {pooled['auc']:.6f}")
    print(f"TP / TN / FP / FN:  {pooled['tp']} / {pooled['tn']} / {pooled['fp']} / {pooled['fn']}")
    print("------------------------------------------------------")
    print(f"EER:                {pooled['eer']:.6f}")
    print(f"EER threshold:      {pooled['eer_threshold']:.6f}")
    print(f"FAR at EER:         {pooled['far_at_eer']:.6f}")
    print(f"FRR at EER:         {pooled['frr_at_eer']:.6f}")

    for k, v in pooled["tar_at_far"].items():
        tar = v["tar"]
        actual_far = v["actual_far"]
        th = v["threshold"]
        print(f"{k.upper():<19} TAR={tar:.6f}  FAR={actual_far if actual_far is not None else 'None'}  TH={th if th is not None else 'None'}")

    if cv_summary is not None:
        print("------------------------------------------------------")
        print("CV SUMMARY (mean ± std over held-out folds)")
        for k in ["accuracy", "auc", "eer", "far", "frr", "tar", "f1"]:
            if k in cv_summary and cv_summary[k]["mean"] is not None:
                print(f"{k.upper():<10} {cv_summary[k]['mean']:.6f} ± {cv_summary[k]['std']:.6f}")

        if "tar_at_far" in cv_summary:
            for k, v in cv_summary["tar_at_far"].items():
                if v["mean"] is not None:
                    print(f"{k.upper():<19} {v['mean']:.6f} ± {v['std']:.6f}")

    print("------------------------------------------------------")
    print("BOOTSTRAP 95% CI (pooled)")
    for k in ["best_accuracy", "auc", "eer", "far", "frr", "f1"]:
        v = bootstrap[k]
        if v["mean"] is not None:
            print(
                f"{k.upper():<14} mean={v['mean']:.6f}  "
                f"95%CI=[{v['ci95_low']:.6f}, {v['ci95_high']:.6f}]"
            )

    print("======================================================\n")
    print(f"[INFO] Saved metrics to: {metrics_out}")


if __name__ == "__main__":
    main()
