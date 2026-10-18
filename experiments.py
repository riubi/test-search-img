import csv
import itertools
import json
import os
import time

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common_points as cp

OUT = os.path.join(cp.OUT_DIR, "experiments")
METHODS = ["SIFT", "ORB", "AKAZE", "BRISK"]


def metrics(res):
    n = len(res["items"])
    return {
        "kp_raw": float(np.mean([it["n_raw"] for it in res["items"]])),
        "kp": float(np.mean([len(it["kp"]) for it in res["items"]])),
        "matches": int(sum(al["matches"] for al in res["align"].values())),
        "ok_pairs": int(sum(st["ok"] for st in res["stats"])),
        "seen_4": int(sum(len(t["points"]) >= n - 2 for t in res["tracks"])),
        "seen_5": int(sum(len(t["points"]) >= n - 1 for t in res["tracks"])),
        "common": len(res["common"]),
    }


def save_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


SWEEPS = {
    "nfeatures": [0, 500, 1000, 2000, 4000],
    "nOctaveLayers": [2, 3, 4, 5, 6],
    "contrastThreshold": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
    "edgeThreshold": [3, 5, 10, 15, 20],
    "sigma": [1.0, 1.6, 2.4, 3.2, 4.0],
    "desc_scale": [1, 2, 3, 4],
    "upright": [False, True],
    "rootsift": [False, True],
    "ratio": [0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    "obj_size": [200, 300, 400, 500],
}


def sift_sweeps():
    rows = []
    for name, values in SWEEPS.items():
        for v in values:
            res = cp.run({name: v}, verbose=False)
            m = metrics(res)
            rows.append({"param": name, "value": v, **m})
            print(name, v, m, flush=True)
    save_csv(rows, f"{OUT}/sift_params.csv")

    fig, axes = plt.subplots(len(SWEEPS), 3, figsize=(13, 2.6 * len(SWEEPS)))
    for r, name in enumerate(SWEEPS):
        sub = [x for x in rows if x["param"] == name]
        xs = [str(x["value"]) for x in sub]
        default = [i for i, x in enumerate(sub) if float(x["value"]) == float(cp.PARAMS[name])]
        ax = axes[r, 0]
        ax.plot(xs, [x["kp_raw"] for x in sub], "o-", label="найдено")
        ax.plot(xs, [x["kp"] for x in sub], "s--", label="после прореживания")
        ax.set_ylabel(name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax = axes[r, 1]
        ax.plot(xs, [x["matches"] for x in sub], "o-", color="tab:green")
        ax = axes[r, 2]
        ax.plot(xs, [x["seen_4"] for x in sub], "o-", color="tab:orange", label=">= 4 из 6")
        ax.plot(xs, [x["seen_5"] for x in sub], "s-", color="tab:purple", label=">= 5 из 6")
        ax.plot(xs, [x["common"] for x in sub], "^-", color="tab:red", label="все 6")
        ax.legend(fontsize=8)
        for ax in axes[r]:
            ax.grid(alpha=0.3)
            if default:
                ax.axvline(default[0], color="gray", ls=":", lw=1)
    axes[0, 0].set_title("ключевых точек на изображение (среднее)")
    axes[0, 1].set_title("соответствий с опорным (сумма по 5 парам)")
    axes[0, 2].set_title("точек, найденных на k изображениях")
    fig.tight_layout()
    fig.savefig(f"{OUT}/sift_params.png", dpi=90)
    plt.close(fig)
    return rows


def compare_methods():
    rows = []
    for name in METHODS:
        t0 = time.time()
        res = cp.run({"method": name}, verbose=False)
        dt = time.time() - t0
        m = metrics(res)
        m["t_det"] = round(res["t_det"], 2)
        m["t_total"] = round(dt, 2)
        m["per_image"] = [it["n_raw"] for it in res["items"]]
        rows.append({"method": name, **m})
        print(name, m, flush=True)
        cp.draw_common(res["items"], res["tracks"], f"{OUT}/partial_{name}.png", res["ref"], 30,
                       f"{name}: точки, найденные хотя бы на 4 из 6 изображений")
    with open(f"{OUT}/methods.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    return rows


def global_baseline(res, ratio=0.8):
    items, norm = res["items"], res["norm"]
    bf = cv2.BFMatcher(norm)
    out = []
    for a, b in itertools.combinations(range(len(items)), 2):
        A, B = items[a], items[b]
        good = [m for m, n in (p for p in bf.knnMatch(A["des"], B["des"], k=2) if len(p) == 2)
                if m.distance < ratio * n.distance]
        inl, uniq, scale = 0, 0, 0.0
        if len(good) >= 3:
            src = np.float32([A["pts"][m.queryIdx] for m in good])
            dst = np.float32([B["pts"][m.trainIdx] for m in good])
            T, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                  ransacReprojThreshold=0.05 * res["params"]["obj_size"])
            if T is not None:
                inliers = [m for m, k in zip(good, mask.ravel()) if k]
                inl = len(inliers)
                uniq = len({m.trainIdx for m in inliers})
                scale = cp.transform_scale(T)
        out.append({"pair": f"{a + 1}-{b + 1}", "ratio": len(good), "inliers": inl, "unique": uniq,
                    "scale": round(scale, 2)})
    return out


def ablation():
    base = cp.run(verbose=False)
    no_mirror = cp.run({"mirror": False}, verbose=False)
    plain = global_baseline(base)
    rows = {
        "default": metrics(base),
        "no_mirror": metrics(no_mirror),
        "no_mirror_pairs": [{"pair": f"{s['pair'][0]}-{s['pair'][1]}", "inliers": s["inliers"], "ok": s["ok"]}
                            for s in no_mirror["stats"]],
        "default_pairs": [{"pair": f"{s['pair'][0]}-{s['pair'][1]}", "ratio": s["ratio"], "mutual": s["mutual"],
                           "inliers": s["inliers"], "final": s["final"], "ok": s["ok"]} for s in base["stats"]],
        "global_baseline": plain,
    }
    with open(f"{OUT}/ablation.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("ablation", rows["default"], rows["no_mirror"], flush=True)
    return base, rows


def subsets(base):
    items, pairs, norm, p = base["items"], base["pairs"], base["norm"], base["params"]
    n = len(items)
    rows = []
    for k in range(2, n + 1):
        counts = [len(cp.common_points(items, pairs, norm, p, list(s))["common"])
                  for s in itertools.combinations(range(n), k)]
        rows.append({"k": k, "mean": float(np.mean(counts)), "min": int(min(counts)), "max": int(max(counts))})
        print("subset", rows[-1], flush=True)
    save_csv(rows, f"{OUT}/subsets.csv")

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ks = [r["k"] for r in rows]
    ax.plot(ks, [r["mean"] for r in rows], "o-", label="среднее по всем наборам")
    ax.fill_between(ks, [r["min"] for r in rows], [r["max"] for r in rows], alpha=0.2, label="мин - макс")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("число изображений в наборе")
    ax.set_ylabel("общих точек")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/subsets.png", dpi=110)
    plt.close(fig)
    return rows


def make_detector(name):
    return cp.make_detector(dict(cp.PARAMS, method=name))


def warp(img, kind, v):
    h, w = img.shape
    H = np.eye(3)
    if kind == "rotation":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), v, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        out = cv2.warpAffine(img, M, (nw, nh), borderValue=int(np.median(img)))
        H[:2] = M
    elif kind == "scale":
        out = cv2.resize(img, None, fx=v, fy=v, interpolation=cv2.INTER_AREA if v < 1 else cv2.INTER_LINEAR)
        H[0, 0] = out.shape[1] / w
        H[1, 1] = out.shape[0] / h
    elif kind == "blur":
        out = cv2.GaussianBlur(img, (0, 0), v) if v > 0 else img.copy()
    elif kind == "noise":
        rng = np.random.default_rng(0)
        out = np.clip(img + rng.normal(0, v, img.shape), 0, 255).astype(np.uint8)
    elif kind == "jpeg":
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, v])
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    elif kind == "mirror":
        out = cv2.flip(img, 1)
        H[0, 0] = -1
        H[0, 2] = w - 1
    else:
        raise ValueError(kind)
    return out, H


def synthetic_one(img, det, norm, kind, v, ratio=0.8, tol=3.0):
    kp1, d1 = det.detectAndCompute(img, None)
    img2, H = warp(img, kind, v)
    kp2, d2 = det.detectAndCompute(img2, None)
    if d1 is None or d2 is None or len(kp2) < 2:
        return {"kp1": len(kp1), "kp2": len(kp2), "rep": 0, "matches": 0, "correct": 0, "precision": 0}
    p1 = cv2.perspectiveTransform(np.float32([k.pt for k in kp1]).reshape(-1, 1, 2), H).reshape(-1, 2)
    p2 = np.float32([k.pt for k in kp2])
    s = np.sqrt(abs(np.linalg.det(H[:2, :2])))
    tol = tol * max(1.0, s)

    h2, w2 = img2.shape
    inside = (p1[:, 0] >= 0) & (p1[:, 0] < w2) & (p1[:, 1] >= 0) & (p1[:, 1] < h2)
    d = np.sqrt(((p1[inside, None] - p2[None]) ** 2).sum(-1))
    rep = float((d.min(1) < tol).mean()) if inside.any() else 0.0

    bf = cv2.BFMatcher(norm)
    good = [m for m, n in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < ratio * n.distance]
    correct = sum(np.linalg.norm(p1[m.queryIdx] - p2[m.trainIdx]) < tol for m in good)
    return {"kp1": len(kp1), "kp2": len(kp2), "rep": rep, "matches": len(good), "correct": int(correct),
            "precision": correct / len(good) if good else 0.0}


SYN = {
    "rotation": [0, 15, 30, 45, 90, 135, 180],
    "scale": [0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0],
    "blur": [0, 1, 2, 3, 4, 6],
    "noise": [0, 5, 10, 20, 30, 45],
    "jpeg": [95, 50, 30, 15, 8, 3],
}
SYN_TITLES = {"rotation": "поворот, градусы", "scale": "масштаб", "blur": "размытие, sigma",
              "noise": "шум, СКО", "jpeg": "качество JPEG"}


def synthetic():
    item = cp.prepare(cp.IMAGES[4], 5, cp.PARAMS)
    img = item["img"]
    rows = []
    for name in METHODS:
        det, norm = make_detector(name)
        for kind, values in SYN.items():
            for v in values:
                r = synthetic_one(img, det, norm, kind, v)
                rows.append({"method": name, "kind": kind, "value": v, **r})
        r = synthetic_one(img, det, norm, "mirror", 0)
        rows.append({"method": name, "kind": "mirror", "value": 1, **r})
        print("synthetic", name, "done", flush=True)
    save_csv(rows, f"{OUT}/synthetic.csv")

    fig, axes = plt.subplots(2, len(SYN), figsize=(4 * len(SYN), 7))
    for c, kind in enumerate(SYN):
        for name in METHODS:
            sub = [r for r in rows if r["method"] == name and r["kind"] == kind]
            xs = [str(r["value"]) for r in sub]
            axes[0, c].plot(xs, [r["precision"] for r in sub], "o-", label=name)
            axes[1, c].plot(xs, [r["rep"] for r in sub], "o-", label=name)
        axes[0, c].set_title(SYN_TITLES[kind])
        axes[0, c].set_ylim(0, 1.02)
        axes[1, c].set_ylim(0, 1.02)
        for ax in axes[:, c]:
            ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("доля верных совпадений")
    axes[1, 0].set_ylabel("повторяемость точек")
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/synthetic.png", dpi=90)
    plt.close(fig)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    base, _ = ablation()
    subsets(base)
    compare_methods()
    synthetic()
    sift_sweeps()


if __name__ == "__main__":
    main()
