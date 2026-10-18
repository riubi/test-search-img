import itertools
import json
import os
import time

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

IMAGES = [f"images/bosmina_{i}.jpg" for i in range(1, 7)]
OUT_DIR = "results"

MIRROR = {2, 3, 5}

PARAMS = {
    "method": "SIFT",
    "nfeatures": 0,
    "nOctaveLayers": 3,
    "contrastThreshold": 0.04,
    "edgeThreshold": 10,
    "sigma": 1.6,
    "desc_scale": 1.0,
    "upright": False,
    "rootsift": False,
    "obj_size": 400,
    "clahe": True,
    "mirror": True,
    "dedup": 0.015,
    "ratio": 0.9,
    "window": 0.3,
    "ransac_thr": 0.05,
    "size_ratio": 2.0,
    "min_inliers": 6,
    "top_n": 30,
}


def organism_mask(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    m = cv2.bitwise_or(edges, dark)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    return mask


def prepare(path, idx, p):
    color = cv2.imread(path)
    if color is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    k0 = 1000 / max(gray.shape)
    small = cv2.resize(gray, None, fx=k0, fy=k0, interpolation=cv2.INTER_AREA)
    mask_small = organism_mask(small)
    area = cv2.countNonZero(mask_small) / k0 ** 2

    k = p["obj_size"] / np.sqrt(area)
    img = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask_small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if p["clahe"]:
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)

    mirrored = p["mirror"] and idx in MIRROR
    if mirrored:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    m = cv2.moments(mask, binaryImage=True)
    center = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)
    return {"idx": idx, "name": os.path.basename(path), "color": color, "img": img, "mask": mask,
            "scale": k, "mirrored": mirrored, "center": center}


def to_original(pts, item):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2).copy()
    if item["mirrored"]:
        pts[:, 0] = item["img"].shape[1] - 1 - pts[:, 0]
    return (pts + 0.5) / item["scale"] - 0.5


def make_detector(p):
    name = p["method"]
    if name == "SIFT":
        det = cv2.SIFT_create(nfeatures=p["nfeatures"], nOctaveLayers=p["nOctaveLayers"],
                              contrastThreshold=p["contrastThreshold"],
                              edgeThreshold=p["edgeThreshold"], sigma=p["sigma"])
        return det, cv2.NORM_L2
    if name == "ORB":
        det = cv2.ORB_create(nfeatures=p.get("orb_nfeatures", 3000), scaleFactor=1.2, nlevels=8,
                             edgeThreshold=31, patchSize=31, fastThreshold=p.get("fast_thr", 10))
        return det, cv2.NORM_HAMMING
    if name == "AKAZE":
        det = cv2.AKAZE_create(threshold=p.get("akaze_thr", 0.0005))
        return det, cv2.NORM_HAMMING
    if name == "BRISK":
        det = cv2.BRISK_create(thresh=p.get("brisk_thr", 30), octaves=3, patternScale=1.0)
        return det, cv2.NORM_HAMMING
    raise ValueError(name)


def remove_duplicates(kp, radius):
    if not kp or radius <= 0:
        return list(kp)
    pts = np.float32([k.pt for k in kp])
    taken = np.zeros(len(kp), bool)
    kept = []
    for i in sorted(range(len(kp)), key=lambda i: -kp[i].response):
        if taken[i]:
            continue
        kept.append(kp[i])
        taken |= ((pts - pts[i]) ** 2).sum(1) < radius ** 2
    return kept


def detect(det, item, p):
    kp_raw = det.detect(item["img"], item["mask"])
    kp = remove_duplicates(kp_raw, p["dedup"] * p["obj_size"])
    if p["method"] == "SIFT" and (p["desc_scale"] != 1 or p["upright"]):
        desc_kp = [cv2.KeyPoint(k.pt[0], k.pt[1], k.size * p["desc_scale"], 0 if p["upright"] else k.angle,
                                k.response, k.octave) for k in kp]
        desc_kp, des = det.compute(item["img"], desc_kp)
        assert len(desc_kp) == len(kp)
    else:
        kp, des = det.compute(item["img"], kp)
    if des is None:
        des = np.zeros((0, 128), np.float32)
    if p["method"] == "SIFT" and p["rootsift"]:
        des = np.sqrt(des / (des.sum(axis=1, keepdims=True) + 1e-7)).astype(np.float32)
    item["n_raw"] = len(kp_raw)
    item["kp"] = kp
    item["pts"] = np.float32([k.pt for k in kp]).reshape(-1, 2)
    item["sizes"] = np.float32([k.size for k in kp])
    item["des"] = des
    return item


def ratio_matches(bf, des_a, des_b, allowed, ratio):
    res = {}
    for pair in bf.knnMatch(des_a, des_b, k=2, mask=allowed):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            res[m.queryIdx] = (m.trainIdx, m.distance)
    return res


def size_ok(sa, sb, scale, p):
    r = sa * scale / sb
    return 1 / p["size_ratio"] < r < p["size_ratio"]


def mutual_matches(bf, a, b, allowed, p, scale=1.0):
    ab = ratio_matches(bf, a["des"], b["des"], allowed, p["ratio"])
    ba = ratio_matches(bf, b["des"], a["des"], np.ascontiguousarray(allowed.T), p["ratio"])
    good = [(i, j, d) for i, (j, d) in ab.items()
            if j in ba and ba[j][0] == i and size_ok(a["sizes"][i], b["sizes"][j], scale, p)]
    return ab, good


def window_mask(pts_pred, pts_b, radius):
    d2 = ((pts_pred[:, None, :] - pts_b[None, :, :]) ** 2).sum(-1)
    return (d2 < radius ** 2).astype(np.uint8)


def transform_scale(T):
    return float(np.sqrt(abs(np.linalg.det(T[:2, :2]))))


def match_pair(a, b, norm, p):
    S = p["obj_size"]
    st = {"pair": (a["idx"], b["idx"]), "ratio": 0, "mutual": 0, "inliers": 0, "final": 0, "ok": False}
    if len(a["kp"]) < 2 or len(b["kp"]) < 2:
        return st, [], None
    bf = cv2.BFMatcher(norm)

    pred = a["pts"] - a["center"] + b["center"]
    allowed = window_mask(pred, b["pts"], p["window"] * S)
    ab, good = mutual_matches(bf, a, b, allowed, p)
    st["ratio"], st["mutual"] = len(ab), len(good)
    if len(good) < 3:
        return st, [], None

    src = np.float32([a["pts"][i] for i, _, _ in good])
    dst = np.float32([b["pts"][j] for _, j, _ in good])
    T, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=p["ransac_thr"] * S,
                                         maxIters=5000, confidence=0.999)
    if T is None:
        return st, [], None
    st["inliers"] = int(inl.sum())
    st["scale"] = transform_scale(T)
    st["angle"] = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))

    pred = cv2.transform(a["pts"][None], T)[0]
    allowed = window_mask(pred, b["pts"], p["ransac_thr"] * S)
    _, refined = mutual_matches(bf, a, b, allowed, p, st["scale"])
    st["final"] = len(refined)
    st["ok"] = st["inliers"] >= p["min_inliers"] and 0.7 < st["scale"] < 1.4
    return st, refined, T


def match_all(items, norm, p):
    pairs, stats = {}, []
    for a, b in itertools.combinations(range(len(items)), 2):
        st, m, T = match_pair(items[a], items[b], norm, p)
        pairs[a, b] = {"matches": m, "T": T, "ok": st["ok"], "inliers": st["inliers"]}
        stats.append(st)
    return pairs, stats


def get_pair(pairs, a, b):
    if a < b:
        return {i: (j, d) for i, j, d in pairs[a, b]["matches"]}
    return {j: (i, d) for i, j, d in pairs[b, a]["matches"]}


def to3x3(T):
    return np.vstack([T, [0, 0, 1]])


def pair_transform(pairs, a, b):
    T = pairs[min(a, b), max(a, b)]["T"]
    if T is None:
        return None
    T = to3x3(T)
    return T if a < b else np.linalg.inv(T)


def mask_iou(a, b, T):
    h, w = b["mask"].shape
    warped = cv2.warpAffine(a["mask"], T[:2], (w, h), flags=cv2.INTER_NEAREST)
    inter = cv2.countNonZero(cv2.bitwise_and(warped, b["mask"]))
    union = cv2.countNonZero(cv2.bitwise_or(warped, b["mask"]))
    return inter / union if union else 0.0


def align_to_reference(items, pairs, ref, subset):
    chain = {ref: np.eye(3)}
    while len(chain) < len(subset):
        best = None
        for (a, b), pr in pairs.items():
            if not pr["ok"] or a not in subset or b not in subset:
                continue
            for u, w in ((a, b), (b, a)):
                if u in chain and w not in chain and (best is None or pr["inliers"] > best[0]):
                    best = (pr["inliers"], u, w)
        if best is None:
            break
        _, u, w = best
        chain[w] = pair_transform(pairs, u, w) @ chain[u]

    result = {}
    for j in subset:
        if j == ref:
            continue
        cands = []
        T = pair_transform(pairs, ref, j)
        if T is not None:
            cands.append(("прямое", T))
        if j in chain:
            cands.append(("через цепочку", chain[j]))
        shift = np.eye(3)
        shift[:2, 2] = items[j]["center"] - items[ref]["center"]
        cands.append(("по силуэту", shift))
        scored = [(mask_iou(items[ref], items[j], T), name, T) for name, T in cands]
        iou, name, T = max(scored, key=lambda s: s[0])
        result[j] = {"T": T, "source": name, "iou": iou}
    return result


def guided_matches(a, b, T, norm, p):
    bf = cv2.BFMatcher(norm)
    pred = cv2.transform(a["pts"][None], T[:2])[0]
    allowed = window_mask(pred, b["pts"], p["ransac_thr"] * p["obj_size"])
    ab = ratio_matches(bf, a["des"], b["des"], allowed, p["ratio"])
    s = transform_scale(T)
    best = {}
    for i, (j, d) in ab.items():
        if size_ok(a["sizes"][i], b["sizes"][j], s, p) and (j not in best or d < best[j][1]):
            best[j] = (i, d)
    return {i: (j, d) for j, (i, d) in best.items()}


def common_points(items, pairs, norm, p, subset=None):
    subset = sorted(subset if subset is not None else range(len(items)))
    weight = [sum(pr["inliers"] for (a, b), pr in pairs.items()
                  if pr["ok"] and r in (a, b) and a in subset and b in subset) for r in subset]
    ref = subset[int(np.argmax(weight))]
    align = align_to_reference(items, pairs, ref, subset)
    others = sorted(align)
    maps = {j: guided_matches(items[ref], items[j], align[j]["T"], norm, p) for j in others}
    for j in others:
        align[j]["matches"] = len(maps[j])

    seen = np.zeros(len(items[ref]["kp"]), int)
    for j in others:
        for i in maps[j]:
            seen[i] += 1
    hist = {k: int((seen == k).sum()) for k in range(1, len(subset))}

    min_seen = max(1, len(others) - 2)
    tracks = []
    for i in np.where(seen >= min_seen)[0]:
        i = int(i)
        track = {ref: i}
        dists = []
        for j in others:
            if i in maps[j]:
                track[j] = maps[j][i][0]
                dists.append(maps[j][i][1])
        confirmed = 0
        for j, k in itertools.combinations(others, 2):
            if j in track and k in track:
                m = get_pair(pairs, j, k)
                if track[j] in m and m[track[j]][0] == track[k]:
                    confirmed += 1
        tracks.append({"points": track, "mean_dist": float(np.mean(dists)), "confirmed": confirmed})

    tracks.sort(key=lambda t: (-len(t["points"]), -t["confirmed"], t["mean_dist"]))
    common = [t for t in tracks if len(t["points"]) == len(subset)]
    return {"ref": ref, "tracks": tracks, "common": common, "align": align, "maps": maps, "hist": hist}


def track_colors(n):
    cmap = plt.get_cmap("hsv")
    return [cmap(i / max(n, 1)) for i in range(n)]


def draw_common(items, tracks, path, ref, top_n=30, title=None):
    tracks = tracks[:top_n]
    colors = track_colors(len(tracks))
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5.2))
    for ax, it in zip(axes.ravel(), items):
        ax.imshow(cv2.cvtColor(it["color"], cv2.COLOR_BGR2RGB))
        h, w = it["color"].shape[:2]
        found = 0
        for num, (t, c) in enumerate(zip(tracks, colors), 1):
            if it["pos"] not in t["points"]:
                continue
            found += 1
            x, y = to_original(it["pts"][t["points"][it["pos"]]], it)[0]
            ax.plot(x, y, "o", ms=8, mfc=c, mec="black", mew=1.2)
            ax.text(x + 0.015 * w, y - 0.015 * w, str(num), color=c, fontsize=11, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
        label = it["name"] + ("  [опорное]" if it["pos"] == ref else "") + ("  [отражено]" if it["mirrored"] else "")
        ax.set_title(f"{label}\n{w}x{h}, точек на снимке: {found} из {len(tracks)}", fontsize=11)
        ax.axis("off")
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_keypoints(items, path):
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.6))
    for ax, it in zip(axes.ravel(), items):
        vis = cv2.drawKeypoints(it["img"], it["kp"], None, color=(0, 200, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cnt, _ = cv2.findContours(it["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnt, -1, (0, 0, 255), 2)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{it['name']}: найдено {it['n_raw']}, после прореживания {len(it['kp'])}", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def draw_alignment(items, ref, align, path):
    cnt, _ = cv2.findContours(items[ref]["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnt, key=cv2.contourArea).astype(np.float32)
    fig, axes = plt.subplots(1, len(items) - 1, figsize=(4 * (len(items) - 1), 4))
    for ax, j in zip(axes, sorted(align)):
        c = cv2.transform(cnt, align[j]["T"][:2])[:, 0]
        ax.imshow(items[j]["img"], cmap="gray")
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], "r-", lw=1.5)
        ax.set_title(f"{items[j]['name']}\n{align[j]['source']}, IoU={align[j]['iou']:.2f}", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def draw_matches(a, b, m, path, limit=60):
    dm = sorted((cv2.DMatch(i, j, float(d)) for i, (j, d) in m.items()), key=lambda x: x.distance)[:limit]
    vis = cv2.drawMatches(a["img"], a["kp"], b["img"], b["kp"], dm, None,
                          matchColor=(0, 0, 255), singlePointColor=(170, 170, 170), flags=0)
    cv2.imwrite(path, vis)


def run(p=None, verbose=True):
    p = dict(PARAMS, **(p or {}))
    det, norm = make_detector(p)
    items = []
    t0 = time.time()
    for pos, path in enumerate(IMAGES):
        it = prepare(path, pos + 1, p)
        it["pos"] = pos
        items.append(detect(det, it, p))
    t_det = time.time() - t0
    pairs, stats = match_all(items, norm, p)
    res = common_points(items, pairs, norm, p)
    res.update(params=p, items=items, pairs=pairs, stats=stats, norm=norm, t_det=t_det)
    if verbose:
        print_report(res)
    return res


def print_report(res):
    items, p, ref = res["items"], res["params"], res["ref"]
    print(f"метод {p['method']}, время детекции {res['t_det']:.2f} c")
    for it in items:
        print(f"  {it['name']}: найдено {it['n_raw']} ключевых точек, после прореживания {len(it['kp'])}")
    print("  пара  тест Лоу  взаимн.  RANSAC  итог  масштаб  угол")
    for st in res["stats"]:
        a, b = st["pair"]
        extra = f"{st['scale']:.2f}  {st['angle']:6.1f}" if "scale" in st else "-"
        print(f"  {a}-{b}  {st['ratio']:7d}  {st['mutual']:7d}  {st['inliers']:6d}  {st['final']:4d}  {extra}"
              + ("" if st["ok"] else "  (ненадёжно)"))
    print(f"  опорное изображение: {items[ref]['name']}")
    for j, al in res["align"].items():
        print(f"    -> {items[j]['name']}: {al['source']}, IoU {al['iou']:.2f}, соответствий {al['matches']}")
    print(f"  точек опорного, найденных на k других снимках: {res['hist']}")
    print(f"  общих точек на всех {len(items)} изображениях: {len(res['common'])}")


def track_rows(items, tracks):
    rows = []
    for num, t in enumerate(tracks, 1):
        row = {"n": num, "images": len(t["points"]), "confirmed": t["confirmed"],
               "mean_dist": round(t["mean_dist"], 1)}
        for it in items:
            if it["pos"] in t["points"]:
                x, y = to_original(it["pts"][t["points"][it["pos"]]], it)[0]
                row[it["name"]] = [round(float(x), 1), round(float(y), 1)]
            else:
                row[it["name"]] = None
        rows.append(row)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    res = run()
    items, p, ref, align = res["items"], res["params"], res["ref"], res["align"]
    n = len(items)

    draw_keypoints(items, f"{OUT_DIR}/keypoints.png")
    draw_alignment(items, ref, align, f"{OUT_DIR}/alignment.png")
    draw_common(items, res["common"], f"{OUT_DIR}/common_points.png", ref, p["top_n"],
                f"Точки, общие для всех {n} изображений ({len(res['common'])})")
    draw_common(items, res["tracks"], f"{OUT_DIR}/common_points_partial.png", ref, p["top_n"],
                f"Лучшие точки, найденные хотя бы на {n - 2} из {n} изображений")
    for j in align:
        draw_matches(items[ref], items[j], res["maps"][j], f"{OUT_DIR}/matches_{ref + 1}_{j + 1}.jpg")

    summary = {
        "keypoints_raw": {it["name"]: it["n_raw"] for it in items},
        "keypoints": {it["name"]: len(it["kp"]) for it in items},
        "pairs": res["stats"],
        "reference": items[ref]["name"],
        "to_reference": {items[j]["name"]: {"source": al["source"], "iou": round(al["iou"], 3),
                                             "matches": al["matches"]} for j, al in align.items()},
        "seen_in_k_images": res["hist"],
        "common_all": len(res["common"]),
        "found_on_n_images": {k: sum(len(t["points"]) == k for t in res["tracks"]) for k in range(n - 2, n + 1)},
        "tracks": track_rows(items, res["tracks"][:p["top_n"]]),
        "params": p,
    }
    with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
