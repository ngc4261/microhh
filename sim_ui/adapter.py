"""sim-hub adapter. MHH = MicroHH (GPU, single precision) Weisman–Klemp 1982 supercell.

Contract: vault `Claude Code 運用/SIM_UI_DOCKER_RULES.md` §3 — four functions, none raises.
Patterned on portUrb's sim_ui/adapter.py (URB).

Run layout (all in run_dir = /work):
  weisman_klemp.ini      written here by prepare()
  run.log                container stdout: stage markers from docker/run_wk.sh
  weisman_klemp.out      MicroHH iteration table (ITER TIME CPUDT DT CFL ...), flushed every line
  *.xy.000.KKKKK.TTTTTTT raw float32 horizontal cross sections (var, level index, model time)
"""
from __future__ import annotations

import glob
import json
import math
import pathlib
import re

ADAPTER_API_VERSION = 1

# cases/weisman_klemp/weisman_klemp.ini with the grid/time/output entries templated.
# pbot is a placeholder: docker/run_wk.sh writes the value printed by weisman_klemp_input.py
# (it depends on ktot/zsize; 95533.2 for 192 levels / 19.2 km, 95268.4 for 96 levels).
_INI = """[master]
npx=1
npy=1

[grid]
itot={itot}
jtot={jtot}
ktot={ktot}

xsize={xsize:.1f}
ysize={ysize:.1f}
zsize={zsize:.1f}

utrans=12.5

swspatialorder=2

[advec]
swadvec=2i62
cflmax=1.2

[diff]
swdiff=smag2
dnmax=0.3

[thermo]
swbasestate=anelastic
swthermo=moist
pbot=0
thvref0=300
swupdatebasestate=1

[micro]
swmicro=nsw6
cflmax=1.2
Nc0=250e6

[boundary]
mbcbot=noslip
mbctop=freeslip
sbcbot=flux
sbctop=flux
sbot=0
stop=0
sbot[thl]=0
stop[thl]=0
sbot[qt]=0
stop[qt]=0
swboundary=surface
z0m=0.0002
z0h=0.0002

[fields]
visc=1.e-5
svisc=1.e-5

rndseed=2
rndamp[thl]=0
rndamp[qt]=0
rndz=100.
rndexp=2.

[buffer]
swbuffer=1
zstart={zstart:.0f}
sigma=0.00223
beta=2.

[time]
starttime=0
endtime={endtime:.0f}
dt=6.
dtmax=30
savetime={endtime:.0f}
outputiter=10
adaptivestep=true
rkorder=4

[stats]
swstats=0

[cross]
swcross=1
sampletime={sampletime:.0f}
crosslist=thl,qlqi_path,qlqi,qr,qr_path,u,v,w
xy=2000,5000,8000
xz={xz:.0f}

[dump]
swdump=0

[limiter]
limitlist=qt,qr
"""


def schema() -> dict:
    return {
        "code": "MHH",
        "title": "MicroHH スーパーセル (WK1982、GPU 版)",
        "description": "MicroHH(C++/CUDA、非弾性 LES、単精度、NSW6 1 モーメント微物理)の "
                       "Weisman–Klemp 1982 理想化スーパーセル。初期プロファイル作成 → init → "
                       "温位バブル → run をコンテナ内で順に実行する。",
        "params": [
            {"name": "itot", "label": "水平格子数(x,y 共通)", "type": "int",
             "default": 96, "min": 32, "max": 768, "unit": ""},
            {"name": "ktot", "label": "鉛直格子数", "type": "int",
             "default": 48, "min": 24, "max": 192, "unit": ""},
            {"name": "xsize_km", "label": "領域一辺(x,y 共通)", "type": "float",
             "default": 76.8, "min": 20.0, "max": 200.0, "unit": "km"},
            {"name": "zsize_km", "label": "領域の高さ(12 km の対流圏界面より上)", "type": "float",
             "default": 19.2, "min": 15.0, "max": 25.0, "unit": "km"},
            {"name": "sim_time_s", "label": "積分時間", "type": "float",
             "default": 600.0, "min": 60.0, "max": 10800.0, "unit": "s"},
            {"name": "out_freq_s", "label": "断面出力の間隔", "type": "float",
             "default": 300.0, "min": 30.0, "max": 3600.0, "unit": "s"},
            {"name": "bubble_amp_K", "label": "バブル振幅", "type": "float",
             "default": 2.0, "min": 0.0, "max": 6.0, "unit": "K"},
            {"name": "bubble_z_km", "label": "バブル中心高度", "type": "float",
             "default": 1.4, "min": 0.5, "max": 5.0, "unit": "km"},
            {"name": "bubble_radius_km", "label": "バブル水平半径", "type": "float",
             "default": 10.0, "min": 1.0, "max": 50.0, "unit": "km"},
            {"name": "bubble_depth_km", "label": "バブル鉛直半径", "type": "float",
             "default": 1.4, "min": 0.2, "max": 5.0, "unit": "km"},
            {"name": "bubble_x_km", "label": "バブル中心 x(負なら領域の 1/3)", "type": "float",
             "default": -1.0, "min": -1.0, "max": 200.0, "unit": "km"},
            {"name": "bubble_y_km", "label": "バブル中心 y(負なら領域の中央)", "type": "float",
             "default": -1.0, "min": -1.0, "max": 200.0, "unit": "km"},
        ],
        "presets": {
            "quick": {"label": "下見(96x96x48, 800m格子, 10分)",
                      "values": {"itot": 96, "ktot": 48, "xsize_km": 76.8, "zsize_km": 19.2,
                                 "sim_time_s": 600.0, "out_freq_s": 300.0}},
            "standard": {"label": "本番相当(384x384x96, 200m格子, 2時間)※10分前後",
                         "values": {"itot": 384, "ktot": 96, "xsize_km": 76.8, "zsize_km": 19.2,
                                    "sim_time_s": 7200.0, "out_freq_s": 600.0}},
        },
        "estimate_hint": "格子点数 x 積分時間に比例。RTX 3090 実測: 192x192x96 で 1800 s が 22 秒",
    }


def estimate_seconds(params: dict) -> float:
    """RTX 3090 calibration (2026-09-23): 192x192x96, 1800 s -> 22 s wall, plus ~10 s of init."""
    try:
        itot = float(params.get("itot", 96))
        ktot = float(params.get("ktot", 48))
        t = float(params.get("sim_time_s", 600.0))
        per_point_second = 22.0 / (192.0 * 192.0 * 96.0 * 1800.0)
        return 10.0 + per_point_second * itot * itot * ktot * t
    except Exception:
        return float("nan")


def prepare(params: dict, run_dir) -> dict:
    run_dir = pathlib.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    itot = int(params.get("itot", 96))
    ktot = int(params.get("ktot", 48))
    xsize = float(params.get("xsize_km", 76.8)) * 1000.0
    zsize = float(params.get("zsize_km", 19.2)) * 1000.0
    endtime = float(params.get("sim_time_s", 600.0))
    out_freq = float(params.get("out_freq_s", 300.0))

    (run_dir / "mhh_params.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    ini = _INI.format(itot=itot, jtot=itot, ktot=ktot, xsize=xsize, ysize=xsize, zsize=zsize,
                      zstart=max(zsize - 4200.0, 0.75 * zsize), endtime=endtime,
                      sampletime=out_freq, xz=xsize / 2.0)
    (run_dir / "weisman_klemp.ini").write_text(ini, encoding="utf-8")

    bubble = ["--bubamp", str(float(params.get("bubble_amp_K", 2.0))),
              "--zbub", f"{float(params.get('bubble_z_km', 1.4)) * 1000.0:.0f}",
              "--lxybub", f"{float(params.get('bubble_radius_km', 10.0)) * 1000.0:.0f}",
              "--lzbub", f"{float(params.get('bubble_depth_km', 1.4)) * 1000.0:.0f}",
              "--seed", "2"]
    bx = float(params.get("bubble_x_km", -1.0))
    by = float(params.get("bubble_y_km", -1.0))
    if bx >= 0:
        bubble += ["--xbub", f"{bx * 1000.0:.0f}"]
    if by >= 0:
        bubble += ["--ybub", f"{by * 1000.0:.0f}"]

    return {
        "command": ["/app/run_wk.sh", *bubble],
        "workdir": "/work",
        "needs": [],
        "env": {},
        "ulimits": {},
    }


def _last_out_row(run_dir: pathlib.Path):
    """Last data row of weisman_klemp.out as floats, or None."""
    try:
        lines = (run_dir / "weisman_klemp.out").read_text(
            encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            try:
                return [float(p) for p in parts]
            except ValueError:
                continue
    return None


def progress(run_dir) -> dict:
    run_dir = pathlib.Path(run_dir)
    state = {"percent": 0.0, "message": "まだログがありません(起動待ち)",
             "finished": False, "failed": False}
    try:
        log = (run_dir / "run.log").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return state

    m = re.search(r"=== STAGE: failed rc=(\d+)", log)
    if m or "ERROR" in log or "Segmentation fault" in log:
        state.update(percent=100.0, failed=True,
                     message=f"異常終了しました(rc={m.group(1) if m else '?'}、run.log を見てください)")
        return state

    try:
        target = float(json.loads((run_dir / "mhh_params.json").read_text()).get("sim_time_s", 600.0))
    except Exception:
        target = None

    if "=== STAGE: run start" not in log:
        stage = "初期プロファイル作成" if "=== STAGE: input" in log else "起動中"
        if "=== STAGE: init" in log:
            stage = "初期化(microhh init)"
        if "=== STAGE: bubble" in log:
            stage = "温位バブルを置いています"
        state.update(percent=2.0, message=stage)
        return state

    row = _last_out_row(run_dir)
    if row is not None and target and target > 0:
        sim_t = row[1]
        pct = min(100.0, 100.0 * sim_t / target)
        state.update(percent=max(pct, 3.0),
                     message=f"モデル時間 {sim_t:.0f}s / {target:.0f}s(ステップ {int(row[0])}、Δt {row[3]:.1f}s)")
    else:
        state.update(percent=3.0, message="計算中(weisman_klemp.out 待ち)")

    if "=== STAGE: run finished rc=0" in log:
        sim_t = row[1] if row is not None else (target or 0.0)
        state.update(percent=100.0, finished=True,
                     message=f"完走しました(モデル時間 {sim_t:.0f}s)")
    return state


def _write_metrics(run_dir: pathlib.Path) -> bool:
    row = _last_out_row(run_dir)
    if row is None:
        return False
    keys = ["iter", "model_time_s", "cpu_dt_s", "dt_s", "cfl", "dnum", "div", "mom", "tke", "mass"]
    metrics = {k: v for k, v in zip(keys, row) if isinstance(v, float) and math.isfinite(v)}
    try:
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError:
        return False
    return True


def _plot_cross_sections(run_dir: pathlib.Path) -> list[dict]:
    """PNG of the last horizontal cross section of qlqi and qr at each output level.

    Files are raw float32 (single-precision build), itot x jtot. Runs inside the project
    container (numpy/matplotlib are in the image); on the hub these imports may fail and
    then no figures are returned.
    """
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    try:
        p = json.loads((run_dir / "mhh_params.json").read_text())
        itot = int(p.get("itot", 96))
        ktot = int(p.get("ktot", 48))
        xsize_km = float(p.get("xsize_km", 76.8))
        zsize_km = float(p.get("zsize_km", 19.2))
    except Exception:
        return []
    dz_km = zsize_km / ktot
    items: list[dict] = []
    fig_dir = run_dir / "figures"
    for var, label, cmap in (("qlqi", "cloud water+ice qlqi", "Blues"), ("qr", "rain qr", "viridis")):
        files = sorted(run_dir.glob(f"{var}.xy.000.*.*"))
        if not files:
            continue
        last_t = max(f.name.split(".")[-1] for f in files)
        for f in sorted(files):
            level, t = f.name.split(".")[-2], f.name.split(".")[-1]
            if t != last_t:
                continue
            try:
                d = np.fromfile(f, dtype=np.float32)
                if d.size != itot * itot:
                    continue
                d = d.reshape(itot, itot) * 1e3
                fig_dir.mkdir(exist_ok=True)
                fig, ax = plt.subplots(figsize=(6, 5.4), constrained_layout=True)
                im = ax.imshow(d, origin="lower", extent=[0, xsize_km, 0, xsize_km], cmap=cmap,
                               vmin=0, vmax=max(float(d.max()), 1e-6))
                z_km = int(level) * dz_km
                ax.set_title(f"{label}  z = {z_km:.1f} km  t = {int(t)} s")
                ax.set_xlabel("x [km]")
                ax.set_ylabel("y [km]")
                fig.colorbar(im, ax=ax, label="g/kg")
                out = fig_dir / f"{var}_xy_k{level}_t{int(t):07d}.png"
                fig.savefig(out, dpi=110)
                plt.close(fig)
                items.append({"path": f"figures/{out.name}", "kind": "png",
                              "caption": f"{label}、高度 {z_km:.1f} km、モデル時間 {int(t)} s の水平断面"})
            except Exception:
                continue
    return items


def results(run_dir) -> list[dict]:
    run_dir = pathlib.Path(run_dir)
    items: list[dict] = []
    if _write_metrics(run_dir):
        items.append({"path": "metrics.json", "kind": "metrics",
                      "caption": "最終ステップの要約(モデル時間、Δt、CFL、運動量・TKE・質量の領域積分)"})
    try:
        items += _plot_cross_sections(run_dir)
    except Exception:
        pass
    return items
