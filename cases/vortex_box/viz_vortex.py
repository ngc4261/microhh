"""vortex_box (MicroHH, Fiedler-type vortex) の図: 鉛直断面、地上付近の風、雲の 3D 等値面。
使い方: python viz_vortex.py <出力フォルダ(all)> <図の置き場>"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "sans-serif"]

NI = NJ = 256
NK = 128
DX = 10000. / NI
DZ = 3200. / NK
x = (np.arange(NI) + 0.5) * DX / 1000.        # km
z = (np.arange(NK) + 0.5) * DZ / 1000.
zh = np.arange(NK) * DZ / 1000.


def rd(path, shape):
    return np.fromfile(path, dtype="<f4").reshape(shape)


def times(pattern):
    return sorted(int(p.rsplit(".", 1)[1]) for p in glob.glob(os.path.join(src, pattern)))


def xy(name, loc, k, t):
    return rd(os.path.join(src, f"{name}.xy.{loc}.{k:05d}.{t:07d}"), (NJ, NI))


def xz(name, loc, t):
    return rd(os.path.join(src, f"{name}.xz.{loc}.00128.{t:07d}"), (NK, NI))


# ---- 1. 鉛直断面(y = 5 km、渦の中心を通る)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, t in zip(axs, (600, 1200, 1800)):
    q = xz("qlqi", "000", t) * 1000.
    w = xz("w", "001", t)
    ax.pcolormesh(x, z, q, cmap="Blues", vmin=0, vmax=2, shading="auto")
    cs = ax.contour(x, zh, w, levels=[-4, 5, 10, 20, 30], colors=["navy", "orange", "red", "darkred", "k"],
                    linewidths=0.9)
    ax.clabel(cs, fontsize=7, fmt="%d")
    ax.set_xlim(2.5, 7.5)
    ax.set_title(f"{t // 60} 分: 雲(g/kg、青)と上昇流 w(m/s、線)")
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("高さ (km)")
fig.suptitle("MicroHH vortex_box — y = 5 km を通る鉛直断面(中央 5 km)", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(dst, "vortex_xz.png"), dpi=120)
plt.close(fig)

# ---- 2. 地上付近(62.5 m)の風速と時系列
ts = times("u.xy.*.00002.*")
k0 = 2
umax, wmax, vmax_t = [], [], []
for t in ts:
    u = xy("u", "100", k0, t)
    v = xy("v", "010", k0, t)
    uc = 0.5 * (u + np.roll(u, -1, axis=1))
    vc = 0.5 * (v + np.roll(v, -1, axis=0))
    umax.append(np.hypot(uc, vc).max())
    wmax.append(xz("w", "001", t).max())
t_last = ts[-1]
u = xy("u", "100", k0, t_last); v = xy("v", "010", k0, t_last)
spd = np.hypot(0.5 * (u + np.roll(u, -1, axis=1)), 0.5 * (v + np.roll(v, -1, axis=0)))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
m = a1.pcolormesh(x, x, spd, cmap="magma_r", shading="auto")
a1.set_xlim(3.5, 6.5); a1.set_ylim(3.5, 6.5); a1.set_aspect("equal")
st = 6
a1.quiver(x[::st], x[::st], (0.5 * (u + np.roll(u, -1, axis=1)))[::st, ::st],
          (0.5 * (v + np.roll(v, -1, axis=0)))[::st, ::st], color="w", scale=600, width=0.003)
fig.colorbar(m, ax=a1, label="風速 (m/s)")
a1.set_title(f"高さ 62 m の風速と風向({t_last // 60} 分、中央 3 km)")
a1.set_xlabel("x (km)"); a1.set_ylabel("y (km)")
tm = np.array(ts) / 60.
a2.plot(tm, umax, label="高さ 62 m の最大風速")
a2.plot(tm, wmax, label="断面内の最大上昇流 w")
a2.set_xlabel("時間 (分)"); a2.set_ylabel("m/s"); a2.grid(alpha=.3); a2.legend()
a2.set_title("渦の強まり")
fig.tight_layout()
fig.savefig(os.path.join(dst, "vortex_surface.png"), dpi=120)
plt.close(fig)

# ---- 3. 雲の 3D 等値面(0.1 g/kg)と強い上昇流(w = 15 m/s)
t_peak = ts[int(np.argmax(umax))]           # 地表付近の風が最も強い時刻に近い 3D 出力
t3 = min(times("qlqi.[0-9]*"), key=lambda t: abs(t - t_peak))
q3 = rd(os.path.join(src, f"qlqi.{t3:07d}"), (NK, NJ, NI))
w3 = rd(os.path.join(src, f"w.{t3:07d}"), (NK, NJ, NI))
sl = slice(64, 192)                       # 中央 5 km
surfs = [(q3, 1e-4, "lightsteelblue", 0.35), (w3, 15., "tomato", 0.6)]
title = f"{t3 // 60} 分: 雲 0.1 g/kg(青)と上昇流 15 m/s(赤)"
p_path = os.path.join(src, f"p.{t3:07d}")
if os.path.exists(p_path):
    # 低気圧の芯: 各高さの水平平均からの偏差 p'(運動学的気圧 m2/s2 → ×ρ≈1.1 で Pa)。
    # 渦の芯を包む面として、地表近くの最小値の 30% の等値面を描く
    p3 = rd(p_path, (NK, NJ, NI))
    pp = p3 - p3.mean(axis=(1, 2), keepdims=True)
    lev_p = 0.3 * pp[:10].min()
    surfs = [(q3, 1e-4, "lightsteelblue", 0.30), (-pp, -lev_p, "crimson", 0.75)]
    title = (f"{t3 // 60} 分: 雲 0.1 g/kg(青)と低気圧の芯 p' < {lev_p * 1.1 / 100:.1f} hPa(赤)\n"
             f"地表付近の p' 最小 {pp[:10].min() * 1.1 / 100:.1f} hPa")
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")
for fld, lev, col, a in surfs:
    sub = fld[:, sl, sl]
    if sub.max() <= lev:
        continue
    verts, faces, _, _ = measure.marching_cubes(sub, lev, step_size=2)
    kk, jj, ii = verts.T
    ax.plot_trisurf((ii + 64) * DX / 1000., (jj + 64) * DX / 1000., faces, kk * DZ / 1000.,
                    color=col, alpha=a, linewidth=0, shade=True)
ax.set_xlim(2.5, 7.5); ax.set_ylim(2.5, 7.5); ax.set_zlim(0, 3.2)
ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)"); ax.set_zlabel("高さ (km)")
ax.view_init(elev=12, azim=-60)
ax.set_title(title)
fig.tight_layout()
fig.savefig(os.path.join(dst, "vortex_3d.png"), dpi=120)
plt.close(fig)

print(f"max wind at 62 m: {max(umax):.1f} m/s (t={ts[int(np.argmax(umax))]} s), last {umax[-1]:.1f}")
print(f"max w in xz: {max(wmax):.1f} m/s")
