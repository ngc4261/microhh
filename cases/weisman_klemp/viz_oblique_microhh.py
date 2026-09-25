#!/usr/bin/env python3
"""MicroHH weisman_klemp binary output -> oblique 3-D videos (no GPU needed).

Reads MicroHH's raw binary dumps (float32, shape (ktot, jtot, itot)):
  - qlqi.<time>       3-D cloud condensate (kg/kg), written every [dump] sampletime (100 s here)
  - w.<time>          3-D vertical velocity, only at [time] savetime restarts (3600 s here)
  - qr_path.xy.000.<time>  2-D rain-water path (kg/m^2), every [cross] sampletime (60 s)
  - grid.0000000      x, xh, y, yh, z, zh (float32)

Draws, for each frame time:
  - storm outline: isosurface of qlqi (g/kg), coloured by height
  - updraft core: semi-transparent red isosurface of w, only for times where a 3-D w file exists
  - floor: rain-water path (kg/m^2)
The grid is subsampled (default every 2nd level, every 3rd column: 768x768x192 -> 256x256x96)
because a 113-million-point isosurface is slow; the vertical axis is exaggerated (VERT_EXAG).

Usage:
    python viz_oblique_microhh.py <run_dir> <out_prefix> [--every 600] [--cond-gkg 0.5] [--w-ms 5]
                                  [--sub 2,3,3]
Writes <out_prefix>_evolution.mp4, <out_prefix>_rotation.mp4 (last time, 360-degree orbit)
and <out_prefix>_final.png.
"""
import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
VERT_EXAG = 4.0


def read_ini_grid(run_dir):
    g = {}
    with open(os.path.join(run_dir, "weisman_klemp.ini")) as f:
        for line in f:
            k = line.split("=")[0].strip()
            if k in ("itot", "jtot", "ktot"):
                g[k] = int(line.split("=")[1])
            elif k in ("endtime",):
                g[k] = float(line.split("=")[1])
    return g


def read_grid(run_dir, itot, jtot, ktot):
    a = np.fromfile(os.path.join(run_dir, "grid.0000000"), dtype=np.float32)
    n = 2 * itot + 2 * jtot + 2 * ktot
    if a.size != n:
        raise SystemExit(f"grid.0000000 has {a.size} values, expected {n} (float32)")
    x = a[:itot]
    y = a[2 * itot:2 * itot + jtot]
    z = a[2 * itot + 2 * jtot:2 * itot + 2 * jtot + ktot]
    return x.astype(np.float64), y.astype(np.float64), z.astype(np.float64)


def read3d(path, ktot, jtot, itot, sub):
    kz, ky, kx = sub
    a = np.fromfile(path, dtype=np.float32).reshape(ktot, jtot, itot)
    return np.ascontiguousarray(a[::kz, ::ky, ::kx])


def read2d(path, jtot, itot, sub):
    _, ky, kx = sub
    a = np.fromfile(path, dtype=np.float32).reshape(jtot, itot)
    return np.ascontiguousarray(a[::ky, ::kx])


def build_scene(x, y, z, cond_gkg_field, w_field, floor_field, cond_gkg, w_ms, label):
    nx, ny, nz = x.size, y.size, z.size
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0]); dz = float(z[1] - z[0])
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(dx, dy, dz * VERT_EXAG),
                        origin=(float(x[0]), float(y[0]), float(z[0]) * VERT_EXAG))
    grid.point_data["cond"] = cond_gkg_field.transpose(2, 1, 0).flatten(order="F")
    grid.point_data["height_km"] = grid.points[:, 2] / VERT_EXAG / 1000.0
    if w_field is not None:
        grid.point_data["w"] = w_field.transpose(2, 1, 0).flatten(order="F")

    pl = pv.Plotter(off_screen=True, window_size=(1280, 960))
    pl.set_background("black")

    floor = pv.ImageData(dimensions=(nx, ny, 1), spacing=(dx, dy, 1),
                         origin=(float(x[0]), float(y[0]), -dz * VERT_EXAG * 0.5))
    floor.point_data["qr_path"] = floor_field.flatten(order="F")
    pl.add_mesh(floor, scalars="qr_path", cmap="turbo", clim=[0, 5],
                show_scalar_bar=False, opacity=0.55)

    try:
        storm = grid.contour(isosurfaces=[cond_gkg], scalars="cond")
        if storm.n_points > 0:
            pl.add_mesh(storm, scalars="height_km", cmap="viridis", clim=[0, 15],
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "height km", "color": "white"},
                        smooth_shading=True, specular=0.3)
    except Exception:
        pass
    if w_field is not None:
        try:
            updraft = grid.contour(isosurfaces=[w_ms], scalars="w")
            if updraft.n_points > 0:
                pl.add_mesh(updraft, color="red", opacity=0.45, smooth_shading=True)
        except Exception:
            pass

    pl.add_text(label, position="upper_left", font_size=11, color="white")
    wtxt = f"red: w > {w_ms:g} m/s (3-D w only every 3600 s)" if w_field is not None else f"(no 3-D w at this time; red w > {w_ms:g} m/s shown only every 3600 s)"
    pl.add_text(f"cloud: qlqi {cond_gkg:g} g/kg (colour = height)   {wtxt}   floor: rain-water path kg/m2",
                position="lower_left", font_size=9, color="white")
    return pl, (nx * dx, ny * dy, nz * dz * VERT_EXAG)


def set_oblique_camera(pl, extent, azimuth_deg, elevation_deg=32.0, dist_factor=1.9):
    lx, ly, lz = extent
    cx, cy, cz = lx * 0.5, ly * 0.5, lz * 0.25
    r = lx * dist_factor
    az = np.deg2rad(azimuth_deg); el = np.deg2rad(elevation_deg)
    pl.camera_position = [(cx + r * np.cos(el) * np.cos(az), cy + r * np.cos(el) * np.sin(az),
                           cz + r * np.sin(el)), (cx, cy, cz), (0, 0, 1)]
    pl.camera.view_angle = 35


def write_mp4(frames, path, fps):
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None) as wr:
        for fr in frames:
            wr.append_data(fr)
    print(f"  wrote {path} ({len(frames)} frames @ {fps} fps)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--every", type=int, default=600, help="frame interval in seconds")
    ap.add_argument("--cond-gkg", type=float, default=0.5, help="qlqi isosurface (g/kg)")
    ap.add_argument("--w-ms", type=float, default=5.0, help="updraft isosurface (m/s)")
    ap.add_argument("--sub", default="2,3,3", help="subsampling k,j,i")
    ap.add_argument("--title", default="MicroHH WK1982 supercell")
    ap.add_argument("--rot-frames", type=int, default=72)
    args = ap.parse_args()
    sub = tuple(int(s) for s in args.sub.split(","))

    g = read_ini_grid(args.run_dir)
    itot, jtot, ktot = g["itot"], g["jtot"], g["ktot"]
    x, y, z = read_grid(args.run_dir, itot, jtot, ktot)
    xs, ys, zs = x[::sub[2]], y[::sub[1]], z[::sub[0]]
    endtime = int(g.get("endtime", 0))

    # only complete dumps (the 3-hour run has 9 truncated qlqi files at 1000-1800 s, 14 MB instead of 453 MB)
    expected = itot * jtot * ktot * 4
    times, skipped = [], []
    for t in range(0, endtime + 1, args.every):
        p = os.path.join(args.run_dir, f"qlqi.{t:07d}")
        if not os.path.exists(p):
            continue
        (times if os.path.getsize(p) == expected else skipped).append(t)
    if skipped:
        print(f"skipping {len(skipped)} truncated dumps: {skipped}", flush=True)
    if not times:
        raise SystemExit("no complete qlqi.<time> dumps found")
    print(f"{len(times)} frames: {times[0]}..{times[-1]} s every {args.every} s; grid {itot}x{jtot}x{ktot} -> "
          f"{xs.size}x{ys.size}x{zs.size}", flush=True)

    def load(t):
        cond = 1000.0 * read3d(os.path.join(args.run_dir, f"qlqi.{t:07d}"), ktot, jtot, itot, sub)
        wpath = os.path.join(args.run_dir, f"w.{t:07d}")
        w = read3d(wpath, ktot, jtot, itot, sub) if os.path.exists(wpath) else None
        fpath = os.path.join(args.run_dir, f"qr_path.xy.000.{t:07d}")
        floor = read2d(fpath, jtot, itot, sub) if os.path.exists(fpath) else np.zeros((ys.size, xs.size), np.float32)
        return cond, w, floor

    frames = []
    t0 = time.time()
    for i, t in enumerate(times):
        cond, w, floor = load(t)
        label = (f"t = {t/3600:.2f} h   {args.title}   {itot}x{jtot}x{ktot} @ {x[1]-x[0]:.0f} m "
                 f"(shown {xs.size}x{ys.size}x{zs.size}, vert x{VERT_EXAG:.0f})")
        pl, extent = build_scene(xs, ys, zs, cond, w, floor, args.cond_gkg, args.w_ms, label)
        set_oblique_camera(pl, extent, azimuth_deg=220 + i * 2.5)
        frames.append(pl.screenshot(return_img=True))
        if i == len(times) - 1:
            pl.screenshot(f"{args.out_prefix}_final.png")
        pl.close()
        wmax = float(w.max()) if w is not None else float("nan")
        print(f"  frame {i+1}/{len(times)} t={t} s  qlqi max {cond.max():.2f} g/kg  w max {wmax:.1f}  ({time.time()-t0:.0f} s)",
              flush=True)
    write_mp4([fr for fr in frames for _ in range(4)], f"{args.out_prefix}_evolution.mp4", fps=6)

    t = times[-1]
    cond, w, floor = load(t)
    label = f"t = {t/3600:.2f} h   {args.title}   (vert x{VERT_EXAG:.0f})"
    pl, extent = build_scene(xs, ys, zs, cond, w, floor, args.cond_gkg, args.w_ms, label)
    rot = []
    for i in range(args.rot_frames):
        set_oblique_camera(pl, extent, azimuth_deg=360.0 * i / args.rot_frames)
        rot.append(pl.screenshot(return_img=True))
    pl.close()
    write_mp4(rot, f"{args.out_prefix}_rotation.mp4", fps=24)
    print(f"done in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
