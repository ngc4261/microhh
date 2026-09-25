"""Add a warm bubble to the initial liquid-water potential temperature field (thl.0000000).

Bubble position, size and amplitude used to be hard-coded. They can now be given on the
command line; every default equals the previous hard-coded value, so running the script
without arguments produces the same field as before (given the same random seed).

    python bubble_theta.py                      # same as before
    python bubble_theta.py --bubamp 2 --zbub 1500 --lzbub 1500 --xbub 38400 --ybub 38400
    python bubble_theta.py --seed 0 --no-noise  # reproducible, no random noise

The bubble follows Weisman & Klemp (1982): theta' = A cos^2(pi r / 2) for r < 1, where
r = sqrt(((x-x0)/Lxy)^2 + ((y-y0)/Lxy)^2 + ((z-z0)/Lz)^2).
"""
import argparse
import math
import os

import numpy as np


def read_grid(ini_path):
    grid = {}
    with open(ini_path) as f:
        for line in f:
            key = line.split('=')[0].strip()
            if key in ('itot', 'jtot', 'ktot'):
                grid[key] = int(line.split('=')[1])
            elif key in ('xsize', 'ysize', 'zsize'):
                grid[key] = float(line.split('=')[1])
    missing = [k for k in ('itot', 'jtot', 'ktot', 'xsize', 'ysize', 'zsize') if k not in grid]
    if missing:
        raise SystemExit(f"{ini_path}: missing {', '.join(missing)}")
    return grid


def add_bubble(thl, dx, dy, dz, xbub, ybub, zbub, lxybub, lzbub, bubamp, noise=True, rng=None):
    """Add the bubble in place. Returns the number of modified points."""
    ktot, jtot, itot = thl.shape
    iminbub = max(int(round((xbub - lxybub) / dx)), 0)
    imaxbub = min(int(round((xbub + lxybub) / dx)) + 1, itot)
    jminbub = max(int(round((ybub - lxybub) / dy)), 0)
    jmaxbub = min(int(round((ybub + lxybub) / dy)) + 1, jtot)
    kminbub = max(int(round((zbub - lzbub) / dz)), 0)
    kmaxbub = min(int(round((zbub + lzbub) / dz)) + 1, ktot)
    if rng is None:
        rng = np.random
    n = 0
    for k in range(kminbub, kmaxbub):
        for j in range(jminbub, jmaxbub):
            for i in range(iminbub, imaxbub):
                dist = math.sqrt(((xbub - i * dx) / lxybub) ** 2
                                 + ((ybub - j * dy) / lxybub) ** 2
                                 + ((zbub - k * dz) / lzbub) ** 2)
                if dist < 1.0:
                    pert = bubamp * np.cos(dist * np.pi / 2) ** 2
                    if noise:
                        pert += bubamp * 0.01 * (rng.rand() - 0.5)
                    thl[k, j, i] = thl[k, j, i] + pert
                    n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ini', default='weisman_klemp.ini', help='case .ini with itot/jtot/ktot/xsize/ysize/zsize')
    p.add_argument('--file', default='thl.0000000', help='thl field to modify in place')
    p.add_argument('--xbub', type=float, default=None, help='bubble centre x (m); default xsize/3')
    p.add_argument('--ybub', type=float, default=None, help='bubble centre y (m); default ysize/2')
    p.add_argument('--zbub', type=float, default=1400., help='bubble centre height (m)')
    p.add_argument('--lxybub', type=float, default=10000., help='horizontal radius (m)')
    p.add_argument('--lzbub', type=float, default=1400., help='vertical radius (m)')
    p.add_argument('--bubamp', type=float, default=2., help='amplitude (K)')
    p.add_argument('--seed', type=int, default=None, help='seed for the +-0.5%% random noise')
    p.add_argument('--no-noise', action='store_true', help='omit the random noise')
    args = p.parse_args()

    g = read_grid(args.ini)
    dx = g['xsize'] / g['itot']
    dy = g['ysize'] / g['jtot']
    dz = g['zsize'] / g['ktot']
    xbub = (1. / 3.) * g['xsize'] if args.xbub is None else args.xbub
    ybub = (1. / 2.) * g['ysize'] if args.ybub is None else args.ybub

    # dtype from the file size (the old heuristic effectively always chose float32).
    npoints = g['itot'] * g['jtot'] * g['ktot']
    size = os.path.getsize(args.file)
    if size == npoints * 8:
        float_type = np.float64
    elif size == npoints * 4:
        float_type = np.float32
    else:
        raise SystemExit(f"{args.file}: size {size} is neither {npoints}*4 nor {npoints}*8 bytes "
                         f"(grid {g['itot']}x{g['jtot']}x{g['ktot']} from {args.ini})")
    thl = np.fromfile(args.file, dtype=float_type).reshape(g['ktot'], g['jtot'], g['itot'])

    rng = np.random.RandomState(args.seed) if args.seed is not None else np.random
    n = add_bubble(thl, dx, dy, dz, xbub, ybub, args.zbub, args.lxybub, args.lzbub, args.bubamp,
                   noise=not args.no_noise, rng=rng)
    thl.tofile(args.file)
    print(f"bubble: centre=({xbub:.0f}, {ybub:.0f}, {args.zbub:.0f}) m, "
          f"radius=({args.lxybub:.0f} h, {args.lzbub:.0f} v) m, amp={args.bubamp} K, "
          f"noise={'off' if args.no_noise else 'on'}, points modified={n}, dtype={float_type.__name__}")


if __name__ == '__main__':
    main()
