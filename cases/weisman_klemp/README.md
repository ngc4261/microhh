# Weisman - Klemp case

This case is the deep convection experiment from Weisman & Klemp (1982, MWR) in which a warm bubble is released, which splits and forms a left and right moving storm. The case has now warm micro enabled and will serve as a benchmark case for deep convection once the full S&B ice microphysics is completed.

## Instructions
1. Run `python3 weisman_klemp_input.py`
1. Run `./microhh init weisman_klemp`
1. Run `python3 bubble_theta.py`
1. Run `./microhh run weisman_klemp`

## Options added in this fork (2026-09)

* `bubble_theta.py --xbub --ybub --zbub --lxybub --lzbub --bubamp --seed --no-noise`:
  position, radii and amplitude of the warm bubble (defaults are the original fixed values).
* `[thermo] swprecipbuoyancy=true` (default `false`): add the weight of the precipitation
  fields of the microphysics (qr, qs, qg) to the buoyancy tendency, as CM1 and most cloud
  models do. With the default, MicroHH's buoyancy uses only thl, qt, ql and qi, so rain and
  graupel inside the updraft do not slow it down. On an 800 m grid (250x250x50, 30 min,
  2 K bubble) this changes the peak updraft from about 65 to 56 m/s.
* `[diff] cs` (default 0.23): the Smagorinsky constant. On coarse grids (800 m) the
  updraft is barely diluted; cs=1.2 gives updrafts comparable to CM1 (WENO5) on the same grid.
  This is tuning, not physics.
* Note on the base state: `weisman_klemp_input.py` integrates the pressure downward from the
  tropopause, which gives a surface pressure of 946-955 hPa depending on the vertical grid
  (WK82 uses 1000 hPa). It does not change the updraft strength noticeably.
