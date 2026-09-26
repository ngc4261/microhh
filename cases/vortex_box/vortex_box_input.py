"""Initial profiles for vortex_box: neutral (theta 300 K), at rest, humid near the
surface (qt 10 g/kg, cloud base about 1 km if lifted) and 80 % relative humidity
aloft, so cloud forms only in the forced updraft and in the vortex core.
Writes vortex_box_input.nc (init group: thl, qt, u, v, ug, vg)."""
import numpy as np
import netCDF4 as nc

cp, Rd, Rv, p0, grav = 1005., 287.04, 461.5, 1.e5, 9.81
ep = Rd/Rv


def esat_liq(T):
    x = max(-80., T-273.15)
    c = [0.6105851e+03, 0.4440316e+02, 0.1430341e+01, 0.2641412e-01, 0.2995057e-03,
         0.2031998e-05, 0.6936113e-08, 0.2564861e-11, -.3704404e-13]
    r = c[8]
    for ci in reversed(c[:8]):
        r = ci + x*r
    return r


def qsat(p, T):
    return ep*esat_liq(T)/(p-(1.-ep)*esat_liq(T))


with open('vortex_box.ini') as f:
    items = dict(l.strip().split('=', 1) for l in f if '=' in l and not l.startswith('#'))
kmax, zsize = int(items['ktot']), float(items['zsize'])
dz = zsize/kmax
z = np.linspace(0.5*dz, zsize-0.5*dz, kmax)

theta0, qt0 = 300., 0.010
thl = np.full(kmax, theta0)
qt = np.zeros(kmax)
for k in range(kmax):
    # dry-adiabatic pressure and temperature of the neutral column
    T = theta0 - grav/cp*z[k]
    p = p0*(T/theta0)**(cp/Rd)
    qt[k] = min(qt0, 0.8*qsat(p, T))
zeros = np.zeros(kmax)

f = nc.Dataset("vortex_box_input.nc", mode="w", datamodel="NETCDF4", clobber=True)
f.createDimension("z", kmax)
f.createVariable("z", "f8", ("z"))[:] = z
g = f.createGroup("init")
for name, val in (("thl", thl), ("qt", qt), ("u", zeros), ("v", zeros), ("ug", zeros), ("vg", zeros)):
    g.createVariable(name, "f8", ("z"))[:] = val
f.close()
print(f"vortex_box_input.nc: {kmax} levels, qt {qt[0]*1e3:.1f} -> {qt[-1]*1e3:.1f} g/kg")
