"""Rebuild MD-1.5's initial state and normalization statistics from ERA5 (the authors did not publish them).

ERA5 monthly means (DKRZ pool, reduced Gaussian N320) -> MD-1.5's 1.5 deg grid (121 x 240, poles included).

Follows the authors' data pipeline (kjhall01/monthly-diffusion MDv0.9, data-pipeline/regrid-era5.ipynb and
flatten_era5.ipynb): xESMF conservative_normed with cell edges from `edges_from_centers` (clipped to [-90, 90] and
[0, 360]), destination lat = linspace(-90, 90, 121), lon = arange(0, 360, 1.5). Their source was RDA ERA5 on a
0.25 deg grid; here the source is the pool's N320 grid, made regular (1280 points per row) by linear interpolation
along each row, as `cdo setgridtype,regular` does.

Writes era5_1p5_197810-201412.nc (float32, varlev x time), the statistics of 1979-01..2014-12 (to compare with the
1979-2014 file in the authors' repository) and 1985-01..2014-12 (the training period; the file aimip uses), computed
as the authors' preprocess_by_variables does (sqrt for q* and MTPR, clipped logit for CI, plain otherwise; mean and
population std over all values), and the October 1978 state (float64). Run at DKRZ with xESMF, eccodes and netCDF4:

    python make_inputs.py <out-dir> <processes>
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import eccodes
import netCDF4
import numpy as np
import scipy.sparse as sp
import xarray as xr
import xesmf as xe
from scipy.special import logit

POOL = "/pool/data/ERA5/E5"
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
YEARS = range(1978, 2015)
LEVELS = [1000, 850, 700, 500, 250, 100, 50]

# (varlev name, pool path, code, stream prefix, level or None, scale)
SPEC = [
    ("SPSFC", "sf/an/1M", 134, "sf00", None, 1.0),
    ("SKTSFC", "sf/an/1M", 235, "sf00", None, 1.0),
    ("VAR_2TSFC", "sf/an/1M", 167, "sf00", None, 1.0),
    ("VAR_2DSFC", "sf/an/1M", 168, "sf00", None, 1.0),
    ("VAR_10USFC", "sf/an/1M", 165, "sf00", None, 1.0),
    ("VAR_10VSFC", "sf/an/1M", 166, "sf00", None, 1.0),
    ("SSTKSFC", "sf/an/1M", 34, "sf00", None, 1.0),
    ("CISFC", "sf/an/1M", 31, "sf00", None, 1.0),
    # pool 1M TP = monthly mean of daily sums (m per day) -> mean rate in kg m-2 s-1 (= MTPR)
    ("MTPRSFC", "sf/fc/1M", 228, "sf12", None, 1000.0 / 86400.0),
    ("MSLSFC", "sf/an/1M", 151, "sf00", None, 1.0),
]
for short, code in (("t", 130), ("q", 133), ("u", 131), ("v", 132)):
    for lev in LEVELS:
        SPEC.append((f"{short}{float(lev)}", "pl/an/1M", code, "pl00", lev, 1.0))
SPEC.append(("z500.0", "pl/an/1M", 129, "pl00", 500, 1.0))
NAMES = [s[0] for s in SPEC]
MASKED = {"SSTKSFC", "CISFC"}


def edges_from_centers(c, clip=None):
    c = np.asarray(c)
    d = np.diff(c)
    e = np.concatenate(([c[0] - d[0] / 2], (c[:-1] + c[1:]) / 2, [c[-1] + d[0] / 2]))
    return np.clip(e, *clip) if clip is not None else e


def read_months(path, level):
    """Yields (yyyymm, values on the regular N320 grid, lat ascending) for one pool file."""
    with open(path, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                if level is not None and int(eccodes.codes_get(gid, "level")) != level:
                    continue
                ym = int(eccodes.codes_get(gid, "dataDate")) // 100
                pl = eccodes.codes_get_array(gid, "pl").astype(int)
                vals = eccodes.codes_get_values(gid).astype("f8")
                if eccodes.codes_get(gid, "bitmapPresent"):
                    vals[vals == eccodes.codes_get(gid, "missingValue")] = np.nan
                lats = eccodes.codes_get_array(gid, "distinctLatitudes")
            finally:
                eccodes.codes_release(gid)
            nreg = 4 * len(pl) // 2  # N320 -> 1280 points per row
            xr_ = np.arange(nreg) * 360.0 / nreg
            out = np.empty((len(pl), nreg))
            o = 0
            for j, n in enumerate(pl):
                row = vals[o:o + n]
                o += n
                x = np.arange(n) * 360.0 / n
                out[j] = np.interp(xr_, np.append(x, 360.0), np.append(row, row[0]))
            if lats[0] > lats[-1]:
                out = out[::-1]
            yield ym, out


def task(args):
    i, year = args
    name, sub, code, stream, level, scale = SPEC[i]
    path = f"{POOL}/{sub}/{code:03d}/E5{stream}_1M_{year}_{code:03d}.grb"
    res = []
    for ym, field in read_months(path, level):
        if ym < 197810:
            continue
        x = field.ravel() * scale
        # conservative_normed: divide by the covered (and, for SST/CI, valid) fraction of each cell; the N320 grid
        # ends 0.07 deg short of the poles, so the polar rows are ~99 % covered even without a mask
        m = np.isfinite(x).astype("f8") if name in MASKED else np.ones_like(x)
        num = W @ np.where(m > 0, x, 0.0)
        den = W @ m
        y = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        res.append((ym, y.reshape(121, 240)))
    return i, year, res


def build_weights():
    gid = None
    path = f"{POOL}/sf/an/1M/167/E5sf00_1M_1979_167.grb"
    with open(path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        lats = np.sort(eccodes.codes_get_array(gid, "distinctLatitudes"))
        nreg = 2 * len(eccodes.codes_get_array(gid, "pl"))
        eccodes.codes_release(gid)
    lons = np.arange(nreg) * 360.0 / nreg
    src = xr.Dataset(coords=dict(lat=("lat", lats), lon=("lon", lons),
                                 lat_b=("lat_b", edges_from_centers(lats, clip=(-90, 90))),
                                 lon_b=("lon_b", edges_from_centers(lons, clip=(0, 360)))))
    lat_out = np.linspace(-90, 90, 121)
    lon_out = np.arange(0, 360, 1.5)
    dst = xr.Dataset(coords=dict(lat=("lat", lat_out), lon=("lon", lon_out),
                                 lat_b=("lat_b", edges_from_centers(lat_out, clip=(-90, 90))),
                                 lon_b=("lon_b", edges_from_centers(lon_out, clip=(0, 360)))))
    r = xe.Regridder(src, dst, "conservative", periodic=True, ignore_degenerate=True)
    w = r.weights.data  # sparse COO, (n_dst, n_src)
    w = sp.csr_matrix((w.data, (w.coords[0], w.coords[1])), shape=w.shape)
    print("weights", w.shape, "row sums min/max", w.sum(axis=1).min(), w.sum(axis=1).max(), flush=True)
    return w, lat_out, lon_out


def stats(data, names, t0, t1, times):
    sel = (times >= t0) & (times <= t1)
    out = {}
    for k, name in enumerate(names):
        da = data[sel, k].astype("f8")
        da = np.where(np.isnan(da).any(axis=0, keepdims=True), np.nan, da)  # authors' mask: NaN at any time
        if name.startswith("q") or name == "MTPRSFC":
            da = np.sqrt(da)
        elif name == "CISFC":
            da = logit(np.clip(da, 1e-3, 1 - 1e-3))
            da = np.where(np.isnan(data[sel, k]), np.nan, da)
        out[name] = {"mu": float(np.nanmean(da)), "sigma": float(np.nanstd(da))}
    return out


if __name__ == "__main__":
    t_start = time.time()
    W, lat_out, lon_out = build_weights()
    months = [197810, 197811, 197812] + [y * 100 + m for y in range(1979, 2015) for m in range(1, 13)]
    tindex = {ym: n for n, ym in enumerate(months)}
    data = np.full((len(months), len(SPEC), 121, 240), np.nan, dtype="f8")
    jobs = [(i, y) for i in range(len(SPEC)) for y in YEARS]
    with ProcessPoolExecutor(max_workers=int(sys.argv[2])) as ex:
        for done, (i, year, res) in enumerate(ex.map(task, jobs, chunksize=1), 1):
            for ym, y in res:
                data[tindex[ym], i] = y
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} tasks, {time.time() - t_start:.0f} s", flush=True)
    missing = [(NAMES[i], months[t]) for t in range(len(months)) for i in range(len(SPEC))
               if np.isnan(data[t, i]).all()]
    print("fully missing fields:", missing[:10], len(missing), flush=True)
    times = np.array(months)
    st79 = stats(data, NAMES, 197901, 201412, times)
    st85 = stats(data, NAMES, 198501, 201412, times)
    json.dump(st79, open(f"{OUT}/variable_statistics_197901-201412_dkrz.json", "w"), indent=4)
    json.dump(st85, open(f"{OUT}/variable_statistics_198501-201412_dkrz.json", "w"), indent=4)
    tvals = np.array([np.datetime64(f"{ym // 100:04d}-{ym % 100:02d}-01") for ym in months])
    ds = xr.Dataset({"da": (("time", "varlev", "lat", "lon"), data.astype("f4"))},
                    coords={"time": tvals, "varlev": NAMES, "lat": lat_out, "lon": lon_out})
    ds.attrs["source"] = "ERA5 monthly means, DKRZ /pool/data/ERA5/E5 (N320), regridded as MDv0.9's pipeline"
    ds.to_netcdf(f"{OUT}/era5_1p5_197810-201412.nc")
    ic = xr.Dataset({"da": (("varlev", "lat", "lon"), data[0])},
                    coords={"varlev": NAMES, "lat": lat_out, "lon": lon_out})
    ic.attrs["time"] = "1978-10 monthly mean"
    ic.to_netcdf(f"{OUT}/era5_1p5_197810_f8.nc")
    # the initial state aimip pins: the 37 state channels as the authors' flat `da(time, varlev, lat, lon)`
    state = [n for n in NAMES if n not in MASKED]
    with netCDF4.Dataset(f"{OUT}/md15_era5_ic_1978-10-01_1p5.nc", "w", format="NETCDF4") as nc:
        nc.title = "MD-1.5 (MDv0.9) initial state: ERA5 monthly mean of October 1978 on the model's 1.5 degree grid"
        nc.source = ("ERA5 monthly means (DKRZ /pool/data/ERA5, N320) regridded as the authors' data pipeline (xESMF "
                     "conservative_normed, edges_from_centers); built by aimip tools/md15/make_inputs.py")
        nc.variables = ("the 37 state channels of MD-1p5 in its order; units as ERA5 (MTPRSFC = TP/86400*1000, "
                        "kg m-2 s-1; z500 geopotential m2 s-2)")
        for dim, size in (("time", 1), ("varlev", len(state)), ("lat", 121), ("lon", 240)):
            nc.createDimension(dim, size)
        t = nc.createVariable("time", "i8", ("time",))
        t.units, t.calendar = "days since 1970-01-01", "proleptic_gregorian"
        t[:] = [(np.datetime64("1978-10-01") - np.datetime64("1970-01-01")).astype(int)]
        nc.createVariable("varlev", str, ("varlev",))[:] = np.array(state, dtype=object)
        for name, vals, units in (("lat", lat_out, "degrees_north"), ("lon", lon_out, "degrees_east")):
            v = nc.createVariable(name, "f8", (name,))
            v[:] = vals
            v.units = units
        da = nc.createVariable("da", "f8", ("time", "varlev", "lat", "lon"), zlib=True, complevel=4)
        da[0] = data[0][[NAMES.index(n) for n in state]]
    json.dump(st85, open(f"{OUT}/md15_variable_statistics_198501-201412.json", "w"), indent=4)
    print(f"done in {time.time() - t_start:.0f} s", flush=True)
