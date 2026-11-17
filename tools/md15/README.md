# MD-1.5 inputs

MD-1.5's AIMIP run (`scripts/run_aimip_runs.py` of `kjhall01/monthly-diffusion`, MDv0.9) reads two inputs the
authors did not publish: the ERA5 monthly mean of October 1978 on the model's 1.5° grid (the initial state) and the
normalization statistics of the training period 1985–2014 (`variable_statistics_198501-201412.json`; the repository
ships a 1979–2014 file instead). `make_inputs.py` rebuilds both from ERA5 with the authors' data pipeline
(`data-pipeline/regrid-era5.ipynb`, `flatten_era5.ipynb`): xESMF `conservative_normed` with cell edges from
`edges_from_centers`, destination latitudes `linspace(-90, 90, 121)` and longitudes `arange(0, 360, 1.5)`, statistics
as `preprocess_by_variables` computes them.

Differences from the authors' inputs:
- Source: the DKRZ ERA5 pool (reduced Gaussian N320, made regular by linear interpolation along each row) instead of
  RDA ERA5 on 0.25°; `MTPRSFC` is the pool's total precipitation (monthly mean of daily sums) / 86400 × 1000.
- October 1978 comes from the final ERA5 back-extension, not RDA's preliminary 1950–1978 release.
- Recomputing the repository's 1979–2014 file gives means that differ by 0.02–0.03 standard deviations for most
  variables, uniformly, so that file was probably made from other data; it is not used.

The forcing (AIMIP SST, sea ice and land–sea mask, Zenodo 17065758) is regridded by the aimip job itself.

aimip pins the files made with this script: `md15_era5_ic_197810_1p5.nc` (6772776 bytes, sha256
`cb38ff2fb434735b815c72adfe6151fcdf1e406effe95b884604629c1abc03da`) and
`md15_variable_statistics_198501-201412.json` (3651 bytes, sha256
`007d471b0187fb63ee2a8e1ed0ae8486eb5761e8958bc1d1d0d2933602a367d6`).
