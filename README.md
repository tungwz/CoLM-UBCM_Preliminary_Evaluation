# CoLM-UBCM Preliminary Evaluation

Preliminary evaluation material for the Common Land Model with Urban Building Community Model (CoLM-UBCM). The repository contains selected model output, quality-controlled site observations, and Jupyter notebooks used to compare CoLM-UBCM configurations with flux-tower observations and Urban-PLUMBER2/PILPS-Urban benchmarks.

## What Is Included

- CoLM-UBCM history, restart, and surface datasets for point-site experiments.
- Cleaned observation files for `AU-Preston` and `FR-Capitole`.
- Analysis notebooks for diurnal composites, Urban-PLUMBER2 model comparison, seasonal FR-Capitole diagnostics, and PILPS-Urban-style RMSE comparison.
- Precomputed Urban-PLUMBER2 summary metrics in CSV format.

The repository is data-heavy because it includes NetCDF model output.

## Repository Structure

```text
.
├── PILPS-Urban/
│   └── PILPS-Urban.ipynb
├── Urban-PLUMBER2/
│   ├── Plumber2_AU.ipynb
│   ├── Seasonal_plot_FR_Capitole.ipynb
│   ├── diurnal_AU.ipynb
│   ├── RMSE_results.csv
│   └── R_results.csv
├── model_output/
│   ├── obs/
│   │   ├── AU-Preston_clean_observations_v1.nc
│   │   └── FR-Capitole_clean_observations_v1.nc
│   ├── veg/AU-Preston/
│   ├── no_irr/AU-Preston/
│   ├── no_irr/FR-Capitole/
│   ├── no_ah/FR-Capitole/
│   ├── urb/AU-Preston/
│   └── slab/AU-Preston/
└── README.md
```

Each model experiment directory generally contains:

- `history/`: monthly model output files named `{site}_hist_{YYYY-MM}.nc`
- `restart/`: restart files used by the model run
- `landdata/`: site surface data such as `srfdata.nc`
- `Site_*.nml`: namelist used for the point-site experiment

## Sites And Experiments

| Site | Available experiments | History files | Notes |
| --- | --- | ---: | --- |
| `AU-Preston` | `veg`, `no_irr`, `urb`, `slab` | 16 each | Main site for diurnal, Urban-PLUMBER2, and PILPS-Urban comparisons |
| `FR-Capitole` | `no_irr`, `no_ah` | 13 each | Seasonal diagnostics; some notebook paths may need local path edits for full comparison |

## Model Configurations

| Directory | Short label | Description |
| --- | --- | --- |
| `model_output/veg` | `Urb_Veg` | CoLM-UBCM with urban vegetation, water, building energy model, and anthropogenic heat options enabled where available |
| `model_output/no_irr` | `Urb_Veg (No Irr)` | Urban vegetation sensitivity experiment with irrigation disabled |
| `model_output/no_ah` | `Urb_Veg (No AHF)` | FR-Capitole sensitivity experiment without anthropogenic heat flux |
| `model_output/urb` | `Urb` | Urban configuration without urban tree component |
| `model_output/slab` | `Slab` | Traditional slab urban parameterization |

The exact switches are recorded in the corresponding `Site_*.nml` files.

## Evaluated Variables

| Diagnostic | Observation variable | CoLM variable | Unit |
| --- | --- | --- | --- |
| `SWup` | upward shortwave radiation | `f_sr` | W m-2 |
| `LWup` | upward longwave radiation | `f_olrg` | W m-2 |
| `Rnet` | net radiation | `f_rnet` | W m-2 |
| `Qh` | sensible heat flux | `f_fsena` | W m-2 |
| `Qle` | latent heat flux | `f_lfevpa` | W m-2 |
| `Qg` | ground/storage heat flux | `f_fgrnd` | W m-2 |

Additional forcing/radiation fields used in the notebooks include `f_xy_solarin` and `f_xy_frl`.

## Analysis Notebooks

### `Urban-PLUMBER2/diurnal_AU.ipynb`

Builds 30-minute diurnal composites for `AU-Preston` and compares `Urb_Veg`, `Urb_Veg (No Irr)`, `Urb`, and `Slab` against observations. It calculates RMSE for the main radiation and turbulent/storage heat flux variables and saves `Figure13.pdf` when run.

### `Urban-PLUMBER2/Plumber2_AU.ipynb`

Compares CoLM-UBCM against Urban-PLUMBER2 model results using the precomputed `R_results.csv` and `RMSE_results.csv` tables. The notebook visualizes CoLM-UBCM relative to the multi-model UCM ensemble for `SWup`, `LWup`, `Qh`, and `Qle`.

### `Urban-PLUMBER2/Seasonal_plot_FR_Capitole.ipynb`

Calculates seasonal `R`, `RMSE`, and `MBE` metrics for `FR-Capitole` and writes `Figure15.jpg` and `Figure15.pdf`. The checked-in data include `no_irr` and `no_ah` FR-Capitole output; the notebook currently also references absolute paths for `slab` and `veg` FR-Capitole runs, so edit `MODEL_SPECS` if those files are stored elsewhere.

### `PILPS-Urban/PILPS-Urban.ipynb`

Runs a PILPS-Urban-style comparison for `AU-Preston` over 2003-11-28 to 2004-11-28. It computes `R`, `RMSE`, `MBE`, model standard deviation, and observation standard deviation for the four local CoLM configurations, then compares RMSE against manually entered `G11_Best` values. It saves `Figure12.pdf` when run.

## Urban-PLUMBER2 Summary Metrics

The CSV files in `Urban-PLUMBER2/` contain benchmark metrics for CoLM-UBCM (`CoLM-U`) and 18 Urban-PLUMBER2 UCMs.

| Variable | CoLM-U R | CoLM-U RMSE (W m-2) |
| --- | ---: | ---: |
| `SWup` | 0.9971 | 3.6265 |
| `LWup` | 0.9931 | 6.3806 |
| `Qh` | 0.9458 | 33.2661 |
| `Qle` | 0.6615 | 37.9636 |

## Environment

Recommended Python packages:

```text
python >= 3.8
jupyter
numpy
pandas
xarray
netCDF4
matplotlib
seaborn
```

Optional but useful:

```text
dask
h5netcdf
```

## Usage

Clone the repository:

```bash
git clone git@github.com:tungwz/CoLM-UBCM_Preliminary_Evaluation.git
cd CoLM-UBCM_Preliminary_Evaluation
```

Create an environment and install dependencies, for example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas xarray netCDF4 matplotlib seaborn
```

Run the notebooks from their own directories so relative paths resolve correctly.

```bash
cd Urban-PLUMBER2
jupyter notebook diurnal_AU.ipynb
jupyter notebook Plumber2_AU.ipynb
jupyter notebook Seasonal_plot_FR_Capitole.ipynb
```

From the repository root:

```bash
cd PILPS-Urban
jupyter notebook PILPS-Urban.ipynb
```

## Notes For Reuse

- Notebook outputs such as `Figure12.pdf`, `Figure13.pdf`, and `Figure15.*` are generated products and are not currently tracked in this repository.
- The notebooks assume a fixed set of variable names in the CoLM history files and observation files. If newer CoLM output uses different names, update the variable mapping blocks first.
- Some namelists contain machine-specific absolute paths from the original run environment. The checked-in NetCDF output can still be analyzed locally, but rerunning CoLM itself requires updating those paths.

## References

- Urban-PLUMBER2: https://urban-plumber.github.io/
- PILPS-Urban: Grimmond, C. S. B., Blackett, M., Best, M. J., Baik, J.-J., Belcher, S. E., Beringer, J., Bohnenstengel, S. I., Calmet, I., Chen, F., Coutts, A., Dandou, A., Fortuniak, K., Gouvea, M. L., Hamdi, R., Hendry, M., Kanda, M., Kawai, T., Kawamoto, Y., Kondo, H., ... Zhang, N. (2011). Initial results from Phase 2 of the international urban energy balance model comparison. International Journal of Climatology, 31(2), 244-272. https://doi.org/10.1002/joc.2227
