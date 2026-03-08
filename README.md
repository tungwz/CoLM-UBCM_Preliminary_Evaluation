# CoLM-UBCM Preliminary Evaluation

Preliminary evaluation of the Common Land Model with Urban Building Community Model (CoLM-UBCM) against flux tower observations and intercomparison with other Urban Canopy Models (UCMs) from the Urban-PLUMBER2 project.

## Repository Structure

```
.
├── PILPS-Urban/
│   └── PILPS-Urban.ipynb          # PILPS-Urban model intercomparison analysis
├── Urban-PLUMBER2/
│   ├── diurnal_AU.ipynb           # Diurnal cycle composite analysis
│   ├── Plumber2_AU.ipynb          # Urban-PLUMBER2 benchmark comparison
│   ├── RMSE_results.csv           # RMSE statistics for all models
│   └── R_results.csv              # Correlation coefficients for all models
├── model_output/
│   ├── veg/AU-Preston/history/    # Urb_Veg: urban vegetation with irrigation
│   ├── no_irr/AU-Preston/history/ # Urb_Veg (NoIrr): urban vegetation, no irrigation
│   ├── urb/AU-Preston/history/    # Urb: pure urban (no vegetation)
│   ├── slab/AU-Preston/history/   # Slab: traditional slab model
│   └── obs/                       # Quality-controlled observations
└── README.md
```

## Model Configurations

Four CoLM-UBCM configurations are evaluated to understand the role of urban vegetation and irrigation:

| Short Name | Full Name | Description |
|------------|-----------|-------------|
| **Urb_Veg** | CoLM-UBCM with irrigation | Full urban building community model with urban vegetation and active irrigation scheme |
| **Urb_Veg (NoIrr)** | CoLM-UBCM without irrigation | Same as Urb_Veg but with irrigation disabled |
| **Urb** | CoLM-UBCM bare urban | Urban building community without any vegetation component |
| **Slab** | Slab model | Traditional bulk urban parameterization using slab approach |

## Site Information

### AU-Preston (Melbourne, Australia)

| Property | Value |
|----------|-------|
| Location | Melbourne, Australia |
| Latitude | 37.73°S |
| Longitude | 145.0°E |
| UTC Offset | +10 hours |
| Analysis Period | August 2003 – November 2004 |
| Time Step | 30 minutes |
| Land Cover | Suburban residential |

### Evaluated Variables

| Variable | Description | Unit | Model Variable |
|----------|-------------|------|----------------|
| SWup | Upward shortwave radiation | W m⁻² | `f_sr` |
| LWup | Upward longwave radiation | W m⁻² | `f_olrg` |
| Rnet | Net radiation | W m⁻² | `f_rnet` |
| Qh | Sensible heat flux | W m⁻² | `f_fsena` |
| Qle | Latent heat flux | W m⁻² | `f_lfevpa` |
| Qg | Ground/storage heat flux | W m⁻² | `f_fgrnd` |

## Model Output Data

### File Naming Convention
```
{site}_hist_{YYYY-MM}.nc
```

Example: `AU-Preston_hist_2003-08.nc`

### Key Variables in Output Files

| Variable | Long Name | Unit |
|----------|-----------|------|
| `f_sr` | Reflected solar radiation at surface | W m⁻² |
| `f_olrg` | Outgoing long-wave radiation | W m⁻² |
| `f_rnet` | Net radiation | W m⁻² |
| `f_fsena` | Sensible heat from canopy to atmosphere | W m⁻² |
| `f_lfevpa` | Latent heat flux from canopy to atmosphere | W m⁻² |
| `f_fgrnd` | Ground heat flux | W m⁻² |
| `f_xy_solarin` | Downward solar radiation at surface | W m⁻² |
| `f_xy_frl` | Atmospheric longwave radiation | W m⁻² |

## Analysis Notebooks

### 1. diurnal_AU.ipynb

**Purpose:** Generates diurnal cycle composites comparing all four model configurations against observations.

**Outputs:**
- 6-panel figure showing diurnal cycles for SWup, LWup, Rnet, Qh, Qle, Qg
- RMSE values for each model configuration
- PDF file: `Figure13.pdf`

**Methodology:**
- Groups all 30-minute data by time-of-day to create diurnal composites (48 points per day)
- Calculates mean diurnal cycle across the entire analysis period
- Computes RMSE for each configuration against observations

### 2. Plumber2_AU.ipynb

**Purpose:** Compares CoLM-UBCM (Urb_Veg) performance against 18 UCMs from Urban-PLUMBER2.

**Models Compared:**
ASLUMv2, ASLUMv3.1, BEPCOL, CLM5U, CM-BEM, CM, K-UCMv1, MUSE, NOAH-SLUCM, SNUUCM, TARGET, TEB-READING, TEB-CNRM, TEB-SPARTCS, UCLEM, UT&C, VTUF-3D, VUCM

**Metrics:**
- Correlation coefficient (R): measures temporal pattern agreement
- Root Mean Square Error (RMSE): measures absolute error magnitude

**Outputs:**
- Box plots showing CoLM-UBCM vs. UCM ensemble distribution
- PDF file: `Figure13.pdf`

### 3. PILPS-Urban.ipynb

**Purpose:** Analysis following PILPS-Urban experimental protocol.

## Performance Summary (AU-Preston)

### CoLM-UBCM (Urb_Veg) vs Observations

| Variable | Correlation (R) | RMSE (W m⁻²) |
|----------|-----------------|--------------|
| SWup | 0.9971 | 3.63 |
| LWup | 0.9931 | 6.38 |
| Qh | 0.9458 | 33.27 |
| Qle | 0.6615 | 37.96 |

### Key Findings

1. **Radiation fluxes**: Excellent agreement (R > 0.99) with low RMSE for both SWup and LWup
2. **Sensible heat**: Good performance (R = 0.95) with moderate RMSE
3. **Latent heat**: Lower correlation reflects challenges in simulating urban evapotranspiration timing and magnitude
4. **Urban vegetation impact**: Comparison across configurations shows the importance of vegetation for Qle simulation

## Requirements

```
Python >= 3.8
xarray
numpy
pandas
matplotlib
seaborn
netCDF4
jupyter
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/your-username/CoLM-UBCM_Preliminary_Evaluation.git
cd CoLM-UBCM_Preliminary_Evaluation
```

2. Run notebooks:
```bash
cd Urban-PLUMBER2
jupyter notebook diurnal_AU.ipynb
jupyter notebook Plumber2_AU.ipynb
```

## References

- Urban-PLUMBER2: [Project Website](https://urban-plumber2.org/)
- PILPS-Urban: Project for Intercomparison of Land-surface Parameterization Schemes - Urban

## License

MIT License
