# A New 3D Urban Building Community Model for Earth System Modeling: Model Formulation and Evaluation


## Overview
This repository contains the plot scripts and model results used for the preliminary evaluation of CoLM-UBCM, supporting the manuscript:
> "A New 3D Urban Building Community Model for Earth System Modeling: Model Formulation and Evaluation" submitted to _Journal of Advances in Modeling Earth Systems_. A related dataset is open access at: 10.5281/zenodo.17919049.

## Code Organization

The code is organized into three main functional directories:

1. **model_output** (`model_output/`)
   Model results from different experiment cases and corresponding observational data.

2. **PILPS-Urban** (`site_analysis/`)
   Model verification of CoLM-UBCM against the PILPS-Urban.

3. **Urban-PLUMBER2** (`global_analysis/`)
   Model verification of CoLM-UBCM against the Urban-PLUMBER2.


## Directory Structure
```bash
├── model_output
│   ├── no_irr
│   │   └── AU-Preston
│   │       ├── history                      
│   │       ├── landdata                     
│   │       └── restart                      
│   ├── obs
│   │   └── AU-Preston_clean_observations_v1.nc
│   ├── slab
│   │   ├── history
│   │   ├── landdata
│   │   └── restart
│   ├── urb
│   │   ├── history
│   │   ├── landdata
│   │   └── restart
│   └── veg
│       ├── history
│       ├── landdata
│       └── restart
├── PILPS-Urban
│   │ 
│   └── G11_AU-Pres.py 
└── Urban-PLUMBER2
    ├── diurnal_AU-Pres.py           # diurnal cycle plot
    ├── Plumber2_AU-Pres.py          # box plot
    ├── RMSE_results.csv             # Except for CoLM-UBCM, the other model results are from https://urban-plumber.github.io/AU-Preston/plots/
    └── R_results.csv                # Except for CoLM-UBCM, the other model results are from https://urban-plumber.github.io/AU-Preston/plots/
```
<br>





