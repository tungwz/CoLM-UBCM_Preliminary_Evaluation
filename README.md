# A New 3D Urban Building Community Model for Earth System Modeling: Model Formulation and Evaluation


## Overview
This repository contains the plot scripts and model results used for the preliminary evaluation of CoLM-UBCM, supporting the manuscript:
> "A New 3D Urban Building Community Model for Earth System Modeling: Model Formulation and Evaluation" submitted to _Journal of Advances in Modeling Earth Systems_.

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
│   └── PILPS-Urban.ipynb 
└── Urban-PLUMBER2
    ├── diurnal_AU.ipynb           # diurnal cycle plot
    ├── Plumber2_AU.ipynb          # box plot
    ├── RMSE_results.csv           # Except for CoLM-UBCM, the other model results are from https://urban-plumber.github.io/AU-Preston/plots/
    └── R_results.csv              # Except for CoLM-UBCM, the other model results are from https://urban-plumber.github.io/AU-Preston/plots/
```
<br>





