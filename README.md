# Marine Diversity

Prototype pipeline for taxonomic data analysis and species‑distribution prediction using OBIS occurrence data and a Random Forest baseline. Includes scripts, notebook, trained model, and visualization artifacts.

This repository contains scripts and notebooks for processing OBIS data (Indian Ocean) and training a simple ML model.

Repository structure (created):

- data/
  - raw/ (place raw data like `obis_indian_ocean.csv`)
  - processed/ (cleaned and processed data)
- notebooks/
  - exploratory_analysis.ipynb
- scripts/
  - 01_fetch_data.py
  - 02_process_data.py
  - 03_prepare_ml_data.py
  - 04_train_evaluate_model.py
  - 05_generate_visuals.py
- outputs/
  - models/
  - plots/
  - reports/
