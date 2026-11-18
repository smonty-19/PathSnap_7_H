# `notebooks/`

**Shared folder, one notebook per owner.** Only edit **your own** notebook. Every notebook is executed top to bottom with its outputs saved. Every number comes from the code in `pathsnap/`, `models/` and `evaluation/`.

| Notebook | Owner | Content |
|---|---|---|
| [`../eda/01_eda.ipynb`](../eda/01_eda.ipynb) | **Patil** | Dataset inventory, missing values, sampling/gaps, speeds, geography, road network & connectivity, route structure, anomalies, Folium maps |
| `02_preprocessing_and_features.ipynb` | **Monty** | Cleaning, per-trip projection check, candidate generation, route-alignment labels, 21 features, dataset summary |
| `03_model_logistic_regression.ipynb` | **Patil** | Model 1: Logistic Regression |
| `04_model_knn.ipynb` | **Monty** | Model 2: distance-weighted KNN |
| `05_model_random_forest.ipynb` | **Sheetal** | Model 3: Random Forest |
| `06_model_mlp.ipynb` | **Sheetal** | Model 4: MLP (with real train/val loss curves) |
| `07_model_lightgbm.ipynb` | **Sanchita** | Model 5: LightGBM (untuned baseline) |
| `08_model_comparison_and_selection.ipynb` | **Sheetal** | Comparison plots across all 5 models + selection |
| `09_bayesian_hyperparameter_tuning.ipynb` | **Sanchita** | Optuna TPE search record, best params, tuned vs untuned |

Run order: build the datasets first (`python -m scripts.build_dataset --split train --stride 2` and `--split student_test`), then 01 → 09.
To re-execute a notebook in place: `jupyter nbconvert --to notebook --execute --inplace <notebook>`.
