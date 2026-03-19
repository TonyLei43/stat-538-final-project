# stat-538-final-project
All the code related to STAT 538 final project using Engression for causal estimation.


## File Structure

```text
STAT-538-Final/
├── notebooks/
│   ├── simulations_outcome.ipynb
│   ├── simulations_augment.ipynb
│   ├── empirical_studies.ipynb
│   └── comparison-gamma.ipynb
├── src/
│   ├── dgp.py
│   └── models.py
├── plots/
├── lalonde.csv/
├── .gitignore
├── README.md
└── requirements.txt
```

## Files

### `notebooks/`
- `simulations_outcome.ipynb`: notebook for METHOD 1 (OUTCOME): replace outcome models and comparing estimators.
- `simulations_augment.ipynb`: notebook for METHOD 2 (AUGMENTATION): augmenting weak-overlap data, and comparing estimators.
- `empirical_studies.ipynb`: notebook for IHDP and Lalonde datasets, augmenting weak-overlap data, and comparing estimators.
- `comparison-gamma.ipynb`: notebook for comparing gamma and covariate space (appendix).


### `src/`
- `dgp.py`: functions for generating the superpopulation data and creating weak overlap.
- `models.py`: estimation and training functions, including OLS, IPW, AIPW, Engression training, augmentation with Engression and other reuseable functions. 

### `plots/`
- plots and figures related to the simulations