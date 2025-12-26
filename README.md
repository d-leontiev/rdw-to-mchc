# Replication Package: CBC Biomarkers for ICU Mortality Prediction

This repository contains the code to replicate the analyses presented in the dissertation investigating Complete Blood Count (CBC) biomarkers, particularly the RDW/MCHC ratio, for predicting ICU mortality.

## Data Requirements

This analysis requires access to two clinical databases:

1. **MIMIC-IV v3.1** (Primary analysis)
   - Available from PhysioNet: https://physionet.org/content/mimiciv/
   - Requires credentialed access

2. **eICU Collaborative Research Database v2.0** (External validation)
   - Available from PhysioNet: https://physionet.org/content/eicu-crd/
   - Requires credentialed access

**Note:** Due to data use agreements, the raw data cannot be shared. Researchers must obtain their own access through PhysioNet.

## File Descriptions

| File | Description |
|------|-------------|
| `harmonisation.py` | Data harmonisation pipeline including unit conversion, plausibility filtering, and safe ratio computation |
| `biomarker_pipeline.ipynb` | Main analysis: systematic evaluation of CBC biomarkers for mortality and prolonged ICU stay prediction |
| `eICU_analysis.ipynb` | External validation of biomarker findings using the eICU database |
| `methods.ipynb` | Statistical validation including DeLong tests, calibration assessment, decision curve analysis, and NRI/IDI |
| `apache_vs_rdw-mchc.ipynb` | Comparison of RDW/MCHC ratio against APACHE clinical severity scores |
| `fairness_test.ipynb` | Data fairness and subgroup analysis |

## Execution Order

1. **Data Preparation**: Run `harmonisation.py` functions to preprocess CBC data
2. **Main Analysis**: Execute `biomarker_pipeline.ipynb` for primary biomarker evaluation
3. **External Validation**: Run `eICU_analysis.ipynb` to validate findings
4. **Statistical Validation**: Execute `methods.ipynb` for rigorous statistical testing
5. **Clinical Comparison**: Run `apache_vs_rdw-mchc.ipynb` to compare with clinical scores
6. **Fairness Analysis**: Execute `fairness_test.ipynb` for subgroup analyses

## Dependencies

```
python >= 3.10
pandas
numpy
scipy
scikit-learn
duckdb
matplotlib
seaborn
```

## Configuration

Before running the notebooks, update the data paths to point to your local MIMIC-IV and eICU data directories:

```python
# Example path configuration
DATA_DIR = Path("/path/to/your/mimic-iv-3.1")
EICU_DIR = Path("/path/to/your/eicu-collaborative-research-database-2.0")
```

## Key Findings

The analysis demonstrates that:
- The RDW/MCHC ratio shows robust predictive performance for ICU mortality (AUC ~0.70)
- Performance is validated across two independent ICU databases
- The biomarker provides incremental value when combined with inflammatory indices (NLR)

## Contact

For questions about this replication package, please contact the dissertation author.
