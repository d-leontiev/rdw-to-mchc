"""
harmonisation.py - Complete CBC Data Harmonisation Pipeline
===========================================================
Single file version with all components integrated.
"""

from typing import Dict, Tuple, Optional, Union, List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# UNIT HARMONISATION
# ============================================================================

@dataclass
class UnitConversion:
    """Defines a unit conversion with bidirectional transformation."""
    from_unit: str
    to_unit: str
    forward_factor: float
    backward_factor: float

    def convert_forward(self, value: float) -> float:
        """Convert from from_unit to to_unit."""
        return value * self.forward_factor

    def convert_backward(self, value: float) -> float:
        """Convert from to_unit to from_unit."""
        return value * self.backward_factor


# Unit mapping for CBC parameters
UNIT_MAP: Dict[str, Dict[str, UnitConversion]] = {
    'hb': {
        'g/dL_to_g/L': UnitConversion('g/dL', 'g/L', 10.0, 0.1),
    },
    'hct': {
        'percent_to_fraction': UnitConversion('%', 'fraction', 0.01, 100.0),
    },
    'mchc': {
        'g/dL_to_g/L': UnitConversion('g/dL', 'g/L', 10.0, 0.1),
    },
}

# Standard units for harmonized output
STANDARD_UNITS = {
    'hb': 'g/dL',
    'hct': '%',
    'rbc': '10^12/L',
    'mcv': 'fL',
    'mch': 'pg',
    'mchc': 'g/dL',
    'rdw': '%',
    'wbc': '10^9/L',
    'neutrophils': '10^9/L',
    'lymphocytes': '10^9/L',
    'monocytes': '10^9/L',
    'eosinophils': '10^9/L',
    'basophils': '10^9/L',
    'platelets': '10^9/L',
}


def detect_unit(values: pd.Series, param: str) -> str:
    """Detect the unit of a parameter based on value ranges."""
    if values.isna().all():
        return STANDARD_UNITS.get(param, 'unknown')

    median_val = values.dropna().median()

    if param == 'hb':
        return 'g/L' if median_val > 50 else 'g/dL'
    elif param == 'hct':
        return 'fraction' if median_val < 1 else '%'
    elif param == 'mchc':
        return 'g/L' if median_val > 100 else 'g/dL'
    else:
        return STANDARD_UNITS.get(param, 'unknown')


def harmonise_units(df: pd.DataFrame,
                   unit_hints: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Harmonize units across all CBC parameters to standard units."""
    df_harmonized = df.copy()
    conversions_applied = {}

    for param in STANDARD_UNITS.keys():
        if param not in df.columns:
            continue

        # Detect current unit if not provided
        if unit_hints and param in unit_hints:
            current_unit = unit_hints[param]
        else:
            current_unit = detect_unit(df[param], param)

        target_unit = STANDARD_UNITS[param]

        # Apply conversion if needed
        if current_unit != target_unit:
            conversion_key = f"{current_unit.replace('/', '_')}_to_{target_unit.replace('/', '_')}"

            if param in UNIT_MAP and conversion_key in UNIT_MAP[param]:
                conversion = UNIT_MAP[param][conversion_key]
                df_harmonized[param] = df[param].apply(
                    lambda x: conversion.convert_forward(x) if pd.notna(x) else np.nan
                )
                conversions_applied[param] = f"{current_unit} → {target_unit}"
                logger.info(f"Converted {param} from {current_unit} to {target_unit}")

    return df_harmonized, conversions_applied


# ============================================================================
# PLAUSIBILITY FILTERING
# ============================================================================

# Biological plausibility bounds for adult ICU patients
PLAUSIBILITY_BOUNDS = {
    'hb': [3.0, 20.0],        # g/dL
    'hct': [10.0, 65.0],      # %
    'rbc': [1.5, 8.0],        # 10^12/L
    'mcv': [60.0, 120.0],     # fL
    'mch': [20.0, 40.0],      # pg
    'mchc': [25.0, 38.0],     # g/dL
    'rdw': [10.0, 30.0],      # %
    'wbc': [0.1, 100.0],      # 10^9/L
    'neutrophils': [0.0, 90.0],  # 10^9/L
    'lymphocytes': [0.0, 50.0],   # 10^9/L
    'monocytes': [0.0, 10.0],     # 10^9/L
    'eosinophils': [0.0, 5.0],   # 10^9/L
    'basophils': [0.0, 2.0],     # 10^9/L
    'platelets': [5.0, 1500.0],  # 10^9/L
}


def apply_plausibility(df: pd.DataFrame,
                      bounds: Dict[str, Tuple[float, float]] = None,
                      action: str = 'exclude') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply plausibility filters to exclude biologically implausible values."""
    if bounds is None:
        bounds = PLAUSIBILITY_BOUNDS

    df_filtered = df.copy()
    exclusion_report = []

    for param, (min_val, max_val) in bounds.items():
        if param not in df.columns:
            continue

        # Find implausible values
        mask_low = df[param] < min_val
        mask_high = df[param] > max_val
        mask_implausible = mask_low | mask_high

        n_low = mask_low.sum()
        n_high = mask_high.sum()
        n_total = mask_implausible.sum()

        if n_total > 0:
            exclusion_report.append({
                'parameter': param,
                'n_below_min': n_low,
                'n_above_max': n_high,
                'n_excluded': n_total,
                'pct_excluded': 100 * n_total / len(df),
                'min_bound': min_val,
                'max_bound': max_val
            })

            if action == 'exclude':
                df_filtered = df_filtered[~mask_implausible]
            elif action == 'nullify':
                df_filtered.loc[mask_implausible, param] = np.nan

    report_df = pd.DataFrame(exclusion_report)
    logger.info(f"Plausibility filtering: {len(df)} → {len(df_filtered)} records")

    return df_filtered, report_df


# ============================================================================
# SAFE RATIO COMPUTATION
# ============================================================================

# Minimum denominator values to avoid numerical instability
RATIO_FLOORS = {
    'lymphocytes': 0.1,    # 10^9/L
    'neutrophils': 0.1,    # 10^9/L
    'monocytes': 0.01,     # 10^9/L
    'mchc': 20.0,          # g/dL
    'mch': 15.0,           # pg
    'rbc': 1.0,            # 10^12/L
    'hct': 10.0,           # %
    'wbc': 0.1,            # 10^9/L
    'platelets': 10.0,     # 10^9/L
}

def _resolve_conversion(param: str, current_unit: str, target_unit: str) -> Optional[UnitConversion]:
    """
    Find a UnitConversion object for (current -> target) or the reverse (target -> current).
    Returns the mapping object; caller decides whether to use forward or backward factor.
    """
    if param not in UNIT_MAP:
        return None
    for conv in UNIT_MAP[param].values():
        if conv.from_unit == current_unit and conv.to_unit == target_unit:
            return conv  # use forward_factor
        if conv.from_unit == target_unit and conv.to_unit == current_unit:
            return conv  # use backward_factor
    return None

def harmonise_units(df: pd.DataFrame,
                    unit_hints: Optional[Dict[str, str]] = None
                    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Harmonise units across CBC parameters to STANDARD_UNITS, using bi-directional conversions."""
    df_harmonized = df.copy()
    conversions_applied: Dict[str, str] = {}

    for param, target_unit in STANDARD_UNITS.items():
        if param not in df.columns:
            continue

        # Detect or use hint
        current_unit = unit_hints.get(param) if unit_hints and param in unit_hints else detect_unit(df[param], param)

        if current_unit == target_unit:
            continue

        conv = _resolve_conversion(param, current_unit, target_unit)
        if conv is None:
            logger.warning(f"No conversion rule for {param}: {current_unit} -> {target_unit}. Leaving as-is.")
            continue

        # Choose direction
        if conv.from_unit == current_unit and conv.to_unit == target_unit:
            factor = conv.forward_factor
        else:  # reverse direction
            factor = conv.backward_factor

        df_harmonized[param] = df[param].apply(lambda x: x * factor if pd.notna(x) else np.nan)
        conversions_applied[param] = f"{current_unit} → {target_unit}"
        logger.info(f"Converted {param} from {current_unit} to {target_unit}")

    return df_harmonized, conversions_applied
def compute_safe_ratio(numerator: pd.Series,
                       denominator: pd.Series,
                       denominator_floor: float = 0.0,
                       ratio_name: str = None) -> pd.Series:
    """Compute ratio safely without ad-hoc epsilons."""
    # Initialize with NaN
    ratio = pd.Series(index=numerator.index, dtype=float)

    # Valid computation mask
    valid_mask = (
        pd.notna(numerator) &
        pd.notna(denominator) &
        (denominator >= denominator_floor)
    )

    # Compute ratio only for valid entries
    if valid_mask.any():
        ratio[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    n_invalid = (~valid_mask).sum()
    if ratio_name and n_invalid > 0:
        logger.debug(f"{ratio_name}: {n_invalid} invalid ratios (denominator < {denominator_floor})")

    return ratio


def compute_cbc_ratios(df: pd.DataFrame,
                       ratio_definitions: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
    """Compute standard CBC ratios with safe denominators."""
    df_with_ratios = df.copy()

    # Default ratio definitions
    if ratio_definitions is None:
        ratio_definitions = {
            'nlr': ('neutrophils', 'lymphocytes'),
            'plr': ('platelets', 'lymphocytes'),
            'mlr': ('monocytes', 'lymphocytes'),
            'rdw_to_mchc': ('rdw', 'mchc'),
            'rdw_to_mch': ('rdw', 'mch'),
            'mcv_to_mchc': ('mcv', 'mchc'),
            'rdw_to_rbc': ('rdw', 'rbc'),
        }

    for ratio_name, (num_param, denom_param) in ratio_definitions.items():
        if num_param in df.columns and denom_param in df.columns:
            floor = RATIO_FLOORS.get(denom_param, 0.0)
            df_with_ratios[ratio_name] = compute_safe_ratio(
                df[num_param],
                df[denom_param],
                denominator_floor=floor,
                ratio_name=ratio_name
            )

    return df_with_ratios


# ============================================================================
# TRANSFORMATIONS
# ============================================================================

def winsorize_features(df: pd.DataFrame,
                       features: list,
                       lower_percentile: float = 1.0,
                       upper_percentile: float = 99.0) -> pd.DataFrame:
    """Winsorize continuous features at specified percentiles."""
    df_winsorized = df.copy()

    for feature in features:
        if feature not in df.columns:
            continue

        lower = df[feature].quantile(lower_percentile / 100)
        upper = df[feature].quantile(upper_percentile / 100)

        df_winsorized[feature] = df[feature].clip(lower=lower, upper=upper)

        n_clipped = ((df[feature] < lower) | (df[feature] > upper)).sum()
        if n_clipped > 0:
            logger.debug(f"Winsorized {feature}: {n_clipped} values clipped to [{lower:.2f}, {upper:.2f}]")

    return df_winsorized


def log_transform_features(df: pd.DataFrame,
                          features: list,
                          offset: float = 1.0) -> pd.DataFrame:
    """Apply log transformation to strictly positive features."""
    df_transformed = df.copy()

    for feature in features:
        if feature not in df.columns:
            continue

        # Only transform if all values are positive (after adding offset)
        if (df[feature].dropna() + offset > 0).all():
            df_transformed[f'log_{feature}'] = np.log(df[feature] + offset)
            logger.debug(f"Log-transformed {feature} (offset={offset})")
        else:
            logger.warning(f"Cannot log-transform {feature}: contains non-positive values")

    return df_transformed


# ============================================================================
# QC REPORT GENERATION
# ============================================================================

def generate_qc_report(df_original: pd.DataFrame,
                       df_processed: pd.DataFrame,
                       conversions: Dict[str, str],
                       exclusion_report: pd.DataFrame,
                       output_dir: str = './outputs/qc') -> None:
    """Generate comprehensive QC report in Markdown and CSV formats."""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    n_original = len(df_original)
    n_processed = len(df_processed)
    n_excluded = n_original - n_processed
    pct_excluded = 100 * n_excluded / n_original if n_original > 0 else 0

    # Generate Markdown report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    markdown_content = f"""# CBC Data Harmonization QC Report
Generated: {timestamp}

## Summary Statistics
- Original records: {n_original:,}
- Processed records: {n_processed:,}
- Excluded records: {n_excluded:,} ({pct_excluded:.1f}%)

## Unit Conversions Applied
"""

    if conversions:
        for param, conversion in conversions.items():
            markdown_content += f"- **{param}**: {conversion}\n"
    else:
        markdown_content += "No unit conversions required.\n"

    markdown_content += "\n## Plausibility Filtering\n"

    if not exclusion_report.empty:
        markdown_content += """
| Parameter | Below Min | Above Max | Total Excluded | % Excluded | Bounds |
|-----------|-----------|-----------|----------------|------------|--------|
"""
        for _, row in exclusion_report.iterrows():
            markdown_content += (
                f"| {row['parameter']} | {row['n_below_min']} | {row['n_above_max']} | "
                f"{row['n_excluded']} | {row['pct_excluded']:.1f}% | "
                f"[{row['min_bound']}, {row['max_bound']}] |\n"
            )
    else:
        markdown_content += "No values excluded by plausibility filtering.\n"

    # Save reports
    md_path = os.path.join(output_dir, 'unit_harmonisation_report.md')
    with open(md_path, 'w') as f:
        f.write(markdown_content)

    # Generate CSV metrics
    metrics = {
        'metric': ['n_original', 'n_processed', 'n_excluded', 'pct_excluded'],
        'value': [n_original, n_processed, n_excluded, pct_excluded]
    }

    metrics_df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, 'unit_harmonisation_metrics.csv')
    metrics_df.to_csv(csv_path, index=False)

    logger.info(f"QC reports saved to {output_dir}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_cbc_data(df: pd.DataFrame,
                    unit_hints: Optional[Dict[str, str]] = None,
                    apply_winsorization: bool = True,
                    apply_log_transform: bool = False,
                    generate_report: bool = True) -> pd.DataFrame:
    """
    Main pipeline for processing CBC data.

    Args:
        df: Input DataFrame with raw CBC data
        unit_hints: Optional hints about current units
        apply_winsorization: Whether to winsorize at 1st/99th percentiles
        apply_log_transform: Whether to apply log transformation
        generate_report: Whether to generate QC report

    Returns:
        Processed DataFrame
    """
    df_original = df.copy()

    # Step 1: Harmonize units
    df_harmonized, conversions = harmonise_units(df, unit_hints)

    # Step 2: Apply plausibility filters
    df_filtered, exclusion_report = apply_plausibility(df_harmonized)

    # Step 3: Compute ratios
    df_with_ratios = compute_cbc_ratios(df_filtered)

    # Step 4: Optional transformations
    if apply_winsorization:
        continuous_features = [col for col in df_with_ratios.columns
                             if col not in ['stay_id', 'hadm_id', 'subject_id']]
        df_with_ratios = winsorize_features(df_with_ratios, continuous_features)

    if apply_log_transform:
        positive_features = ['wbc', 'neutrophils', 'lymphocytes', 'platelets']
        df_with_ratios = log_transform_features(df_with_ratios, positive_features)

    # Step 5: Generate QC report
    if generate_report:
        generate_qc_report(df_original, df_with_ratios, conversions, exclusion_report)

    return df_with_ratios


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n = 100

    sample_data = pd.DataFrame({
        'stay_id': range(1, n + 1),
        'hb': np.random.normal(12.0, 2.0, n),
        'hct': np.random.normal(36.0, 6.0, n),
        'rbc': np.random.normal(4.5, 0.8, n),
        'mcv': np.random.normal(90.0, 8.0, n),
        'mch': np.random.normal(30.0, 3.0, n),
        'mchc': np.random.normal(33.0, 2.0, n),
        'rdw': np.random.normal(14.0, 2.0, n),
        'wbc': np.random.normal(8.0, 3.0, n),
        'neutrophils': np.random.normal(5.0, 2.0, n),
        'lymphocytes': np.random.normal(2.0, 0.8, n),
        'monocytes': np.random.normal(0.5, 0.2, n),
        'platelets': np.random.normal(250.0, 80.0, n)
    })

    # Add some extreme values
    sample_data.loc[0, 'hb'] = 25.0  # Implausible
    sample_data.loc[1, 'lymphocytes'] = 0.05  # Below floor

    print("Processing sample CBC data...")
    processed_data = process_cbc_data(
        sample_data,
        apply_winsorization=True,
        generate_report=True
    )

    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"Processed shape: {processed_data.shape}")
    print(f"\nNew columns added: {[col for col in processed_data.columns if col not in sample_data.columns]}")
    print("\nProcessing complete!")

    import pandas as pd
    import numpy as np
    from harmonisation import harmonise_units, apply_plausibility, STANDARD_UNITS


    def test_bidirectional_unit_conversion():
        df = pd.DataFrame({'hb': [125.0, 150.0]})  # g/L
        out, conv = harmonise_units(df, unit_hints={'hb': 'g/L'})
        assert out['hb'].round(1).tolist() == [12.5, 15.0]
        assert conv['hb'] == 'g/L → g/dL'


    def test_plausibility_exclude_no_warning():
        df = pd.DataFrame({'hb': [5.0, 30.0, 10.0]})  # 30 is implausible
        # Should exclude the 30.0 row
        out, rep = apply_plausibility(df, bounds={'hb': (3.0, 20.0)}, action='exclude')
        assert len(out) == 2
        assert rep.loc[rep['parameter'] == 'hb', 'n_excluded'].item() == 1


    def test_plausibility_nullify_cells():
        df = pd.DataFrame({'hb': [2.0, 10.0, 25.0]})
        out, rep = apply_plausibility(df, bounds={'hb': (3.0, 20.0)}, action='nullify')
        assert pd.isna(out.loc[0, 'hb']) and pd.isna(out.loc[2, 'hb'])
        assert out.loc[1, 'hb'] == 10.0