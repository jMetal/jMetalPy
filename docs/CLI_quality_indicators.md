# Quality Indicator CLI for jMetalPy

A command-line interface for computing quality indicators between two fronts (solution front and reference front).

## Features

This CLI tool supports the following quality indicators:

- **Additive Epsilon (epsilon)**: Measures the minimum additive factor needed to weakly dominate the reference front
- **Inverted Generational Distance (igd)**: Measures the average distance from reference points to the nearest solution
- **Inverted Generational Distance Plus (igdplus)**: IGD variant using dominance-based distance calculation
- **Hypervolume (hv)**: Volume of objective space dominated by the front
- **Normalized Hypervolume (nhv)**: Hypervolume normalized by the reference front's hypervolume
- **All indicators**: Compute all indicators at once

## Installation

The CLI is included with jMetalPy. No additional installation is required.

## Usage

### Basic Usage

```bash
python -m jmetal.util.quality_indicator_cli <front.csv> <reference.csv> <indicator> [options]
```

### Examples

#### Compute IGD between two fronts
```bash
python -m jmetal.util.quality_indicator_cli front.csv reference.csv igd
```

#### Compute all indicators with custom reference point
```bash
python -m jmetal.util.quality_indicator_cli front.csv reference.csv all --ref-point 2.0,2.0
```

#### Normalize fronts and output as JSON
```bash
python -m jmetal.util.quality_indicator_cli front.csv reference.csv all --normalize --format json
```

#### Compute epsilon indicator only
```bash
python -m jmetal.util.quality_indicator_cli front.csv reference.csv epsilon
```

### Options

- `--normalize`: Normalize both fronts using reference_only strategy
- `--ref-point V1,V2,...`: Custom reference point for HV/NHV (overrides auto-generation)
- `--format {text,json}`: Output format (default: text)
- `--margin M`: Margin added when auto-building reference point (default: 0.1)
- `-h, --help`: Show help message

## File Format

CSV files should contain numeric data with one solution per row and one objective per column.

Example `front.csv`:
```
0.0,1.0
0.2,0.8
0.4,0.6
0.6,0.4
0.8,0.2
1.0,0.0
```

Example `reference.csv`:
```
0.1,0.9
0.3,0.7
0.5,0.5
0.7,0.3
0.9,0.1
```

## Output

### Text Format (default)
```
Result (epsilon): 0.1
Result (igd): 0.1414213562373095
Result (igdplus): 0.1
Result (hv): 0.84
Result (nhv): -0.037037037037037
```

### JSON Format
```json
{
  "epsilon": 0.1,
  "igd": 0.1414213562373095,
  "igdplus": 0.1,
  "hv": 0.84,
  "nhv": -0.037037037037037
}
```

## Notes

### Reference Points
- **HV and NHV** require a reference point that is dominated by all solutions in the front
- If no reference point is provided, one is automatically generated using the maximum values of the reference front plus a margin
- For normalized data, the default reference point is `[1.1, 1.1, ...]`

### Normalization
- Uses "reference_only" strategy: normalizes both fronts based on the bounds of the reference front
- Useful when fronts have different scales or when you want to focus on relative performance

### Normalized Hypervolume (NHV)
- Calculated as: `NHV = 1 - HV(front) / HV(reference)`
- Can be negative if the solution front dominates the reference front
- Values closer to 0 indicate better performance

## Error Handling

The CLI provides informative error messages for common issues:
- File not found
- Invalid CSV format
- Dimension mismatches between fronts
- Invalid reference point format
- Missing reference points for HV/NHV indicators

## Integration with jMetalPy

This CLI tool is built on top of jMetalPy's quality indicator implementations and can be used:
- As a standalone tool for evaluating algorithm results
- In experimental pipelines and scripts
- For comparing different optimization runs
- In continuous integration systems for performance monitoring

## Inspiration

This CLI tool is inspired by the MetaJul Julia implementation, ensuring consistency and compatibility with the broader multi-objective optimization community.