# AutoML-DomainKnowledge

# Thesis: Enhancing AutoML Performance through Domain-Specific Knowledge

This repository contains the code, scripts, and datasets used in thesis: **"Evaluating The Impact of Integrating Domain-Specific Assumptions Into Auto-sklearn"**. The project aims to explore whether the integration of domain-specific knowledge into Auto-sklearn can improve efficiency and performance.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Directory Structure](#directory-structure)
- [Usage](#usage)
- [Results and Visualizations](#results-and-visualizations)
- [Contact](#contact)

---

## Overview
AutoML systems frequently neglect the potential advantages of including domain-specific knowledge in the learning process. This project explore the impact of incorporating the premise about non-linear separability into Auto-sklearn. Key objectives include:
- Pruning the search space of models to favor non-linear algorithms.
- Comparing efficiency and performance metrics (accuracy, F1-score, and precision) of the modified and standard Auto-sklearn systems.

---

## Requirements
All required Python package can be found in `requirements.txt` and can be installed using the following command:

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- Python 3.9.20
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `scipy`
- `tqdm`, `openml`
- `auto-sklearn`

---

## Directory Structure
```
.
├── ExcludeDatasets.csv                # List of datasets to exclude (see section 4.2.1)
├── LinearDatasets.csv                 # Results of linear separability test
├── LoadOpenML_CC18.py                 # Script to load benchmark dataset OpenML-CC18 from OpenML 
├── TestLinearSeparability.py          # Script to test datasets for linear separability 
├── TrainSystems.py                    # Script to train modified and standard Auto-sklearn systems
├── Visualize.py                       # Script for result visualization and analysis
├── ThesisAlgoInfoClassifiers.csv      # All classifiers with information (linear/non-linear)
├── requirements.txt                   # Python dependencies
```

---

## Usage

### Step 1: Load OpenML Datasets
Use `LoadOpenML_CC18.py` to download all datasets from the OpenML-CC18 benchmark.

```bash
python LoadOpenML_CC18.py
```

This script:
- Downloads datasets from a benchmark suite (OpenML-CC18).
- Saves each datasets features and target labels as CSV files in a separate directory.

---

### Step 2: Test Linear Separability
Run `TestLinearSeparability.py` to determine which of the datasets out of the benchmark are non-linearly separable:

```bash
python TestLinearSeparability.py
```

Output:
- `LinearDatasets.csv`: A file that indicates which datasets are and are not linearly separable.

---

### Step 3: Train Systems
To train the modified and standard Auto-sklearn systems, run `TrainSystems.py`:

```bash
python TrainSystems.py
```

This script:
- Trains both systems on each non-linearly separable datasets.
- Measures runtime and performance metrics.
- Saves results and convergence plots in the dataset directories.

---

### Step 4: Visualize Results
Run `Visualize.py` to generate visualizations comparing the performance of the two systems:

```bash
python Visualize.py
```

Key outputs:
- Histograms of log-scaled time proptions.
- Scatter plots for accuracy, F1-score, and precision (Modified vs Standard).
- Distribution plots of performance differences.

---

## Results and Visualizations
Key findings from this project:
- **Efficiency**: Incorporating assumptions like non-linear separability significantly improves efficiency.
- **Performance**: in our experiment we found no statistically significant effect on performance
metrics accuracy, f1-score and precision.

Visual outputs include:
- Log-scaled time ratio histograms.
- Scatter plots of accuracy and F1-score comparisons.
- Distribution plots for performance metrics.

---

## Contact
For any questions or further collaboration, please contact:  
**Tim Haasdijk**  
**Email**: [t.p.haasdijk@gmail.com]
