# scintegration

**Integration Score Calculator for Single-Cell Batch Correction Evaluation**

`scintegration` is a Python package that provides a comprehensive metric for evaluating batch correction methods in single-cell RNA-seq data. It calculates an **integration score** that balances biology preservation against batch effect removal, helping researchers choose the best batch correction approach for their data.

## Key Features

- **Single Metric**: One integration score that captures the tradeoff between preserving biological signal and removing batch effects
- **cz-benchmarks Integration**: Works seamlessly with [cz-benchmarks](https://github.com/chanzuckerberg/czbenchmarks) clustering and classification tasks
- **Multiple Metrics**: Combines ARI, NMI, Silhouette, and F1 scores for robust evaluation
- **Interpretable**: Scores range from -0.5 (poor) to +0.5 (excellent) with clear interpretation
- **Class Imbalance Aware**: Uses F1 score (macro-averaged) to handle datasets with rare cell types

## Installation

### Basic Installation

```bash
pip install scintegration
```

### With Visualization Tools

```bash
pip install scintegration[visualization]
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/SkylarPurks/scintegration.git
cd scintegration

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

## Quick Start

```python
from scintegration import IntegrationScoreEvaluator
from czbenchmarks.tasks import ClusteringTask, MetadataLabelPredictionTask

# Step 1: Run czbenchmarks tasks (clustering + classification)
# For biological labels (e.g., cell types)
clustering_task_bio = ClusteringTask()
clustering_results_bio = clustering_task_bio.evaluate(
    embeddings={'scvi': scvi_embeddings, 'scgpt': scgpt_embeddings},
    labels=celltype_labels
)

classification_task_bio = MetadataLabelPredictionTask()
classification_results_bio = classification_task_bio.evaluate(
    embeddings={'scvi': scvi_embeddings, 'scgpt': scgpt_embeddings},
    labels=celltype_labels
)

# For batch labels (e.g., donors)
clustering_task_batch = ClusteringTask()
clustering_results_batch = clustering_task_batch.evaluate(
    embeddings={'scvi': scvi_embeddings, 'scgpt': scgpt_embeddings},
    labels=donor_labels
)

classification_task_batch = MetadataLabelPredictionTask()
classification_results_batch = classification_task_batch.evaluate(
    embeddings={'scvi': scvi_embeddings, 'scgpt': scgpt_embeddings},
    labels=donor_labels
)

# Step 2: Calculate integration scores
evaluator = IntegrationScoreEvaluator()
results = evaluator.evaluate(
    clustering_results_biology=clustering_results_bio,
    clustering_results_batch=clustering_results_batch,
    classification_results_biology=classification_results_bio,
    classification_results_batch=classification_results_batch
)

# Step 3: Analyze results
print(results.summary())
print(f"\nBest model: {results.best_model}")
print(f"Integration Score: {results.best_score.integration_score:.4f}")
print(f"Biology preservation: {results.best_score.B:.4f}")
print(f"Batch leakage: {results.best_score.L:.4f}")

# Convert to DataFrame for further analysis
df = results.to_dataframe()
print(df)
```

## Understanding the Integration Score

### Formula

The integration score is calculated as:

```
IS = (B - L) / (2 × (B + L))
```

Where:
- **B** (Biology): Average of ARI, NMI, Silhouette, and F1 for biological labels (e.g., cell types)
- **L** (Leakage): Average of ARI, NMI, Silhouette, and F1 for batch labels (e.g., donors)

### Score Interpretation

| Score Range | Interpretation | Description |
|-------------|----------------|-------------|
| IS ≥ 0.2    | **Excellent** | Strong biology preservation, minimal batch effects |
| 0.1 ≤ IS < 0.2 | **Very Good** | Clear biology signal over batch effects |
| 0.05 ≤ IS < 0.1 | **Good** | Biology preservation exceeds batch leakage |
| 0.0 ≤ IS < 0.05 | **Marginal** | Biology and batch effects similar |
| -0.1 ≤ IS < 0.0 | **Poor** | Batch effects exceed biology signal |
| IS < -0.1   | **Very Poor** | Strong batch effects, weak biology |

### Why This Metric?

1. **Balances Two Goals**: Good batch correction should preserve biology (high B) while removing batch effects (low L)
2. **Single Number**: Easy to compare methods and choose the best one
3. **Handles Imbalance**: Uses F1 score instead of accuracy, treating rare cell types fairly
4. **Theory-Grounded**: Based on established metrics (ARI, NMI, Silhouette, F1)

## Advanced Usage

### Custom Metric Weights

```python
# Emphasize clustering over classification
evaluator = IntegrationScoreEvaluator(
    weights={
        'ari': 0.3,
        'nmi': 0.3,
        'silhouette': 0.3,
        'f1': 0.1
    }
)
```

### Access Individual Components

```python
results = evaluator.evaluate(...)

for model_name, score in results.scores.items():
    print(f"\n{model_name}:")
    print(f"  Integration Score: {score.integration_score:.4f}")
    print(f"  Biology (B): {score.B:.4f}")
    print(f"    - ARI: {score.biology_components['ari']:.3f}")
    print(f"    - NMI: {score.biology_components['nmi']:.3f}")
    print(f"    - Silhouette: {score.biology_components['silhouette']:.3f}")
    print(f"    - F1: {score.biology_components['f1']:.3f}")
```

### Ranked Models

```python
# Get models sorted by performance
for rank, (model_name, score) in enumerate(results.get_ranked_models(), 1):
    print(f"#{rank}: {model_name} (IS = {score.integration_score:.4f})")
```

## Example Results

From a real dataset with 8,045 cells, 25 cell types (extreme imbalance: 2003:1), and 3 donors:

```
================================================================================
INTEGRATION SCORE RESULTS
================================================================================

#1 SCVI:
  Integration Score: +0.0702
  Biology (B):       0.6970
  Leakage (L):       0.5597
    Biology metrics: ARI=0.732, NMI=0.785, Sil=0.717, F1=0.554
    Batch metrics:   ARI=0.605, NMI=0.657, Sil=0.564, F1=0.413

#2 SCGPT:
  Integration Score: +0.0533
  Biology (B):       0.6822
  Leakage (L):       0.5764
    Biology metrics: ARI=0.720, NMI=0.774, Sil=0.699, F1=0.536
    Batch metrics:   ARI=0.620, NMI=0.672, Sil=0.577, F1=0.437

================================================================================
 BEST MODEL: SCVI (IS = +0.0702)
================================================================================
```

**Interpretation**: scVI achieves the best integration with ~70% biology preservation and ~56% batch leakage, resulting in a positive integration score indicating good batch correction.

## API Reference

### Main Classes

- **`IntegrationScoreEvaluator`**: Main API for calculating scores
- **`IntegrationScoreResults`**: Container for results with helper methods
- **`ModelScore`**: Individual model's score with components

### Core Functions

- **`calculate_integration_score(B, L)`**: Core formula
- **`compute_B_score(...)`**: Biology preservation score
- **`compute_L_score(...)`**: Batch leakage score
- **`normalize_silhouette(sil_score)`**: Convert silhouette from [-1,1] to [0,1]

### Utility Functions

- **`interpret_integration_score(score)`**: Get qualitative interpretation
- **`batch_effect_percentage(L)`**: Convert L to percentage
- **`biology_preservation_percentage(B)`**: Convert B to percentage

## Citation

If you use this package in your research, please cite:

```bibtex
@software{scintegration2026,
  author = {SkylarPurks},
  title = {scintegration: Integration Score Calculator for Single-Cell Batch Correction},
  year = {2026},
  url = {https://github.com/SkylarPurks/scintegration}
}
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run tests (`pytest`)
5. Format code (`black .`)
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of [czbenchmarks](https://github.com/chanzuckerberg/czbenchmarks)
- Inspired by integration metrics from the single-cell community
- Uses established metrics: ARI, NMI, Silhouette coefficient, F1 score

---