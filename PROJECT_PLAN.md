
# Data Detective AI — Project Plan

## Problem Statement

Analysts often spend considerable time inspecting unfamiliar datasets,
cleaning errors, exploring patterns, and preparing reports. Standard
dashboards show results but may not explain which patterns need investigation.

## Proposed Solution

Build a dashboard that combines EDA, data-quality assessment, interactive
cleaning, anomaly detection, and optional generative AI explanations.

## Workflow

1. Upload a CSV or Excel dataset.
2. Inspect rows, columns, types, missing values, and duplicates.
3. Preview and apply cleaning operations.
4. Explore distributions, category comparisons, and correlations.
5. Identify unusual records with Isolation Forest.
6. Ask questions using the offline question-answering feature.
7. Generate optional AI explanations from dataset evidence.
8. Download cleaned data and reports.

## Technology Stack

- Python
- Pandas and NumPy
- Streamlit
- Plotly
- Scikit-learn
- ReportLab
- Ollama for optional local GenAI

## Differentiator: Evidence-to-Explanation Engine

The application connects detected patterns with explanations and possible
follow-up actions.

Explanations should distinguish:
- Observed evidence
- Possible interpretations
- Limitations and uncertainty
- Suggested validation steps

## Evaluation

Test with datasets containing missing values, duplicate records, outliers,
categorical variables, date fields, and different column names.

Validate cleaning results, chart accuracy, anomaly flags, and report exports.

## Future Enhancements

- More flexible natural-language analytics
- Statistical significance testing
- Time-series forecasting
- More explainable anomaly detection
- Automatic business KPI discovery
- Improved PDF and HTML reports
