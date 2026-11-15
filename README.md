
# Data Detective AI

An interactive data analytics platform combining exploratory data analysis,
data-quality checks, machine-learning anomaly detection, and optional
generative AI explanations.

## Features

- CSV and Excel upload
- Data profiling and quality score
- Missing-value and duplicate detection
- Interactive data-cleaning preview
- Bar, histogram, scatter, line, and box charts
- Correlation heatmap
- Isolation Forest anomaly detection
- Offline natural-language questions about columns, values, summaries,
  filters, numeric statistics, and grouped rankings
- Optional local GenAI explanations using Ollama
- CSV and PDF report exports

## Requirements

- Python 3.10 or newer
- VS Code or another Python editor

## Installation on Windows

Open the project folder in VS Code and run these commands in the terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Start the application

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:
http://localhost:8501

## Optional GenAI setup

1. Install Ollama from https://ollama.com/
2. Run this command:

```powershell
ollama pull llama3.2
```

3. Keep the Ollama service running.
4. Open the AI Insights tab in the application.

For a machine with limited memory, you can try the smaller 1B model instead:

```powershell
ollama pull llama3.2:1b
$env:OLLAMA_MODEL = "llama3.2:1b"
streamlit run app.py
```

If Ollama is unavailable or cannot load its model, AI Insights displays
offline statistical insights from the current dataset and shows the Ollama
error. The other analytics features do not require a paid AI subscription.

Ask Your Data works offline with common data questions: dataset summaries,
column names, unique values, missing values, most-common categories, numeric
aggregations, simple filters, and rankings or aggregates by a categorical
column. Questions outside those supported patterns are reported as
unrecognized rather than answered by guessing.

## Dataset

Keep sample_data.csv in the same directory as app.py.
The application loads this sample dataset automatically when no upload
is selected and the sample-data option is enabled.

## Limitations

- The included dataset is synthetic.
- Anomalies are unusual records, not necessarily errors or fraud.
- Correlation does not prove causation.
- AI-generated explanations should be verified.
