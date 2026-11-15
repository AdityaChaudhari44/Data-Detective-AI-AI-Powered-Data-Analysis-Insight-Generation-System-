
import io
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Data Detective AI",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Data Detective AI")
st.caption("Explore data quality, discover patterns, and generate useful insights.")

# ---------- Data loading ----------
def read_dataset(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def quality_report(df):
    total = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    missing_rate = missing / total if total else 0
    duplicate_rate = duplicates / len(df) if len(df) else 0
    score = round(max(0, 100 * (1 - 0.7 * missing_rate
                                - 0.3 * duplicate_rate)), 1)
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": missing,
        "duplicate_rows": duplicates,
        "quality_score": score
    }


def numeric_data(df):
    return df.select_dtypes(include=np.number).replace(
        [np.inf, -np.inf], np.nan
    )


def normalized_text(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def sales_series(df):
    numeric = numeric_data(df)
    columns = {normalized_text(col): col for col in numeric.columns}
    sales_column = next(
        (
            columns[name]
            for name in ("sales", "total sales", "revenue", "total revenue")
            if name in columns
        ),
        None
    )
    if sales_column is not None:
        return numeric[sales_column], str(sales_column), f"`{sales_column}` column"

    quantity_column = columns.get("quantity")
    price_column = columns.get("unit price")
    if quantity_column is None or price_column is None:
        return None, None, None

    valid = numeric[quantity_column].notna() & numeric[price_column].notna()
    sales = numeric[quantity_column] * numeric[price_column]
    discount_column = columns.get("discount")
    if discount_column is not None:
        sales = sales * (1 - numeric[discount_column].fillna(0) / 100)
        calculation = (
            "Quantity x Unit_Price x (1 - Discount/100); missing "
            "discounts are treated as 0%"
        )
    else:
        calculation = "Quantity x Unit_Price"
    return sales.where(valid), "net sales", calculation


def clean_data(df, method, remove_duplicates):
    result = df.copy()
    actions = []

    if remove_duplicates:
        count = int(result.duplicated().sum())
        result = result.drop_duplicates()
        actions.append(f"Removed {count} duplicate rows.")

    if method == "Median / Mode":
        for col in result.columns:
            if result[col].isna().any():
                if pd.api.types.is_numeric_dtype(result[col]):
                    value = result[col].median()
                    if pd.notna(value):
                        result[col] = result[col].fillna(value)
                else:
                    modes = result[col].mode(dropna=True)
                    if not modes.empty:
                        result[col] = result[col].fillna(modes.iloc[0])

    elif method == "Zero / Unknown":
        for col in result.columns:
            value = 0 if pd.api.types.is_numeric_dtype(
                result[col]
            ) else "Unknown"
            result[col] = result[col].fillna(value)

    elif method == "Drop rows with missing values":
        before = len(result)
        result = result.dropna()
        actions.append(f"Removed {before - len(result)} incomplete rows.")

    if not actions:
        actions.append("No duplicate-removal action was needed.")

    return result, actions


def answer_question(question, df):
    q = re.sub(r"\s+", " ", question.lower()).strip()
    if not q:
        return "Enter a question first."

    def contains_term(*terms):
        return any(re.search(rf"\b{re.escape(term)}\b", q) for term in terms)

    if contains_term("quality") or (
        contains_term("score") and contains_term("data", "dataset")
    ):
        report = quality_report(df)
        return (
            f"Data quality score: {report['quality_score']}/100\n"
            f"Rows: {report['rows']:,}; columns: {report['columns']:,}\n"
            f"Missing cells: {report['missing_cells']:,}; "
            f"duplicate rows: {report['duplicate_rows']:,}"
        )

    if contains_term("missing", "null", "empty"):
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return "No missing values were found."
        return (
            f"Missing cells: {int(missing.sum()):,}\n"
            "Missing values by column:\n" + missing.to_string()
        )

    if contains_term("duplicate", "duplicates"):
        return f"Duplicate rows: {df.duplicated().sum()}"

    if contains_term("correlation", "correlations", "correlate", "related"):
        numeric = numeric_data(df)
        numeric = numeric.dropna(axis=1, how="all")
        numeric_cols = list(numeric.columns)
        if len(numeric_cols) < 2:
            return "At least two numeric columns are required."
        corr = numeric.corr()
        pairs = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        ).stack()
        if pairs.empty:
            return "No correlations could be calculated from these columns."
        return "Correlation results:\n" + pairs.sort_values(
            key=abs, ascending=False
        ).head(10).to_string()

    normalized_question = f" {normalized_text(q)} "
    matched_columns = [
        col for col in df.columns
        if normalized_text(col)
        and f" {normalized_text(col)} " in normalized_question
    ]
    matched_columns.sort(key=lambda col: len(normalized_text(col)), reverse=True)
    selected = max(
        matched_columns,
        key=lambda col: len(normalized_text(col)),
        default=None
    )

    if contains_term("row count", "number of rows", "how many rows"):
        return f"Number of rows: {len(df):,}"
    if contains_term("column count", "number of columns", "how many columns"):
        return f"Number of columns: {len(df.columns):,}"

    if contains_term("column names", "columns", "fields", "variables"):
        return "Dataset columns:\n" + "\n".join(
            f"- {col} ({df[col].dtype})" for col in df.columns
        )

    if contains_term("summary", "summarize", "overview", "describe the dataset"):
        numeric = numeric_data(df)
        lines = [
            f"Rows: {len(df):,}; columns: {len(df.columns):,}",
            f"Missing cells: {int(df.isna().sum().sum()):,}",
            f"Duplicate rows: {int(df.duplicated().sum()):,}",
            "Columns: " + ", ".join(map(str, df.columns))
        ]
        if not numeric.empty:
            lines.append("Numeric summary:\n" + numeric.describe().round(2).to_string())
        return "\n\n".join(lines)

    if selected is not None and contains_term(
        "unique", "distinct", "different values", "categories", "category"
    ):
        values = df[selected].dropna().unique()
        if not len(values):
            return f"{selected} has no non-missing values."
        preview = ", ".join(str(value) for value in values[:50])
        suffix = f"\nShowing 50 of {len(values)} values." if len(values) > 50 else ""
        return f"Unique values in {selected} ({len(values)}):\n{preview}{suffix}"

    filtered_df = df
    applied_filters = []
    numeric = numeric_data(df)
    numeric_mentions = [col for col in matched_columns if col in numeric.columns]
    comparison = re.search(
        r"\b(greater than|more than|above|over|at least|less than|"
        r"fewer than|below|under|at most|equal to|equals|is|>=|<=|>|<|==)\s*"
        r"([-+]?\d[\d,]*(?:\.\d+)?)\b",
        q
    )
    if comparison and numeric_mentions:
        filter_column = numeric_mentions[0]
        operator = comparison.group(1)
        threshold = float(comparison.group(2).replace(",", ""))
        values = numeric[filter_column]
        if operator in ("greater than", "more than", "above", "over", ">"):
            mask = values > threshold
        elif operator in ("at least", ">="):
            mask = values >= threshold
        elif operator in ("less than", "fewer than", "below", "under", "<"):
            mask = values < threshold
        elif operator in ("at most", "<="):
            mask = values <= threshold
        else:
            mask = values == threshold
        filtered_df = filtered_df.loc[mask.fillna(False)]
        applied_filters.append(
            f"{filter_column} {operator} {threshold:g}"
        )

    category_matches = []
    for col in df.columns:
        if col in numeric.columns:
            continue
        distinct_values = df[col].dropna().astype(str).unique()
        for value in distinct_values[:2000]:
            normalized_value = normalized_text(value)
            if (
                len(normalized_value) > 1
                and f" {normalized_value} " in normalized_question
                and normalized_value not in {
                    "all", "average", "mean", "median", "maximum",
                    "minimum", "highest", "lowest", "most", "least",
                    "common", "unique", "distinct", "missing", "null"
                }
            ):
                category_matches.append((col, value, normalized_value))
    if category_matches:
        longest = max(len(item[2]) for item in category_matches)
        best_matches = [item for item in category_matches if len(item[2]) == longest]
        matched_columns_set = set(matched_columns)
        preferred = [item for item in best_matches if item[0] in matched_columns_set]
        if len(preferred) == 1:
            best_matches = preferred
        if len(best_matches) == 1:
            col, value, _ = best_matches[0]
            filtered_df = filtered_df.loc[
                df.loc[filtered_df.index, col].astype(str) == value
            ]
            applied_filters.append(f"{col} = {value}")

    frequency_intent = contains_term(
        "most common", "common", "mode", "frequent", "top values",
        "most popular", "most frequent", "most sold"
    )
    if selected is not None and frequency_intent and selected not in numeric.columns:
        return f"Most common values in {selected}:\n" + (
            filtered_df[selected].value_counts(dropna=False).head(10).to_string()
        )

    if selected is not None and contains_term("distribution", "breakdown", "frequency"):
        return f"Value distribution for {selected}:\n" + (
            filtered_df[selected].value_counts(dropna=False).head(20).to_string()
        )

    if applied_filters and contains_term("show", "find", "list", "display", "which rows"):
        rows = filtered_df.head(50)
        suffix = (
            f"\nShowing 50 of {len(filtered_df)} matching rows."
            if len(filtered_df) > 50 else ""
        )
        return (
            f"Matching rows ({len(filtered_df)}): "
            + ", ".join(applied_filters)
            + "\n\n"
            + rows.to_string(index=False)
            + suffix
        )

    grouped_column = next(
        (
            col for col in matched_columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ),
        None
    )
    group_ranking = contains_term(
        "highest", "largest", "maximum", "most", "top",
        "lowest", "smallest", "minimum", "least"
    )
    is_sales_question = contains_term(
        "sales", "revenue", "turnover", "sold", "sells", "selling"
    )
    if grouped_column is not None and group_ranking:
        numeric = numeric_data(filtered_df)
        measure_column = next(
            (col for col in matched_columns if col in numeric.columns),
            None
        )
        if is_sales_question:
            sales, measure_name, calculation = sales_series(filtered_df)
            if sales is None:
                return (
                    "I can't calculate sales because this dataset has no "
                    "Sales/Revenue column or both Quantity and Unit_Price columns."
                )
        elif measure_column is not None:
            sales = numeric[measure_column]
            calculation = f"the `{measure_column}` column"
            measure_name = str(measure_column)
        else:
            count_column = next(
                (col for col in matched_columns if col != grouped_column),
                None
            )
            grouped = pd.DataFrame({
                "group": filtered_df[grouped_column].fillna("(missing)").astype(str),
                "value": (
                    filtered_df[count_column]
                    if count_column is not None
                    else filtered_df[grouped_column]
                )
            }, index=filtered_df.index)
            counts = grouped.groupby("group")["value"].count()
            descending = not contains_term(
                "lowest", "smallest", "minimum", "least"
            )
            counts = counts.sort_values(ascending=not descending)
            if counts.empty:
                return f"No values found in {grouped_column}."
            ranking = "\n".join(
                f"{name}: {value:,}" for name, value in counts.head(5).items()
            )
            return (
                f"{'Most' if descending else 'Fewest'} records by "
                f"{grouped_column}: {counts.index[0]} ({counts.iloc[0]}).\n\n"
                f"Top {min(5, len(counts))} groups:\n{ranking}"
            )

        grouped = pd.DataFrame({
            "group": filtered_df[grouped_column].fillna("(missing)").astype(str),
            "sales": sales
        }, index=filtered_df.index).dropna(subset=["sales"])
        if grouped.empty:
            return "There are no complete sales values to compare."
        totals = grouped.groupby("group")["sales"].sum()
        descending = not contains_term("lowest", "smallest", "minimum", "least")
        totals = totals.sort_values(ascending=not descending)
        best_value = totals.iloc[0]
        best_groups = totals[totals == best_value].index.tolist()
        ranking = "\n".join(
            f"{name}: {value:,.2f}" for name, value in totals.head(5).items()
        )
        winner = ", ".join(best_groups)
        return (
            f"{'Highest' if descending else 'Lowest'} total {measure_name} "
            f"by {grouped_column}: {winner} "
            f"({best_value:,.2f}).\n"
            f"Calculation: {calculation}.\n\n"
            f"Top {min(5, len(totals))} {grouped_column} totals:\n{ranking}"
        )

    metric = None
    if contains_term("average", "avg", "mean"):
        metric = "average"
    elif contains_term("median"):
        metric = "median"
    elif contains_term("maximum", "highest", "largest", "max"):
        metric = "maximum"
    elif contains_term("minimum", "lowest", "smallest", "min"):
        metric = "minimum"
    elif contains_term("total", "sum"):
        metric = "total"
    elif contains_term("count", "how many"):
        metric = "count"

    if (
        metric
        and grouped_column is not None
        and contains_term("by", "per", "each", "grouped", "group")
    ):
        numeric = numeric_data(filtered_df)
        measure_column = next(
            (col for col in matched_columns if col in numeric.columns),
            None
        )
        if metric == "count":
            count_column = next(
                (col for col in matched_columns if col != grouped_column),
                None
            )
            if count_column is not None:
                values = filtered_df[count_column]
                measure_name = str(count_column)
            else:
                values = filtered_df[grouped_column]
                measure_name = "rows"
            grouped = pd.DataFrame({
                "group": filtered_df[grouped_column].fillna("(missing)").astype(str),
                "value": values
            }, index=filtered_df.index)
            results = grouped.groupby("group")["value"].count().sort_values(
                ascending=False
            )
        else:
            if measure_column is None:
                return "Name the numeric column to calculate by group."
            grouped = pd.DataFrame({
                "group": filtered_df[grouped_column].fillna("(missing)").astype(str),
                "value": numeric[measure_column]
            }, index=filtered_df.index).dropna(subset=["value"])
            if grouped.empty:
                return "There are no numeric values to calculate for these groups."
            measure_name = str(measure_column)
            aggregation = {
                "average": "mean",
                "median": "median",
                "maximum": "max",
                "minimum": "min",
                "total": "sum"
            }[metric]
            results = grouped.groupby("group")["value"].agg(aggregation)
            results = results.sort_values(
                ascending=metric in ("minimum", "lowest")
            )
        if results.empty:
            return "There are no values to calculate for these groups."
        result_lines = "\n".join(
            f"{name}: {value:,.2f}" for name, value in results.head(10).items()
        )
        return f"{metric.title()} {measure_name} by {grouped_column}:\n{result_lines}"

    if metric:
        numeric = numeric_data(filtered_df)
        numeric_cols = list(numeric.columns)
        selected_measure = next(
            (col for col in matched_columns if col in numeric_cols),
            None
        )
        if selected_measure is None and len(numeric_cols) == 1:
            selected_measure = numeric_cols[0]
        selected = selected_measure
        if selected is None:
            return "Name a numeric column, for example 'average Quantity'."
        if selected not in numeric_cols:
            return f"{selected} is not a numeric column; choose a numeric column for {metric}."

        series = numeric[selected].dropna()
        if series.empty:
            if applied_filters:
                return "No rows match these filters: " + ", ".join(applied_filters)
            return f"No numeric values found in {selected}."
        if metric == "average":
            return f"Average {selected}: {series.mean():,.2f}"
        if metric == "median":
            return f"Median {selected}: {series.median():,.2f}"
        if metric == "maximum":
            return f"Maximum {selected}: {series.max():,.2f}"
        if metric == "minimum":
            return f"Minimum {selected}: {series.min():,.2f}"
        if metric == "total":
            return f"Total {selected}: {series.sum():,.2f}"
        if metric == "count":
            return f"Non-missing values in {selected}: {series.count():,}"

    if selected is not None:
        return f"Most common values in {selected}:\n" + (
            df[selected].value_counts(dropna=False).head(10).to_string()
            )

    return (
        "I couldn't identify that question. Try 'Show missing values', "
        "'What is the average Quantity?', 'Show correlations', "
        "'What is the data quality score?', or 'Show the most common Region'."
    )


def detect_anomalies(df, proportion):
    nums = numeric_data(df).dropna(axis=1, how="all")
    if len(df) < 5 or nums.shape[1] == 0:
        return None, "Need at least 5 rows and one numeric column."

    nums = nums.fillna(nums.median()).fillna(0)
    scaled = StandardScaler().fit_transform(nums)

    model = IsolationForest(
        contamination=proportion,
        random_state=42,
        n_estimators=150
    )
    result = df.copy()
    result["Anomaly_Status"] = np.where(
        model.fit_predict(scaled) == -1, "Anomaly", "Normal"
    )
    return result, None


def ai_explanation(prompt):
    try:
        import ollama
        response = ollama.chat(
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful data analyst. Use only supplied "
                        "evidence. Separate observations from hypotheses, "
                        "mention limitations, and never claim correlation "
                        "proves causation."
                    )
                },
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"], None
    except Exception as exc:
        return None, str(exc)


def offline_insights(df, report):
    insights = [
        f"Dataset contains **{report['rows']:,} rows** and "
        f"**{report['columns']:,} columns**, with a quality score of "
        f"**{report['quality_score']}/100**."
    ]

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        insights.append("No missing values were found.")
    else:
        details = ", ".join(
            f"{col}: {count:,}" for col, count in missing.head(5).items()
        )
        insights.append(f"Columns with the most missing values: {details}.")

    if report["duplicate_rows"]:
        insights.append(
            f"Found **{report['duplicate_rows']:,} duplicate rows**; "
            "review them before analysis."
        )
    else:
        insights.append("No duplicate rows were found.")

    numeric = numeric_data(df).dropna(axis=1, how="all")
    if not numeric.empty:
        summaries = []
        for col in numeric.columns[:5]:
            values = numeric[col].dropna()
            if values.empty:
                continue
            summaries.append(
                f"{col}: mean {values.mean():,.2f}, "
                f"median {values.median():,.2f}"
            )
        if summaries:
            insights.append("Numeric summaries: " + "; ".join(summaries) + ".")

        if len(numeric.columns) > 1:
            pairs = numeric.corr().where(
                np.triu(np.ones((len(numeric.columns),) * 2), k=1).astype(bool)
            ).stack()
            if not pairs.empty:
                column_pair = pairs.abs().idxmax()
                correlation = pairs.loc[column_pair]
                insights.append(
                    f"Strongest numeric correlation is between "
                    f"**{column_pair[0]}** and **{column_pair[1]}** "
                    f"(r = {correlation:.2f}); correlation does not imply "
                    "causation."
                )

    return "### Offline data insights\n\n" + "\n\n".join(
        f"- {insight}" for insight in insights
    )


def make_pdf(df, report):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("Data Detective AI Report", styles["Title"]),
        Paragraph(
            "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["Normal"]
        ),
        Spacer(1, 16),
        Paragraph("Dataset Quality Summary", styles["Heading2"])
    ]

    rows = [["Metric", "Value"]]
    rows += [[key.replace("_", " ").title(), str(value)]
             for key, value in report.items()]

    table = Table(rows, colWidths=[240, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B64")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))
    story.extend([table, Spacer(1, 16)])
    story.append(Paragraph("Descriptive Statistics", styles["Heading2"]))

    summary = df.describe(include="all").fillna("").reset_index()
    summary = summary.astype(str).iloc[:20, :7]
    summary_rows = [summary.columns.tolist()] + summary.values.tolist()

    summary_table = Table(summary_rows, repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7)
    ]))
    story.append(summary_table)
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"]
    )
    sample_enabled = st.checkbox("Use sample dataset if no upload", True)

if uploaded is not None:
    try:
        dataset_key = f"{uploaded.name}-{uploaded.size}"
        if st.session_state.get("dataset_key") != dataset_key:
            loaded = read_dataset(uploaded)
            st.session_state["original_df"] = loaded.copy()
            st.session_state["working_df"] = loaded.copy()
            st.session_state["dataset_key"] = dataset_key
            st.session_state.pop("answer", None)
            st.session_state.pop("answer_question_text", None)
    except Exception as exc:
        st.error(f"Unable to read file: {exc}")
        st.stop()

elif sample_enabled:
    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_data.csv"
    )
    if os.path.exists(sample_path):
        if st.session_state.get("dataset_key") != "sample":
            loaded = pd.read_csv(sample_path)
            st.session_state["original_df"] = loaded.copy()
            st.session_state["working_df"] = loaded.copy()
            st.session_state["dataset_key"] = "sample"
            st.session_state.pop("answer", None)
            st.session_state.pop("answer_question_text", None)
    elif "working_df" not in st.session_state:
        st.warning("Add sample_data.csv beside app.py or upload a file.")
        st.stop()

if "working_df" not in st.session_state:
    st.info("Upload a dataset to begin.")
    st.stop()

df = st.session_state["working_df"].copy()

if df.empty or len(df.columns) == 0:
    st.warning("This dataset is empty.")
    st.stop()

report = quality_report(df)

# ---------- KPI cards ----------
a, b, c, d = st.columns(4)
a.metric("Rows", f"{report['rows']:,}")
b.metric("Columns", f"{report['columns']:,}")
c.metric("Quality score", f"{report['quality_score']}/100")
d.metric("Missing cells", f"{report['missing_cells']:,}")

st.progress(report["quality_score"] / 100)
st.caption("Quality score is a simple indicator, not a guarantee of accuracy.")

tabs = st.tabs([
    "Overview", "Data Profile", "Clean Data",
    "Anomalies", "Ask Your Data", "AI Insights", "Export"
])

# ---------- Overview ----------
with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)

    numeric_cols = list(df.select_dtypes(include=np.number).columns)
    categorical_cols = list(df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns)

    st.subheader("Interactive visualization")
    chart = st.selectbox(
        "Chart type",
        ["Bar chart", "Histogram", "Scatter plot", "Line chart", "Box plot"]
    )

    if chart == "Bar chart":
        if categorical_cols:
            x = st.selectbox("Category", categorical_cols)
            if numeric_cols:
                y = st.selectbox("Value", numeric_cols)
                grouped = df.groupby(x, dropna=False)[y].sum().reset_index()
                grouped = grouped.sort_values(y, ascending=False).head(30)
                fig = px.bar(grouped, x=x, y=y, color=x)
            else:
                counts = df[x].fillna("Missing").value_counts().head(30)
                fig = px.bar(x=counts.index, y=counts.values,
                             labels={"x": x, "y": "Count"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    elif chart == "Histogram":
        if numeric_cols:
            col = st.selectbox("Numeric column", numeric_cols)
            st.plotly_chart(
                px.histogram(df, x=col, nbins=30),
                use_container_width=True
            )
        else:
            st.info("No numeric columns found.")

    elif chart == "Scatter plot":
        if len(numeric_cols) >= 2:
            x = st.selectbox("X axis", numeric_cols)
            y = st.selectbox("Y axis", [c for c in numeric_cols if c != x])
            st.plotly_chart(px.scatter(df, x=x, y=y),
                            use_container_width=True)
        else:
            st.info("Need at least two numeric columns.")

    elif chart == "Line chart":
        if numeric_cols:
            x = st.selectbox("X axis", list(df.columns))
            y = st.selectbox("Y axis", numeric_cols)
            plot_df = df[[x, y]].dropna()
            st.plotly_chart(px.line(plot_df, x=x, y=y, markers=True),
                            use_container_width=True)
        else:
            st.info("No numeric columns found.")

    elif chart == "Box plot":
        if numeric_cols:
            y = st.selectbox("Column to inspect", numeric_cols)
            st.plotly_chart(px.box(df, y=y), use_container_width=True)
        else:
            st.info("No numeric columns found.")

    if len(numeric_cols) >= 2:
        st.subheader("Correlation heatmap")
        corr = df[numeric_cols].corr()
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", aspect="auto",
                      color_continuous_scale="RdBu_r"),
            use_container_width=True
        )
        st.caption("Correlation does not establish causation.")

# ---------- Data Profile ----------
with tabs[1]:
    st.subheader("Data quality profile")
    left, right = st.columns(2)

    with left:
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.plotly_chart(
                px.bar(x=missing.index, y=missing.values,
                       labels={"x": "Column", "y": "Missing values"}),
                use_container_width=True
            )

    with right:
        profile = pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(x) for x in df.dtypes],
            "Missing": [int(df[c].isna().sum()) for c in df.columns],
            "Unique values": [int(df[c].nunique()) for c in df.columns]
        })
        st.dataframe(profile, use_container_width=True, hide_index=True)

    st.subheader("Descriptive statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)
    st.write(f"Duplicate rows: **{df.duplicated().sum()}**")
    st.write(f"Fully empty columns: **{int(df.isna().all().sum())}**")

# ---------- Cleaning ----------
with tabs[2]:
    st.subheader("Preview and apply data cleaning")
    method = st.selectbox(
        "Missing-value strategy",
        ["Leave missing values", "Median / Mode",
         "Zero / Unknown", "Drop rows with missing values"]
    )
    remove_dupes = st.checkbox("Remove duplicate rows", True)

    if st.button("Generate cleaning preview"):
        cleaned, actions = clean_data(
            df,
            method if method != "Leave missing values" else "Leave missing values",
            remove_dupes
        )
        st.session_state["clean_preview"] = cleaned
        st.session_state["clean_actions"] = actions

    if "clean_preview" in st.session_state:
        preview = st.session_state["clean_preview"]
        for action in st.session_state["clean_actions"]:
            st.write("• " + action)
        st.write(f"Rows before: {len(df)} | Rows after: {len(preview)}")
        st.dataframe(preview.head(100), use_container_width=True)

        if st.button("Apply cleaning"):
            st.session_state["working_df"] = preview.copy()
            st.session_state.pop("clean_preview", None)
            st.session_state.pop("clean_actions", None)
            st.session_state.pop("answer", None)
            st.session_state.pop("answer_question_text", None)
            st.rerun()

        if st.button("Discard preview"):
            st.session_state.pop("clean_preview", None)
            st.session_state.pop("clean_actions", None)
            st.rerun()

# ---------- Anomaly detection ----------
with tabs[3]:
    st.subheader("Unusual-record detection")
    st.caption("An unusual record is not necessarily an error.")
    proportion = st.slider(
        "Expected anomaly proportion", 0.01, 0.20, 0.05, 0.01
    )

    if st.button("Detect anomalies"):
        result, error = detect_anomalies(df, proportion)
        if error:
            st.warning(error)
        else:
            st.session_state["anomaly_result"] = result

    if "anomaly_result" in st.session_state:
        result = st.session_state["anomaly_result"]
        unusual = result[result["Anomaly_Status"] == "Anomaly"]
        st.metric("Flagged records", len(unusual))
        st.dataframe(unusual, use_container_width=True)
        st.download_button(
            "Download anomaly results",
            result.to_csv(index=False).encode("utf-8"),
            "anomaly_results.csv",
            "text/csv"
        )

# ---------- Ask Your Data ----------
with tabs[4]:
    st.subheader("Ask Your Data")
    st.write("This basic question-answering feature works offline.")
    question = st.text_input(
        "Ask a question",
        placeholder="What is the average Quantity?"
    )
    if st.button("Answer question"):
        st.session_state["answer"] = answer_question(question, df)
        st.session_state["answer_question_text"] = question
    if (
        "answer" in st.session_state
        and st.session_state.get("answer_question_text") == question
    ):
        st.text_area("Answer", st.session_state["answer"], height=180)

    st.caption(
        "Try: Show missing values; list the columns; "
        "What is the average Quantity for Online?; "
        "Which Sales_Channel has the highest sales?; "
        "Show the most common Region."
    )

# ---------- Generative AI ----------
with tabs[5]:
    st.subheader("Evidence-to-Explanation Engine")
    st.write(
        "Generate explanations from a dataset summary using a local Ollama model."
    )
    numeric = numeric_data(df)
    evidence = (
        f"Rows: {len(df)}; columns: {len(df.columns)}\n"
        f"Quality report: {report}\n"
        f"Column types:\n{df.dtypes.to_string()}\n"
        f"Numeric summary:\n{numeric.describe().to_string()[:4000]}\n"
    )
    focus = st.text_area(
        "What should the AI investigate?",
        "Summarize patterns, possible risks, limitations, and next steps."
    )
    if st.button("Generate AI explanation"):
        prompt = (
            "Analyze this dataset profile. Separate observed facts from "
            "hypotheses. Do not invent values. Mention limitations and "
            "suggest three practical next steps.\n\n"
            f"Question: {focus}\n\nEvidence:\n{evidence}"
        )
        with st.spinner("Contacting local AI model..."):
            explanation, error = ai_explanation(prompt)
            st.session_state["ai_error"] = error
            st.session_state["ai_result"] = (
                explanation if explanation else offline_insights(df, report)
            )

    if "ai_result" in st.session_state:
        if st.session_state.get("ai_error"):
            st.warning(
                "Ollama could not generate an explanation, so these are "
                "offline statistical insights instead. "
                f"Details: {st.session_state['ai_error'][:250]}"
            )
        st.markdown(st.session_state["ai_result"])

    st.info(
        "Generated explanations require a running Ollama service and an "
        "installed model. If Ollama is unavailable, this tab shows offline "
        "statistical insights from the current dataset instead."
    )

# ---------- Export ----------
with tabs[6]:
    st.subheader("Download results")
    st.download_button(
        "Download current dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "analyzed_dataset.csv",
        "text/csv"
    )
    st.download_button(
        "Download quality report (CSV)",
        pd.DataFrame([report]).to_csv(index=False).encode("utf-8"),
        "data_quality_report.csv",
        "text/csv"
    )

    if st.button("Prepare PDF report"):
        try:
            st.session_state["pdf_data"] = make_pdf(df, report)
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    if "pdf_data" in st.session_state:
        st.download_button(
            "Download PDF report",
            st.session_state["pdf_data"],
            "data_detective_report.pdf",
            "application/pdf"
        )

st.divider()
st.caption(
    "Data Detective AI | Validate findings before making important decisions."
)
