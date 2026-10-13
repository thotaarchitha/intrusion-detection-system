"""Explainability dashboard: streamlit run app.py"""

from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from ids import data, pipeline
from ids.explain import explain

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "ids_model.joblib"

st.set_page_config(page_title="Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Training the model for the first time (about a minute)...")
def load_model():
    if not MODEL_PATH.exists():
        pipeline.run(data_dir=DATA_DIR, model_path=MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["preprocessor"], artifact["metrics"]


@st.cache_data
def load_test_set():
    data.download(DATA_DIR)
    return data.load("test", DATA_DIR)


def score(raw):
    """Predict every connection in `raw` and return a table for display."""
    proba = model.predict_proba(prep.transform(raw))
    table = pd.DataFrame({
        "predicted": model.classes_[proba.argmax(axis=1)],
        "confidence": proba.max(axis=1),
        "protocol": raw["protocol_type"].values,
        "service": raw["service"].values,
        "flag": raw["flag"].values,
        "src_bytes": raw["src_bytes"].values,
    }, index=raw.index)
    if "attack" in raw:
        table["actual"] = data.to_category(raw["attack"]).values
        table["attack name"] = raw["attack"].values
    return table


def show_explanation(raw_row):
    result = explain(model, prep, raw_row)
    if result["label"] == "Normal":
        st.success(result["summary"])
    else:
        st.error(result["summary"])

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**What pushed the model toward this verdict**")
        reasons = result["reasons"]
        # Strongest reasons for the verdict, plus the strongest one against it.
        top = pd.concat([reasons.head(6), reasons.tail(1)]).drop_duplicates("feature")
        top = top.assign(points=top["impact"] * 100,
                         direction=["for" if v > 0 else "against" for v in top["impact"]])
        chart = alt.Chart(top).mark_bar().encode(
            x=alt.X("points:Q", title="change in probability (percentage points)"),
            y=alt.Y("reason:N", sort="-x", title=None, axis=alt.Axis(labelLimit=420)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["for", "against"], range=["#d62728", "#7f7f7f"]),
                            legend=None),
            tooltip=["reason", alt.Tooltip("points:Q", format=".1f")],
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(
            f"Before looking at this connection the model's prior for "
            f"**{result['label']}** was {result['baseline']:.0%}. Each bar is how "
            f"much one feature moved that toward the final {result['confidence']:.0%}. "
            "Negative bars argued against it."
        )
    with right:
        st.markdown("**Probability for each category**")
        st.dataframe(
            pd.Series(result["probabilities"], name="probability").sort_values(ascending=False),
            column_config={"probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
            use_container_width=True,
        )
        with st.expander("Raw connection record"):
            st.dataframe(raw_row.T, use_container_width=True)


model, prep, metrics = load_model()

st.title("🛡️ Intelligent Intrusion Detection")
st.write(
    "A Random Forest trained on the NSL-KDD network traffic dataset flags **DoS**, "
    "**Probe** and **Brute Force** activity, and explains each alert in plain English "
    "so an analyst can decide quickly whether to act on it."
)

alerts_tab, perf_tab, upload_tab = st.tabs(["🚨 Alerts", "📊 Model performance", "📁 Check your own traffic"])

with alerts_tab:
    test = load_test_set()
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Connections to replay from the test set", 50, 2000, 300, step=50)
    seed = c2.number_input("Random seed", value=7, step=1)
    only_alerts = c3.toggle("Only show alerts", value=True)

    sample = test.sample(n, random_state=int(seed))
    table = score(sample)
    view = table[table["predicted"] != "Normal"] if only_alerts else table

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Connections", f"{len(table):,}")
    m2.metric("Alerts raised", f"{(table['predicted'] != 'Normal').sum():,}")
    m3.metric("Correct on this sample", f"{(table['predicted'] == table['actual']).mean():.0%}")
    m4.metric("Missed attacks", f"{((table['predicted'] == 'Normal') & (table['actual'] != 'Normal')).sum():,}")

    st.dataframe(
        view,
        column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
        use_container_width=True, height=280,
    )

    if view.empty:
        st.info("No alerts in this sample. Try another seed or a larger sample.")
    else:
        choice = st.selectbox(
            "Pick a connection to explain",
            view.index,
            format_func=lambda i: f"#{i}: {view.at[i, 'predicted']} ({view.at[i, 'confidence']:.0%}) on {view.at[i, 'service']}",
        )
        show_explanation(sample.loc[[choice]])

with perf_tab:
    st.metric("Accuracy on the official NSL-KDD test set", f"{metrics['accuracy']:.1%}")
    st.write(
        f"Trained on {metrics['train_rows']:,} connections, tested on {metrics['test_rows']:,}. "
        f"The test set deliberately contains **{len(metrics['unseen_attacks_in_test'])} attack types "
        "the model never saw in training**, so this is a harder and more honest number than a random split."
    )
    report = pd.DataFrame(metrics["report"]).T.loc[metrics["labels"], ["precision", "recall", "f1-score", "support"]]
    report["support"] = report["support"].astype(int)
    st.dataframe(report.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"}),
                 use_container_width=True)

    st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
    cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    st.dataframe(cm, use_container_width=True)
    with st.expander("Attack types in the test set that never appear in training"):
        st.write(", ".join(metrics["unseen_attacks_in_test"]))

with upload_tab:
    st.write(
        "Upload a CSV in NSL-KDD format: the 41 connection features in order, no header. "
        "An attack label column is optional. You can try it with a few lines copied from `data/KDDTest+.txt`."
    )
    uploaded = st.file_uploader("CSV file", type=["csv", "txt"])
    if uploaded:
        raw = pd.read_csv(uploaded, header=None)
        if raw.shape[1] < len(data.FEATURE_COLUMNS):
            st.error(f"Expected at least {len(data.FEATURE_COLUMNS)} columns, got {raw.shape[1]}.")
        else:
            raw = raw.iloc[:, :len(data.ALL_COLUMNS)]
            raw.columns = data.ALL_COLUMNS[:raw.shape[1]]
            if "attack" in raw and raw["attack"].map(data.ATTACK_CATEGORY).isna().any():
                raw = raw.drop(columns="attack")
            results = score(raw)
            st.dataframe(results, use_container_width=True)
            pick = st.selectbox("Explain row", results.index)
            show_explanation(raw.loc[[pick]])
