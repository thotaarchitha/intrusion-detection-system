"""Per-alert explanations for a Random Forest, in plain English.

How it works: every tree starts at its root with the class mix of the whole
training set (the "baseline"). Each split moves the prediction toward or away
from a class, and we credit that change to the feature the split tested.
Summed over the path and averaged over all trees, the baseline plus every
feature's credit equals the forest's predicted probability exactly, so the
explanation always adds up to what the model actually said.
"""

import numpy as np
import pandas as pd

FLAG_MEANING = {
    "SF": "completed normally",
    "S0": "was a connection attempt that never got a reply (half-open SYN)",
    "REJ": "was rejected by the destination",
    "RSTO": "was reset by the client",
    "RSTR": "was reset by the server",
    "RSTOS0": "was reset by the client before the server replied",
    "SH": "sent a SYN then hung up without completing the handshake",
    "S1": "was established but not closed",
    "S2": "was established, then the client tried to close it",
    "S3": "was established, then the server tried to close it",
    "OTH": "had no clear start or end",
}


def _pct(v):
    """Turn a 0-1 rate into the start of a sentence: "none of the", "only 7% of", ..."""
    if v == 0:
        return "none of the"
    if v == 1:
        return "all of the"
    return f"only {v:.0%} of" if v < 0.15 else f"{v:.0%} of"


# Each entry turns a raw feature value into a short phrase an analyst can read.
DESCRIBE = {
    "duration": lambda v: f"the connection lasted {v:,.0f} seconds",
    "src_bytes": lambda v: f"the client sent {v:,.0f} bytes",
    "dst_bytes": lambda v: f"the server sent back {v:,.0f} bytes",
    "wrong_fragment": lambda v: f"{v:.0f} malformed packet fragments",
    "urgent": lambda v: f"{v:.0f} urgent packets",
    "hot": lambda v: f"{v:.0f} suspicious actions inside the session (e.g. entering system directories)",
    "num_failed_logins": lambda v: f"{v:.0f} failed login attempts",
    "logged_in": lambda v: "the user successfully logged in" if v else "the user never logged in",
    "num_compromised": lambda v: f"{v:.0f} 'compromised' conditions were hit",
    "root_shell": lambda v: "a root shell was obtained" if v else "no root shell was obtained",
    "su_attempted": lambda v: "an 'su root' command was attempted" if v else "no 'su root' was attempted",
    "num_root": lambda v: f"{v:.0f} operations were run as root",
    "num_file_creations": lambda v: f"{v:.0f} files were created",
    "num_shells": lambda v: f"{v:.0f} shell prompts were opened",
    "num_access_files": lambda v: f"{v:.0f} access-control files were touched",
    "is_guest_login": lambda v: "it was a guest / anonymous login" if v else "it was not a guest login",
    "count": lambda v: f"{v:.0f} connections to the same host in the last 2 seconds",
    "srv_count": lambda v: f"{v:.0f} connections to the same service in the last 2 seconds",
    "serror_rate": lambda v: f"{_pct(v)} recent connections to this host were half-open (SYN errors)",
    "srv_serror_rate": lambda v: f"{_pct(v)} recent connections to this service were half-open (SYN errors)",
    "rerror_rate": lambda v: f"{_pct(v)} recent connections to this host were rejected",
    "srv_rerror_rate": lambda v: f"{_pct(v)} recent connections to this service were rejected",
    "same_srv_rate": lambda v: f"{_pct(v)} recent connections to this host used the same service",
    "diff_srv_rate": lambda v: f"{_pct(v)} recent connections to this host hit different services",
    "srv_diff_host_rate": lambda v: f"{_pct(v)} recent connections to this service came from different hosts",
    "dst_host_count": lambda v: f"{v:.0f} of the last 100 connections went to this destination host",
    "dst_host_srv_count": lambda v: f"{v:.0f} of the last 100 connections went to this destination service",
    "dst_host_same_srv_rate": lambda v: f"{_pct(v)} connections to the destination used the same service",
    "dst_host_diff_srv_rate": lambda v: f"{_pct(v)} connections to the destination hit different services",
    "dst_host_same_src_port_rate": lambda v: f"{_pct(v)} connections to the destination came from the same source port",
    "dst_host_srv_diff_host_rate": lambda v: f"{_pct(v)} connections to the destination service came from different hosts",
    "dst_host_serror_rate": lambda v: f"{_pct(v)} connections to the destination host were half-open (SYN errors)",
    "dst_host_srv_serror_rate": lambda v: f"{_pct(v)} connections to the destination service were half-open (SYN errors)",
    "dst_host_rerror_rate": lambda v: f"{_pct(v)} connections to the destination host were rejected",
    "dst_host_srv_rerror_rate": lambda v: f"{_pct(v)} connections to the destination service were rejected",
    "land": lambda v: "source and destination address/port were identical" if v else "source and destination differed",
}


def contributions(model, x):
    """Return (baseline, contrib) for one encoded row `x` (a 1-row DataFrame).

    baseline: shape (n_classes,)   average root-node class mix
    contrib:  shape (n_features, n_classes)   how much each feature moved each class
    """
    x = np.asarray(x, dtype=np.float32).reshape(1, -1)
    n_features, n_classes = x.shape[1], len(model.classes_)
    baseline = np.zeros(n_classes)
    contrib = np.zeros((n_features, n_classes))

    for estimator in model.estimators_:
        tree = estimator.tree_
        # Normalise node values to class probabilities (safe across sklearn versions).
        values = tree.value[:, 0, :]
        probs = values / values.sum(axis=1, keepdims=True)
        path = estimator.decision_path(x).indices  # node ids from root to leaf
        baseline += probs[path[0]]
        for parent, child in zip(path[:-1], path[1:]):
            contrib[tree.feature[parent]] += probs[child] - probs[parent]

    n_trees = len(model.estimators_)
    return baseline / n_trees, contrib / n_trees


def _phrase(feature, raw_row):
    """Describe one encoded feature using the connection's raw values."""
    for column in ("protocol_type", "service", "flag"):
        prefix = column + "_"
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            actual = raw_row[column]
            if column == "flag":
                return f"the connection {FLAG_MEANING.get(actual, f'had TCP flag {actual}')}"
            noun = "protocol" if column == "protocol_type" else "service"
            if actual == value:
                return f"the {noun} was {actual}"
            return f"the {noun} was {actual}, not {value}"
    value = raw_row[feature]
    describe = DESCRIBE.get(feature)
    return describe(value) if describe else f"{feature} = {value}"


def explain(model, preprocessor, raw_row, top_n=3):
    """Explain one connection. `raw_row` is a 1-row DataFrame in NSL-KDD format.

    Returns a dict with the prediction, confidence, a plain-English summary and
    a table of the strongest reasons.
    """
    x = preprocessor.transform(raw_row)
    proba = model.predict_proba(x)[0]
    k = int(np.argmax(proba))
    label = model.classes_[k]
    baseline, contrib = contributions(model, x)

    raw = raw_row.iloc[0]
    reasons = pd.DataFrame({
        "feature": x.columns,
        "impact": contrib[:, k],
    })
    # One-hot columns of the same original field say the same thing; keep the strongest.
    reasons["field"] = reasons["feature"].map(_source_field)
    reasons = (reasons.sort_values("impact", ascending=False)
                      .drop_duplicates("field")
                      .reset_index(drop=True))
    reasons["reason"] = [_phrase(f, raw) for f in reasons["feature"]]

    supporting = reasons[reasons["impact"] > 0].head(top_n)
    if label == "Normal":
        lead = f"Looks normal ({proba[k]:.0%} confidence)"
    else:
        lead = f"Flagged as {label} ({proba[k]:.0%} confidence)"
    if supporting.empty:
        summary = lead + "."
    else:
        bullets = "; ".join(supporting["reason"])
        summary = f"{lead} mainly because {bullets}."

    return {
        "label": label,
        "confidence": float(proba[k]),
        "probabilities": dict(zip(model.classes_, proba.round(4))),
        "baseline": float(baseline[k]),
        "summary": summary,
        "reasons": reasons[["reason", "impact", "feature"]],
    }


def _source_field(feature):
    for column in ("protocol_type", "service", "flag"):
        if feature.startswith(column + "_"):
            return column
    return feature
