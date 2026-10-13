import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ids import data
from ids.explain import contributions, explain
from ids.pipeline import Preprocessor


def make_connections(n=300, seed=0):
    """Synthetic NSL-KDD-shaped rows: DoS rows are half-open floods, the rest look normal."""
    rng = np.random.default_rng(seed)
    rows = pd.DataFrame(0.0, index=range(n), columns=data.FEATURE_COLUMNS)
    rows["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n)
    rows["service"] = rng.choice(["http", "ftp", "private"], n)
    rows["flag"] = "SF"
    rows["src_bytes"] = rng.integers(100, 5000, n)
    rows["count"] = rng.integers(1, 20, n)
    is_dos = rng.random(n) < 0.4
    rows.loc[is_dos, "flag"] = "S0"
    rows.loc[is_dos, "count"] = rng.integers(200, 500, is_dos.sum())
    rows.loc[is_dos, "serror_rate"] = 1.0
    rows.loc[is_dos, "src_bytes"] = 0
    rows["attack"] = np.where(is_dos, "neptune", "normal")
    rows["difficulty"] = 20
    return rows


@pytest.fixture(scope="module")
def trained():
    raw = make_connections()
    prep = Preprocessor().fit(raw)
    model = RandomForestClassifier(n_estimators=20, random_state=0).fit(
        prep.transform(raw), data.to_category(raw["attack"]))
    return raw, prep, model


def test_every_attack_name_maps_to_a_category():
    assert set(data.ATTACK_CATEGORY.values()) == set(data.CATEGORIES)
    with pytest.raises(ValueError, match="Unmapped"):
        data.to_category(pd.Series(["normal", "made_up_attack"]))


def test_preprocessor_drops_constant_columns_and_handles_unseen_categories(trained):
    raw, prep, _ = trained
    assert "num_outbound_cmds" in prep.dropped
    new = raw.head(1).copy()
    new["service"] = "never_seen_before"
    X = prep.transform(new)
    assert list(X.columns) == prep.selected
    assert X.filter(like="service_").sum(axis=1).item() == 0


def test_contributions_add_up_to_predicted_probability(trained):
    raw, prep, model = trained
    for i in range(5):
        x = prep.transform(raw.iloc[[i]])
        baseline, contrib = contributions(model, x)
        np.testing.assert_allclose(baseline + contrib.sum(axis=0), model.predict_proba(x)[0], atol=1e-9)


def test_explanation_names_the_flood_signals(trained):
    raw, prep, model = trained
    dos_row = raw[raw["attack"] == "neptune"].iloc[[0]]
    result = explain(model, prep, dos_row)
    assert result["label"] == "DoS"
    assert result["summary"].startswith("Flagged as DoS")
    assert any(word in result["summary"] for word in ("half-open", "connections", "bytes"))
