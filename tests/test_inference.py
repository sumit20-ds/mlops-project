import pytest

from src import config
from src.inference import ModelNotLoadedError, load_model, predict
from src.schema import StudentFeatures

SAMPLE = {
    "Study_Hours": 4.5,
    "Age": 21,
    "Avg_Daily_Usage_Hours": 4.0,
    "Daily_Unlocks": 134,
    "Physical_Activity_Hours": 2.2,
    "Sleep_Hours_Per_Night": 6.7,
    "Stress_Level": "Medium",
    "Gender": "Male",
    "Academic_Level": "Undergraduate",
    "Most_Used_Platform": "Facebook",
    "Purpose_Of_Use": "Networking",
    "Country": "India",
}


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model artifact not present")
def test_load_model():
    model = load_model()
    assert hasattr(model, "predict")


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model artifact not present")
def test_predict_returns_float_in_plausible_range():
    row = StudentFeatures(**SAMPLE).to_model_row()
    [score] = predict([row])
    assert isinstance(score, float)
    assert 0 <= score <= 10


def test_missing_model_raises_clear_error(tmp_path):
    load_model.cache_clear()
    with pytest.raises(ModelNotLoadedError, match="not found"):
        load_model(str(tmp_path / "does_not_exist.pkl"))
    load_model.cache_clear()


def test_grouped_country_bucketing():
    row = StudentFeatures(**{**SAMPLE, "Country": "Narnia"}).to_model_row()
    assert row["Grouped_country"] == "Other"
    row_top = StudentFeatures(**{**SAMPLE, "Country": "India"}).to_model_row()
    assert row_top["Grouped_country"] == "India"


def test_schema_rejects_invalid_stress_level():
    with pytest.raises(Exception):
        StudentFeatures(**{**SAMPLE, "Stress_Level": "Extreme"})
