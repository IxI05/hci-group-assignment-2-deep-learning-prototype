import importlib.util
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "HCI-images-deep learning" / "test_2.py"

spec = importlib.util.spec_from_file_location("farm2energy_app", APP_PATH)
farm2energy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm2energy_app)


def test_analyze_manure_image_returns_extended_detection_fields():
    model = farm2energy_app.build_machine_learning_model()
    image = np.zeros((380, 500, 3), dtype=np.uint8)
    cv2.rectangle(image, (110, 210), (430, 320), (35, 70, 95), -1)

    result = farm2energy_app.analyze_manure_image(image, model)

    assert result["manure_type"] == "Cow manure"
    assert result["condition"] in {"Good", "Moderate", "Wet / fresh"}
    assert result["weight_kg"] > 0
    assert result["biogas_m3"] > 0
    assert result["estimated_return_rm"] > 0
    assert 60 <= result["confidence_rate"] <= 98
    assert result["co2e_reduction_kg"] == round(result["weight_kg"] * 0.39, 1)


def test_format_report_values_prepares_detection_data_for_report_form():
    result = {
        "manure_type": "Cow manure",
        "condition": "Good",
        "weight_kg": 46.2,
        "biogas_m3": 5.6,
        "estimated_return_rm": 3.08,
        "confidence_rate": 92,
        "co2e_reduction_kg": 18.0,
    }

    report_values = farm2energy_app.format_report_values(result)

    assert report_values["waste_type"] == "Cow manure"
    assert report_values["estimated_quantity"] == "46.2 kg"
    assert "Condition: Good" in report_values["condition_notes"]
    assert "Confidence: 92%" in report_values["condition_notes"]
    assert "Biogas estimate: 5.6 m3" in report_values["condition_notes"]
    assert "CO2e reduction: 18.0 kg" in report_values["condition_notes"]
