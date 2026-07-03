import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "HCI-images-deep learning" / "test_2.py"

spec = importlib.util.spec_from_file_location("farm2energy_app", APP_PATH)
farm2energy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm2energy_app)


def test_report_waste_is_the_initial_interface():
    assert farm2energy_app.get_initial_screen() == "report_waste"


def test_report_detection_button_opens_detection_interface():
    next_screen = farm2energy_app.get_next_screen("report_waste", "use_ai_detection")

    assert next_screen == "detection"


def test_applying_detection_returns_to_report_waste_interface():
    next_screen = farm2energy_app.get_next_screen("detection", "apply_detection")

    assert next_screen == "report_waste"
