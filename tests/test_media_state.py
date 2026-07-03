import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "HCI-images-deep learning" / "test_2.py"

spec = importlib.util.spec_from_file_location("farm2energy_app", APP_PATH)
farm2energy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm2energy_app)


def test_capture_button_becomes_retake_after_image_analysis():
    assert farm2energy_app.get_capture_button_text("camera_preview") == "Capture & Analyze"
    assert farm2energy_app.get_capture_button_text("analyzed_image") == "Retake Image"


def test_retake_returns_media_state_to_camera_preview():
    assert farm2energy_app.get_next_media_state("analyzed_image", "retake_image") == "camera_preview"


def test_upload_keeps_analysis_state_and_closes_camera():
    assert farm2energy_app.get_next_media_state("camera_preview", "upload_image") == "analyzed_image"
    assert farm2energy_app.get_next_media_state("analyzed_image", "upload_image") == "analyzed_image"
    assert farm2energy_app.should_close_camera_for_upload() is True
