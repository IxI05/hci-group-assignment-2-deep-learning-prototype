import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "HCI-images-deep learning" / "test_2.py"

spec = importlib.util.spec_from_file_location("farm2energy_app", APP_PATH)
farm2energy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm2energy_app)


def test_upload_analysis_uses_uploaded_image_without_restarting_camera_stream():
    policy = farm2energy_app.get_upload_image_policy()

    assert policy["restart_camera_before_analysis"] is False
    assert policy["display_uploaded_image_immediately"] is True


def test_audio_stops_when_detection_information_is_applied():
    assert farm2energy_app.should_stop_audio_for_action("apply_detection") is True


def test_submit_report_feedback_is_inline_submitted_text():
    assert farm2energy_app.get_submit_feedback_text() == "Submitted"


def test_interface_supports_maximized_window_and_scrolling():
    assert farm2energy_app.should_start_maximized() is True
    assert farm2energy_app.should_use_scrollable_content() is True


def test_default_location_is_empty():
    assert farm2energy_app.get_default_location() == ""


def test_default_condition_note_is_empty():
    assert farm2energy_app.get_default_condition_note() == ""


def test_empty_location_does_not_block_image_upload(monkeypatch):
    class FakeCapture:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(farm2energy_app, "open_camera_capture", lambda: FakeCapture())
    monkeypatch.setattr(farm2energy_app.filedialog, "askopenfilename", lambda filetypes: "sample.png")
    monkeypatch.setattr(
        farm2energy_app.cv2,
        "imread",
        lambda path: np.zeros((80, 80, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(farm2energy_app.messagebox, "showerror", lambda *args, **kwargs: None)

    root = farm2energy_app.tk.Tk()
    root.withdraw()
    try:
        app = farm2energy_app.AdvancedBioenergyApp(root)
        app.location_var.set("")
        app.process_and_ml_predict = lambda image: None

        app.upload_file_analyze()

        assert app.location_var.get() == ""
        assert app.media_state == "analyzed_image"
    finally:
        root.destroy()
