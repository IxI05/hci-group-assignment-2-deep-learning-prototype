import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "HCI-images-deep learning" / "test_2.py"

spec = importlib.util.spec_from_file_location("farm2energy_app", APP_PATH)
farm2energy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm2energy_app)


def test_ui_font_family_uses_native_defaults_for_windows_and_mac():
    assert farm2energy_app.get_ui_font_family("win32") == "Segoe UI"
    assert farm2energy_app.get_ui_font_family("darwin") == "Helvetica Neue"


def test_camera_backend_prefers_directshow_on_windows():
    assert farm2energy_app.get_camera_backend("win32") == farm2energy_app.cv2.CAP_DSHOW
    assert farm2energy_app.get_camera_backend("darwin") is None


def test_windows_launcher_is_available():
    launcher = PROJECT_ROOT / "run_app.bat"

    assert launcher.exists()
    launcher_text = launcher.read_text(encoding="utf-8")
    assert "HCI-images-deep learning\\test_2.py" in launcher_text
    assert "python" in launcher_text.lower()
