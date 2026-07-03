import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "HCI-images-deep learning" / "test_2.py"

spec = importlib.util.spec_from_file_location("farm2energy_app", APP_PATH)
farm2energy_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(farm2energy_app)


def test_button_styles_keep_text_visible_in_normal_active_and_hover_states():
    for variant in ("primary", "orange", "green", "outline"):
        palette = farm2energy_app.get_button_palette(variant)

        assert palette["bg"] != palette["fg"]
        assert palette["activebackground"] != palette["activeforeground"]
        assert palette["hoverbackground"] != palette["hoverforeground"]


def test_filled_buttons_do_not_turn_white_when_clicked_or_hovered():
    for variant in ("primary", "orange", "green"):
        palette = farm2energy_app.get_button_palette(variant)

        assert palette["activebackground"] != "white"
        assert palette["hoverbackground"] != "white"
