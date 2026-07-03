import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import threading
import time
import subprocess
import sys

# ---- Introducing machine learning libraries----
from sklearn.linear_model import LinearRegression
# --- Introducing Voice dataset (Text-to-Speech)
import pyttsx3 


GREEN_DARK = "#1f5120"
GREEN = "#2e7d32"
GREEN_SOFT = "#edf8f0"
GREEN_LINE = "#a6e9bd"
ORANGE = "#e65100"
BG = "#f4f6f1"
TEXT = "#202124"
MUTED = "#6b7280"


def get_ui_font_family(platform_name=None):
    """Choose a native-looking font that exists on the target platform."""
    current_platform = platform_name or sys.platform
    if current_platform.startswith("win"):
        return "Segoe UI"
    if current_platform == "darwin":
        return "Helvetica Neue"
    return "Arial"


def ui_font(size, weight=None):
    if weight:
        return (get_ui_font_family(), size, weight)
    return (get_ui_font_family(), size)


def get_camera_backend(platform_name=None):
    """Use DirectShow on Windows to make OpenCV camera startup more reliable."""
    current_platform = platform_name or sys.platform
    if current_platform.startswith("win") and hasattr(cv2, "CAP_DSHOW"):
        return cv2.CAP_DSHOW
    return None


def open_camera_capture():
    backend = get_camera_backend()
    if backend is None:
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(0, backend)


def get_initial_screen():
    return "report_waste"


def get_next_screen(current_screen, action):
    transitions = {
        ("report_waste", "use_ai_detection"): "detection",
        ("detection", "back_to_report"): "report_waste",
        ("detection", "apply_detection"): "report_waste",
    }
    return transitions.get((current_screen, action), current_screen)


def get_upload_image_policy():
    return {
        "restart_camera_before_analysis": False,
        "display_uploaded_image_immediately": True,
    }


def should_stop_audio_for_action(action):
    return action == "apply_detection"


def get_submit_feedback_text():
    return "Submitted"


def should_start_maximized():
    return True


def should_use_scrollable_content():
    return True


def build_machine_learning_model():
    """
    Machine Learning Section: simulation learning from a historical dataset.
    Features: [percentage of dark manure pixels, number of manure segments].
    Label: estimated manure weight in kg.
    """
    x_train = np.array([
        [5.0, 1],
        [12.0, 1],
        [15.0, 5],
        [25.0, 8],
        [2.0, 3],
        [8.0, 6],
    ])
    y_train = np.array([2.0, 4.5, 15.0, 30.0, 6.0, 18.0])

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def analyze_manure_image(image, model):
    """Extract CV features and return the full Deep Learning Detection result."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([10, 20, 20])
    upper_brown = np.array([30, 255, 120])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    waste_ratio = float((np.sum(mask > 0) / mask.size) * 100)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
    object_count = len(valid_contours)

    input_data = np.array([[waste_ratio, object_count]])
    predicted_weight = max(0.0, float(model.predict(input_data)[0]))

    predicted_biogas = predicted_weight * 0.12
    estimated_revenue = predicted_biogas * 0.55
    co2e_reduction = predicted_weight * 0.39

    if object_count == 0 or waste_ratio < 0.5:
        confidence_rate = 45
        condition = "Needs review"
    else:
        confidence_rate = int(min(98, 62 + (min(waste_ratio, 18.0) * 1.4) + min(object_count, 5) * 4))
        if waste_ratio >= 12:
            condition = "Good"
        elif waste_ratio >= 4:
            condition = "Moderate"
        else:
            condition = "Wet / fresh"

    return {
        "waste_ratio": round(waste_ratio, 1),
        "object_count": object_count,
        "weight_kg": round(predicted_weight, 1),
        "biogas_m3": round(predicted_biogas, 1),
        "estimated_return_rm": round(estimated_revenue, 2),
        "confidence_rate": confidence_rate,
        "co2e_reduction_kg": round(co2e_reduction, 1),
        "manure_type": "Cow manure",
        "condition": condition,
        "contours": valid_contours,
    }


def format_report_values(result):
    """Prepare detection values for the Report Waste form."""
    notes = (
        f"Condition: {result['condition']}\n"
        f"Confidence: {result['confidence_rate']}%\n"
        f"Biogas estimate: {result['biogas_m3']} m3\n"
        f"CO2e reduction: {result['co2e_reduction_kg']} kg\n"
        f"Estimated return: RM {result['estimated_return_rm']:.2f}"
    )
    return {
        "waste_type": result["manure_type"],
        "estimated_quantity": f"{result['weight_kg']:.1f} kg",
        "condition_notes": notes,
    }


def play_alert_sound(frequency=1800, duration_ms=250, platform_name=None, runner=None):
    """Play a short completion sound on Windows, macOS, or other platforms."""
    current_platform = platform_name or sys.platform
    runner = runner or subprocess.run

    try:
        if current_platform.startswith("win"):
            import winsound

            winsound.Beep(frequency, duration_ms)
        elif current_platform == "darwin":
            runner(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            print("\a", end="", flush=True)
    except Exception as e:
        print(f"[Audio Error] Alert sound playback failed: {e}")


class AdvancedBioenergyApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Farm2Energy")
        self.window.geometry("1120x780")
        self.window.minsize(860, 620)
        self.window.configure(bg=BG)
        if should_start_maximized():
            self.maximize_window()

        # Initialize the AI ​​voice engine
    

        # 1. Initialize and train the machine learning model (初始化并训练机器学习模型).
        self.train_machine_learning_model()

        # 2. Camera
        self.cap = open_camera_capture() # allow system to open camera
        self.current_frame = None
        self.is_camera_running = self.cap is not None and self.cap.isOpened()
        self.audio_lock = threading.Lock()
        self.speech_stop_event = threading.Event()
        self.speech_process = None
        self.speech_engine = None
        self.last_analysis_result = None
        self.active_screen = get_initial_screen()

        # 3. Create UI Interface (创建 UI 界面)
        self.create_widgets()
        if not self.is_camera_running:
            self.video_label.config(
                text="Camera not active\nUpload an image to analyze",
                fg="white",
                font=ui_font(15, "bold"),
                justify=tk.CENTER,
            )

        # 4. Enable thread to update camera footage in real time (时更新Camera画面)
        self.video_thread = threading.Thread(target=self.video_stream_loop, daemon=True)
        self.video_thread.start()

    def maximize_window(self):
        try:
            self.window.state("zoomed")
            return
        except tk.TclError:
            pass

        try:
            self.window.attributes("-zoomed", True)
            return
        except tk.TclError:
            pass

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        self.window.geometry(f"{screen_width}x{screen_height}+0+0")

    def stop_audio_feedback(self):
        self.speech_stop_event.set()
        process = self.speech_process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception as e:
                print(f"[Audio Error] Unable to stop speech process: {e}")

        engine = self.speech_engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def speak_results_async(self, result):
        """
        Audio API Technique: 
        Plays the completion chime and spoken report in one ordered background
        task so macOS does not cut off the speech.
        """
        self.stop_audio_feedback()
        self.speech_stop_event = threading.Event()
        stop_event = self.speech_stop_event
        text = (
            "Analysis complete. "
            f"The model confidence is {result['confidence_rate']} percent. "
            f"The detected manure type is {result['manure_type']} with {result['condition']} condition. "
            f"The estimated mass is {result['weight_kg']:.1f} kilograms. "
            f"The estimated bio gas production potential is {result['biogas_m3']:.1f} cubic meters."
        )

        def speak():
            with self.audio_lock:
                if stop_event.is_set():
                    return
                play_alert_sound()
                if not stop_event.is_set():
                    self.speak_text(text, stop_event)

        # Execute asynchronously so the GUI doesn't freeze or lag during speech.
        # This is intentionally non-daemon so the announcement can finish cleanly.
        speech_thread = threading.Thread(target=speak)
        speech_thread.start()

    def speak_text(self, text, stop_event=None):
        """Speak text using the most reliable available engine for the platform."""
        stop_event = stop_event or threading.Event()
        if sys.platform == "darwin":
            process = None
            try:
                process = subprocess.Popen(
                    ["say", "-r", "155", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.speech_process = process
                while process.poll() is None:
                    if stop_event.is_set():
                        process.terminate()
                        process.wait(timeout=1)
                        return
                    time.sleep(0.05)
                return
            except Exception as e:
                print(f"[Audio Error] macOS speech playback failed: {e}")
            finally:
                if self.speech_process is process:
                    self.speech_process = None

        try:
            engine = pyttsx3.init()
            self.speech_engine = engine
            engine.setProperty('rate', 155)
            engine.setProperty('volume', 1.0)

            voices = engine.getProperty('voices')
            for voice in voices:
                if "EN" in voice.id.upper() or "ENGLISH" in voice.id.upper():
                    engine.setProperty('voice', voice.id)
                    break

            engine.say(text)
            engine.startLoop(False)
            while engine.isBusy():
                if stop_event.is_set():
                    engine.stop()
                    break
                engine.iterate()
                time.sleep(0.05)
            try:
                engine.endLoop()
            except RuntimeError:
                pass
        except Exception as e:
            print(f"[Audio Error] Speech engine playback failed: {e}")
        finally:
            self.speech_engine = None
    

    def train_machine_learning_model(self):
        """
        Machine Learning Section: Simulation learning from a large historical dataset.
        Feature X includes: [Percentage of dark feces, Number of individual feces identified (to correct for near-large to far-small)]
        Label Y includes: [Actual weight of collected feces (kg)]
        """
        print("[ML Base] Loading manure dataset and training ML model...")
        self.ml_model = build_machine_learning_model()
        print("[ML Base] Machine learning model trained successfully!")

    def create_widgets(self):
        header = tk.Label(
            self.window,
            text="Farm2Energy - Report Waste / Deep Learning Detection",
            font=ui_font(16, "bold"),
            bg=GREEN_DARK,
            fg="white",
            pady=12,
        )
        header.pack(fill=tk.X)

        self.content_frame = tk.Frame(self.window, bg=BG)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        self.content_canvas = tk.Canvas(self.content_frame, bg=BG, highlightthickness=0)
        self.content_scrollbar = tk.Scrollbar(self.content_frame, orient=tk.VERTICAL, command=self.content_canvas.yview)
        self.content_canvas.configure(yscrollcommand=self.content_scrollbar.set)
        self.content_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.content_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.screen_container = tk.Frame(self.content_canvas, bg=BG)
        self.screen_window = self.content_canvas.create_window((0, 0), window=self.screen_container, anchor="nw")
        self.screen_container.bind("<Configure>", self.update_scroll_region)
        self.content_canvas.bind("<Configure>", self.resize_scroll_window)
        self.content_canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.content_canvas.bind_all("<Button-4>", self.on_mousewheel)
        self.content_canvas.bind_all("<Button-5>", self.on_mousewheel)

        self.detection_frame = tk.LabelFrame(
            self.screen_container,
            text=" Deep Learning Detection ",
            font=ui_font(11, "bold"),
            bg="white",
            fg=TEXT,
            padx=12,
            pady=10,
        )

        detection_header = tk.Frame(self.detection_frame, bg="white")
        detection_header.pack(fill=tk.X, pady=(0, 8))
        tk.Button(
            detection_header,
            text="Back to Report Waste",
            font=ui_font(10, "bold"),
            bg="white",
            fg=GREEN_DARK,
            command=self.show_report_interface,
            relief=tk.GROOVE,
            bd=2,
        ).pack(side=tk.LEFT)
        tk.Label(
            detection_header,
            text="Run image analysis, then apply the result to the report form.",
            font=ui_font(10),
            bg="white",
            fg=MUTED,
            anchor="e",
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        self.video_label = tk.Label(self.detection_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=(0, 8))

        scan_controls = tk.Frame(self.detection_frame, bg="white")
        scan_controls.pack(fill=tk.X, pady=(0, 10))

        btn_capture = tk.Button(
            scan_controls,
            text="Capture & Analyze",
            font=ui_font(11, "bold"),
            bg=ORANGE,
            fg="white",
            command=self.capture_and_analyze,
            height=2,
        )
        btn_capture.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        btn_upload = tk.Button(
            scan_controls,
            text="Upload & Analyze",
            font=ui_font(11, "bold"),
            bg=GREEN,
            fg="white",
            command=self.upload_file_analyze,
            height=2,
        )
        btn_upload.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        summary_frame = tk.Frame(self.detection_frame, bg="white")
        summary_frame.pack(fill=tk.X, pady=(0, 8))

        self.lbl_confidence = self.create_metric(summary_frame, "Confidence", "--", 0, 0)
        self.lbl_co2e = self.create_metric(summary_frame, "CO2e reduction", "--", 0, 1)

        result_frame = tk.LabelFrame(
            self.detection_frame,
            text=" Deep learning result - Model v2.1 image analysis ",
            font=ui_font(10, "bold"),
            bg="white",
            fg=TEXT,
            padx=10,
            pady=8,
        )
        result_frame.pack(fill=tk.X, pady=(0, 10))

        self.lbl_features = self.create_result_tile(result_frame, "Features", "--", 0, 0)
        self.lbl_type = self.create_result_tile(result_frame, "Manure type", "--", 0, 1)
        self.lbl_condition = self.create_result_tile(result_frame, "Condition", "--", 1, 0)
        self.lbl_weight = self.create_result_tile(result_frame, "Mass estimate", "--", 1, 1)
        self.lbl_biogas = self.create_result_tile(result_frame, "Biogas", "--", 2, 0)
        self.lbl_revenue = self.create_result_tile(result_frame, "Estimated return", "--", 2, 1)

        self.apply_button = tk.Button(
            self.detection_frame,
            text="Apply Information to Report Waste",
            font=ui_font(11, "bold"),
            bg="white",
            fg=GREEN_DARK,
            disabledforeground="#9ca3af",
            relief=tk.GROOVE,
            bd=2,
            command=self.apply_detection_to_report,
            state=tk.DISABLED,
            height=2,
        )
        self.apply_button.pack(fill=tk.X, pady=(0, 2))

        self.report_frame = tk.LabelFrame(
            self.screen_container,
            text=" Report Waste Interface ",
            font=ui_font(11, "bold"),
            bg="white",
            fg=TEXT,
            padx=14,
            pady=12,
        )

        image_card = tk.Frame(self.report_frame, bg=GREEN_SOFT, highlightbackground=GREEN_LINE, highlightthickness=1)
        image_card.pack(fill=tk.X, pady=(0, 14))
        tk.Label(
            image_card,
            text="MANURE IMAGE",
            font=ui_font(10, "bold"),
            bg=GREEN_SOFT,
            fg=TEXT,
            anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(10, 0))
        self.image_status_label = tk.Label(
            image_card,
            text="Required JPG/PNG evidence for AI analysis",
            font=ui_font(9),
            bg=GREEN_SOFT,
            fg=MUTED,
            anchor="w",
        )
        self.image_status_label.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.waste_type_var = tk.StringVar(value="Cow manure")
        self.quantity_var = tk.StringVar()
        self.location_var = tk.StringVar(value="Klang, Selangor")

        self.add_field_label(self.report_frame, "WASTE TYPE")
        self.waste_type_combo = ttk.Combobox(
            self.report_frame,
            textvariable=self.waste_type_var,
            values=("Cow manure", "Goat manure", "Chicken manure", "Mixed manure"),
            state="readonly",
            font=ui_font(11),
        )
        self.waste_type_combo.pack(fill=tk.X, ipady=6, pady=(0, 12))

        self.add_field_label(self.report_frame, "ESTIMATED QUANTITY")
        self.quantity_entry = tk.Entry(self.report_frame, textvariable=self.quantity_var, font=ui_font(12), relief=tk.GROOVE, bd=2)
        self.quantity_entry.pack(fill=tk.X, ipady=9, pady=(0, 12))

        self.add_field_label(self.report_frame, "FARM LOCATION")
        self.location_entry = tk.Entry(self.report_frame, textvariable=self.location_var, font=ui_font(12), relief=tk.GROOVE, bd=2)
        self.location_entry.pack(fill=tk.X, ipady=9, pady=(0, 12))

        self.add_field_label(self.report_frame, "CONDITION NOTES")
        self.condition_text = tk.Text(self.report_frame, height=7, font=ui_font(11), wrap=tk.WORD, relief=tk.GROOVE, bd=2)
        self.condition_text.insert("1.0", "Stored under covered area. Access through east gate.")
        self.condition_text.pack(fill=tk.BOTH, expand=True, pady=(0, 14))

        tk.Button(
            self.report_frame,
            text="Use AI Detection",
            font=ui_font(12, "bold"),
            bg="white",
            fg=GREEN_DARK,
            command=self.open_detection_interface,
            relief=tk.GROOVE,
            bd=2,
            height=2,
        ).pack(fill=tk.X, pady=(0, 10))

        tk.Button(
            self.report_frame,
            text="Submit Report",
            font=ui_font(12, "bold"),
            bg=GREEN_DARK,
            fg="white",
            command=self.submit_report_preview,
            height=2,
        ).pack(fill=tk.X, pady=(0, 4))
        self.submit_status_label = tk.Label(
            self.report_frame,
            text="",
            font=ui_font(11, "bold"),
            bg="white",
            fg=GREEN_DARK,
        )
        self.submit_status_label.pack(fill=tk.X, pady=(6, 0))
        self.show_screen(get_initial_screen())

    def show_screen(self, screen_name):
        self.report_frame.pack_forget()
        self.detection_frame.pack_forget()
        if screen_name == "detection":
            self.detection_frame.pack(fill=tk.BOTH, expand=True)
        else:
            self.report_frame.pack(fill=tk.BOTH, expand=True)
            screen_name = "report_waste"
        self.active_screen = screen_name
        self.window.after_idle(self.reset_scroll_position)

    def update_scroll_region(self, _event=None):
        self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all"))

    def resize_scroll_window(self, event):
        self.content_canvas.itemconfigure(self.screen_window, width=event.width)

    def reset_scroll_position(self):
        self.content_canvas.yview_moveto(0)
        self.update_scroll_region()

    def on_mousewheel(self, event):
        if event.num == 4:
            self.content_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.content_canvas.yview_scroll(1, "units")
        elif event.delta:
            self.content_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def open_detection_interface(self):
        self.show_screen(get_next_screen(self.active_screen, "use_ai_detection"))

    def show_report_interface(self):
        self.show_screen(get_next_screen(self.active_screen, "back_to_report"))

    def create_metric(self, parent, label, value, row, column):
        card = tk.Frame(parent, bg="white", highlightbackground="#e5e7eb", highlightthickness=1)
        card.grid(row=row, column=column, sticky="ew", padx=5, pady=4)
        parent.grid_columnconfigure(column, weight=1)
        value_label = tk.Label(card, text=value, font=ui_font(18, "bold"), bg="white", fg=TEXT)
        value_label.pack(anchor="w", padx=14, pady=(8, 0))
        tk.Label(card, text=label, font=ui_font(10), bg="white", fg=MUTED).pack(anchor="w", padx=14, pady=(0, 8))
        return value_label

    def create_result_tile(self, parent, title, value, row, column):
        tile = tk.Frame(parent, bg=GREEN_SOFT)
        tile.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
        parent.grid_columnconfigure(column, weight=1)
        tk.Label(tile, text=title.upper(), font=ui_font(9, "bold"), bg=GREEN_SOFT, fg=MUTED).pack(anchor="w", padx=12, pady=(8, 0))
        value_label = tk.Label(tile, text=value, font=ui_font(12, "bold"), bg=GREEN_SOFT, fg=TEXT)
        value_label.pack(anchor="w", padx=12, pady=(0, 8))
        return value_label

    def add_field_label(self, parent, text):
        tk.Label(
            parent,
            text=text,
            font=ui_font(9, "bold"),
            bg="white",
            fg="#666666",
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 4))

    def video_stream_loop(self):
        """Projecting camera feed onto the Tkinter interface in real time"""
        while self.is_camera_running:
            if self.cap is None or not self.cap.isOpened():
                self.is_camera_running = False
                break
            ret, frame = self.cap.read()
            if ret and self.is_camera_running:
                self.current_frame = cv2.resize(frame, (500, 380))
                # 转换颜色并在界面显示
                cv_img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv_img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
            time.sleep(0.03)

    def process_and_ml_predict(self, image):
        """Core CV engine: extract features, update AI cards, and enable report autofill."""
        result = analyze_manure_image(image, self.ml_model)
        self.last_analysis_result = result

        cv2.drawContours(image, result["contours"], -1, (80, 190, 70), 2)
        for contour in result["contours"]:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (125, 210, 90), 2)
        cv2.putText(image, "Scanning", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 255, 230), 2)
        cv2.putText(image, f"{result['confidence_rate']}%", (410, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 255, 230), 2)

        self.lbl_confidence.config(text=f"{result['confidence_rate']}%")
        self.lbl_co2e.config(text=f"{result['co2e_reduction_kg']:.1f} kg")
        self.lbl_features.config(text=f"{result['waste_ratio']:.1f}% | {result['object_count']} units")
        self.lbl_type.config(text=result["manure_type"])
        self.lbl_condition.config(text=result["condition"])
        self.lbl_weight.config(text=f"{result['weight_kg']:.1f} kg")
        self.lbl_biogas.config(text=f"{result['biogas_m3']:.1f} m3")
        self.lbl_revenue.config(text=f"RM {result['estimated_return_rm']:.2f}")
        self.apply_button.config(state=tk.NORMAL, bg=GREEN_SOFT)
        self.image_status_label.config(text=f"Ready - AI analysis confidence {result['confidence_rate']}%")

        self.speak_results_async(result)

        # 在界面展示带有分析标记的静态处理结果
        self.is_camera_running = False # 暂停摄像头实时刷新，冻结画面查看结果
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_to_show = Image.fromarray(rgb_img)
        imgtk = ImageTk.PhotoImage(image=img_to_show)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

    def apply_detection_to_report(self):
        if not self.last_analysis_result:
            messagebox.showwarning("No analysis", "Please complete AI detection before applying information.")
            return

        if should_stop_audio_for_action("apply_detection"):
            self.stop_audio_feedback()
        report_values = format_report_values(self.last_analysis_result)
        self.waste_type_var.set(report_values["waste_type"])
        self.quantity_var.set(report_values["estimated_quantity"])
        self.condition_text.delete("1.0", tk.END)
        self.condition_text.insert("1.0", report_values["condition_notes"])
        self.show_screen(get_next_screen(self.active_screen, "apply_detection"))
        self.submit_status_label.config(text="AI detection information applied.")

    def submit_report_preview(self):
        self.submit_status_label.config(text=get_submit_feedback_text())

    def capture_and_analyze(self):
        """Photo and button action"""
        if self.current_frame is not None:
            # copy current images to making ML analysis(复制当前画面进行机器学习分析)
            frame_to_analyze = self.current_frame.copy()
            self.process_and_ml_predict(frame_to_analyze)
        else:
            messagebox.showwarning("Warning: No camera footage detected.")

    def upload_file_analyze(self):
        """Restore camera or upload local file"""
        self.is_camera_running = False
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Image error", "The selected image could not be opened.")
                return
            img_resized = cv2.resize(img, (500, 380))
            self.current_frame = img_resized.copy()
            self.show_processed_image(img_resized)
            self.process_and_ml_predict(img_resized)

    def show_processed_image(self, image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_to_show = Image.fromarray(rgb_img)
        imgtk = ImageTk.PhotoImage(image=img_to_show)
        self.video_label.config(image=imgtk, text="")
        self.video_label.image = imgtk

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBioenergyApp(root)
    root.mainloop()
