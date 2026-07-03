import tkinter as tk
from tkinter import filedialog, messagebox
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
        self.window.geometry("1000x700")
        self.window.configure(bg="#f4f6f9")

        # Initialize the AI ​​voice engine
    

        # 1. Initialize and train the machine learning model (初始化并训练机器学习模型).
        self.train_machine_learning_model()

        # 2. Camera
        self.cap = cv2.VideoCapture(0) # allow system to open camera
        self.current_frame = None
        self.is_camera_running = True
        self.audio_lock = threading.Lock()

        # 3. Create UI Interface (创建 UI 界面)
        self.create_widgets()

        # 4. Enable thread to update camera footage in real time (时更新Camera画面)
        self.video_thread = threading.Thread(target=self.video_stream_loop, daemon=True)
        self.video_thread.start()

    def speak_results_async(self, weight, biogas):
        """
        Audio API Technique: 
        Plays the completion chime and spoken report in one ordered background
        task so macOS does not cut off the speech.
        """
        text = (
            "Analysis complete. "
            f"The machine learning model predicts a total waste weight of {weight:.1f} kilograms. "
            f"The estimated bio gas production potential is {biogas:.1f} cubic meters."
        )

        def speak():
            with self.audio_lock:
                play_alert_sound()
                self.speak_text(text)

        # Execute asynchronously so the GUI doesn't freeze or lag during speech.
        # This is intentionally non-daemon so the announcement can finish cleanly.
        speech_thread = threading.Thread(target=speak)
        speech_thread.start()

    def speak_text(self, text):
        """Speak text using the most reliable available engine for the platform."""
        if sys.platform == "darwin":
            try:
                subprocess.run(
                    ["say", "-r", "155", text],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception as e:
                print(f"[Audio Error] macOS speech playback failed: {e}")

        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 155)
            engine.setProperty('volume', 1.0)

            voices = engine.getProperty('voices')
            for voice in voices:
                if "EN" in voice.id.upper() or "ENGLISH" in voice.id.upper():
                    engine.setProperty('voice', voice.id)
                    break

            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[Audio Error] Speech engine playback failed: {e}")
    

    def train_machine_learning_model(self):
        """
        Machine Learning Section: Simulation learning from a large historical dataset.
        Feature X includes: [Percentage of dark feces, Number of individual feces identified (to correct for near-large to far-small)]
        Label Y includes: [Actual weight of collected feces (kg)]
        """
        print("[ML Base] Loading manure dataset and training ML model...")
        
        # Simulated historical dataset: (Percentage, Number of unique samples) -> Actual weight
        # 模拟历史数据集：(占比, 独立数量) -> 真实重量
        # Even if the proportion is large (due to proximity), if the quantity is small, the actual weight will be light.
        #解决近大远小的问题（Impotant!!!）
        X_train = np.array([
            [5.0,  1],   # 离得近，只有1堆 -> 2.0kg
            [12.0, 1],   # 极近，只有1堆 -> 4.5kg
            [15.0, 5],   # 正常距离，5堆 -> 15.0kg
            [25.0, 8],   # 满地都是，8堆 -> 30.0kg
            [2.0,  3],   # 离得远，虽小但有3堆 -> 6.0kg (纠正近大远小)
            [8.0,  6],   # 较远，有6堆 -> 18.0kg
        ])
        Y_train = np.array([2.0, 4.5, 15.0, 30.0, 6.0, 18.0])

        # Initialize and train a linear regression machine learning model
        self.ml_model = LinearRegression()
        self.ml_model.fit(X_train, Y_train)
        print("[ML Base] Machine learning model trained successfully!")

    def create_widgets(self):
        # Top header
        header = tk.Label(self.window, text="Farm2Energy", 
                          font=("Microsoft YaHei", 15, "bold"), bg="#1b5e20", fg="white", pady=10)
        header.pack(fill=tk.X)

        main_frame = tk.Frame(self.window, bg="#f4f6f9")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left column: Camera/Image Preview (左栏：摄像头/图像预览)
        left_frame = tk.LabelFrame(main_frame, text=" Real-time image acquisition (Visual API) ", font=("Microsoft YaHei", 10, "bold"), bg="white")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.video_label = tk.Label(left_frame, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right column: Control and ML Prediction Report(右栏：控制与 ML 预测报告)
        right_frame = tk.Frame(main_frame, bg="#f4f6f9", width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        # Control Button
        btn_capture = tk.Button(right_frame, text="📸 Take photos and analyze", font=("Microsoft YaHei", 11, "bold"), bg="#e65100", fg="white", command=self.capture_and_analyze, height=2)
        btn_capture.pack(fill=tk.X, pady=5)

        btn_upload = tk.Button(right_frame, text="📁 Upload local file", font=("Microsoft YaHei", 11), bg="#2e7d32", fg="white", command=self.upload_file_analyze)
        btn_upload.pack(fill=tk.X, pady=5)

        # ML Predictive Output
        self.report_frame = tk.LabelFrame(right_frame, text="ML Model Prediction ", font=("Microsoft YaHei", 10, "bold"), bg="white", padx=10, pady=10)
        self.report_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.lbl_features = tk.Label(self.report_frame, text="Feature extraction (percentage | number): --", font=("Microsoft YaHei", 10), bg="white", anchor="w")
        self.lbl_features.pack(fill=tk.X, pady=5)

        self.lbl_weight = tk.Label(self.report_frame, text="ML Predicting total fecal weight: -- kg", font=("Microsoft YaHei", 12, "bold"), fg="#1b5e20", bg="white", anchor="w")
        self.lbl_weight.pack(fill=tk.X, pady=10)

        self.lbl_biogas = tk.Label(self.report_frame, text="Expected biogas production: -- m³", font=("Microsoft YaHei", 11), bg="white", anchor="w")
        self.lbl_biogas.pack(fill=tk.X, pady=5)

        self.lbl_revenue = tk.Label(self.report_frame, text="Estimated Farmer Profits: -- RM", font=("Microsoft YaHei", 11), bg="white", anchor="w")
        self.lbl_revenue.pack(fill=tk.X, pady=5)

    def video_stream_loop(self):
        """Projecting camera feed onto the Tkinter interface in real time"""
        while self.is_camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = cv2.resize(frame, (500, 380))
                # 转换颜色并在界面显示
                cv_img = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv_img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
            time.sleep(0.03)

    def process_and_ml_predict(self, image):
        """核心处理：使用计算机视觉提取特征，并用真实的机器学习模型进行预测"""
        #Core CV Engine: extracts pixel contours, executes ML prediction, and fires iterative Audio speech"""
        # 1. Feature Extraction A: Pixel density filtering
        # 1. 提取特征 A: 粪便颜色占比
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_brown = np.array([10, 20, 20])
        upper_brown = np.array([30, 255, 100])
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        waste_ratio = float((np.sum(mask > 0) / mask.size) * 100)

        # 2. 提取特征 B: 识别独立的粪便数量 (以此对抗近大远小问题)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 过滤掉太小的轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        object_count = len(valid_contours)

        # 如果什么都没检测到，给个保底数据
        if object_count == 0 and waste_ratio < 0.5:
            object_count = 0

        # 3. 🎨 视觉 API 叠加反馈：在画面上把检测到的轮廓和数量画出来
        cv2.drawContours(image, valid_contours, -1, (0, 255, 0), 2)
        cv2.putText(image, f"Detected Blocks: {object_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. 🤖 ML Model predict
        # 把当前提取到的特征组合成二维数组 [[占比, 数量]] 喂给模型
        input_data = np.array([[waste_ratio, object_count]])
        predicted_weight = self.ml_model.predict(input_data)[0]
        predicted_weight = max(0.0, float(predicted_weight)) # 避免负数

        # 5. 计算衍生能效数据
        predicted_biogas = predicted_weight * 0.04
        estimated_revenue = predicted_biogas * 0.55
        
        # Call the asynchronous AI voice broadcast API 
        self.speak_results_async(predicted_weight, predicted_biogas)

        # 7. Update UI Oupcome(更新 UI 结果)
        self.lbl_features.config(text=f"Features: Density {waste_ratio:.1f}% | Segmented Units {object_count}")
        self.lbl_weight.config(text=f"ML Predicted Weight: {predicted_weight:.2f} kg")
        self.lbl_biogas.config(text=f"Predicted Biogas Yield: {predicted_biogas:.2f} m³")
        self.lbl_revenue.config(text=f"Estimated Financial Return: RM {estimated_revenue:.2f}")

        # 在界面展示带有分析标记的静态处理结果
        self.is_camera_running = False # 暂停摄像头实时刷新，冻结画面查看结果
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_to_show = Image.fromarray(rgb_img)
        imgtk = ImageTk.PhotoImage(image=img_to_show)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

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
        self.is_camera_running = True # 重新激活camera流
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            img = cv2.imread(file_path)
            img_resized = cv2.resize(img, (500, 380))
            self.process_and_ml_predict(img_resized)

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedBioenergyApp(root)
    root.mainloop()
