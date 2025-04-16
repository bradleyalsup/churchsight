from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QStackedWidget, QListView, QFrame, QLineEdit, QFileDialog, QTextEdit
)
from PyQt6.QtGui import QFont, QImage, QPixmap, QIcon
from PyQt6.QtCore import Qt, QTimer, QSize
import cv2
import face_recognition
import os
import time
from datetime import datetime
import csv

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChurchSight")
        self.setGeometry(100, 100, 1280, 800)

        self.setStyleSheet(
            "QMainWindow { background-color: #1F1F23; color: #E5E7EB; }"
            "QLabel { font-family: 'Segoe UI'; font-size: 14px; color: #E5E7EB; }"
            "QPushButton { background-color: #3B82F6; color: white; border-radius: 6px; padding: 8px 16px; font-size: 14px; }"
            "QPushButton:hover { background-color: #2563EB; }"
            "QFrame#Sidebar { background-color: #111827; }"
            "QListWidget { background-color: #111827; border: none; color: #9CA3AF; font-size: 14px; }"
            "QListWidget::item:selected { background-color: #374151; color: #FFFFFF; }"
            "QFrame#Card { background-color: #27272A; border-radius: 10px; padding: 16px; }"
        )

        self.face_encodings = []
        self.face_counts = []
        self.known_faces = {}
        self.unknown_encodings = []
        self.unknown_counts = []
        self.session_start = time.time()

        self.load_known_faces()
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 20, 10, 10)

        logo = QLabel("👁 ChurchSight")
        logo.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        sidebar_layout.addWidget(logo)
        sidebar_layout.addSpacing(20)

        self.nav_list = QListWidget()
        self.nav_list.setViewMode(QListView.ViewMode.ListMode)
        self.nav_list.setSpacing(10)
        self.nav_list.setFixedWidth(180)
        self.nav_list.setStyleSheet("QListWidget::item { padding: 8px; }")
        for item in ["Detect", "Library", "Label", "Settings"]:
            self.nav_list.addItem(item)
        self.nav_list.currentRowChanged.connect(self.display_tab)

        sidebar_layout.addWidget(self.nav_list)
        sidebar_layout.addStretch()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.build_detect_tab())
        self.stack.addWidget(self.build_library_tab())
        self.stack.addWidget(self.build_label_tab())
        self.stack.addWidget(self.build_settings_tab())

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.stack)

        self.nav_list.setCurrentRow(0)

    def build_detect_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        self.video_label = QLabel("Camera loading...")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 6px;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.stats_label = QLabel("Faces: 0 | Session Time: 0s")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(100)

        layout.addWidget(self.video_label)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.log_view)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        return tab

    def build_library_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("🖼️ Face Library"))

        self.library_faces = QListWidget()
        self.library_faces.setViewMode(QListWidget.ViewMode.IconMode)
        self.library_faces.setIconSize(QSize(120, 120))
        self.library_faces.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.load_face_library()

        layout.addWidget(self.library_faces)
        return tab

    def build_label_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(40, 40, 40, 40)

        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("Enter name for selected face")

        label_btn = QPushButton("Label Face")
        label_btn.clicked.connect(self.label_face)

        layout.addWidget(QLabel("📝 Tag Unknown Faces"))
        layout.addWidget(self.label_input)
        layout.addWidget(label_btn)

        return tab

    def build_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel("⚙️ Settings (coming soon)"))
        layout.addStretch()
        return tab

    def display_tab(self, index):
        self.stack.setCurrentIndex(index)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)

        for i, (enc, (top, right, bottom, left)) in enumerate(zip(encs, locs)):
            match = face_recognition.compare_faces(list(self.known_faces.values()), enc, tolerance=0.5)
            name = "Unknown"
            if True in match:
                name = list(self.known_faces.keys())[match.index(True)]
            else:
                match = face_recognition.compare_faces(self.unknown_encodings, enc, tolerance=0.5)
                index = match.index(True) if True in match else None
                if index is not None and self.unknown_counts[index] >= 2:
                    continue
                elif index is not None:
                    self.unknown_counts[index] += 1
                else:
                    self.unknown_encodings.append(enc)
                    self.unknown_counts.append(1)

                path = f"unknown_faces/face_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.jpg"
                os.makedirs("unknown_faces", exist_ok=True)
                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:
                    cv2.imwrite(path, face_img)

        self.display_log_line(f"Detected {len(encs)} face(s)")
        self.refresh_log()
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))
        elapsed = int(time.time() - self.session_start)
        self.stats_label.setText(f"Faces: {len(encs)} | Session Time: {elapsed}s")

    def label_face(self):
        name = self.label_input.text().strip()
        if not name:
            return
        target_dir = os.path.join("known_faces", name)
        os.makedirs(target_dir, exist_ok=True)
        for filename in os.listdir("unknown_faces"):
            src_path = os.path.join("unknown_faces", filename)
            dst_path = os.path.join(target_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
            os.rename(src_path, dst_path)
        self.label_input.clear()
        self.load_known_faces()
        self.load_face_library()

    def load_known_faces(self):
        self.known_faces.clear()
        if not os.path.exists("known_faces"):
            return
        for person in os.listdir("known_faces"):
            folder = os.path.join("known_faces", person)
            if not os.path.isdir(folder):
                continue
            for img_file in os.listdir(folder):
                path = os.path.join(folder, img_file)
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    self.known_faces[person] = encs[0]

    def load_face_library(self):
        self.library_faces.clear()
        for folder in ["unknown_faces", "known_faces"]:
            if not os.path.exists(folder):
                continue
            for root, _, files in os.walk(folder):
                for file in files:
                    if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    path = os.path.join(root, file)
                    item = QListWidgetItem(QIcon(path), "")
                    item.setToolTip(path)
                    self.library_faces.addItem(item)

    def refresh_log(self):
        today = datetime.now().strftime("%Y-%m-%d")
        log_path = os.path.join("logs", f"{today}.csv")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                self.log_view.setText(f.read())

    def display_log_line(self, line):
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d')}.csv")
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%H:%M:%S"), line])