

import sys
import os
import glob
import time
import json
import cv2
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QSlider, QFileDialog, QHBoxLayout, QGroupBox, QFormLayout, QCheckBox,
    QSpacerItem, QSizePolicy, QSpinBox
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer


from src.util import Geometry

class ImageSequencePlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Sequence Player")

        # GUI Elements
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)


        # Path labels
        self.dataset_path_label = QLabel("Dataset path: Not loaded")
        self.keyframe_json_path_label = QLabel("Keyframe JSON path: Not loaded")
        self.map_points_json_path_label = QLabel("Map Points JSON path: Not loaded")

        # Statistics labels
        self.pose_label = QLabel("-")
        self.frame_id_label = QLabel("-")
        self.timestamp_label = QLabel("-")
        self.map_point_info_label = QLabel("-")

        # Statistics and paths
        stats_group = QGroupBox("File paths and statistics")

        stats_layout = QFormLayout()
        stats_layout.addRow(self.dataset_path_label)
        stats_layout.addRow(self.keyframe_json_path_label)
        stats_layout.addRow(self.map_points_json_path_label)
        stats_layout.addRow("Frame ID:", self.frame_id_label)
        stats_layout.addRow("Timestamp:", self.timestamp_label)
        stats_layout.addRow("Estimated Pose:", self.pose_label)
        stats_layout.addRow("Map Point Info:", self.map_point_info_label)

        stats_group.setLayout(stats_layout)

        # Control section
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.fps_label = QLabel("FPS:")
        self.fps_input = QSpinBox()
        self.fps_input.setRange(1, 120)
        self.fps_input.setValue(30)  # Default FPS
        self.play_button = QPushButton("Start")
        self.prev_button = QPushButton("Previous Frame")
        self.next_button = QPushButton("Next Frame")
        self.load_button = QPushButton("Load Dataset (Images)")
        self.load_json_button = QPushButton("Load Keyframe JSON")
        self.keyframe_checkbox = QCheckBox("Show only keyframes")
        self.load_map_points_button = QPushButton("Load Map Points JSON")
        self.show_map_points_checkbox = QCheckBox("Show map points")

        
        # Load buttons stacked vertically
        load_buttons_layout = QVBoxLayout()
        load_buttons_layout.addWidget(self.load_button)
        load_buttons_layout.addWidget(self.load_json_button)
        load_buttons_layout.addWidget(self.load_map_points_button)

        # Checkbox group (can expand later)
        checkbox_layout = QVBoxLayout()
        checkbox_layout.addWidget(self.keyframe_checkbox)
        checkbox_layout.addWidget(self.show_map_points_checkbox)
        checkbox_layout.addStretch()


        # First row: previous / next frame
        frame_nav_row = QHBoxLayout()
        frame_nav_row.addWidget(self.next_button)
        frame_nav_row.addWidget(self.prev_button)

        # FPS input side-by-side
        play_fps_row = QHBoxLayout()
        play_fps_row.addWidget(self.fps_label)
        play_fps_row.addWidget(self.fps_input)

        play_layout = QVBoxLayout()
        play_layout.addStretch()
        play_layout.addLayout(frame_nav_row)
        play_layout.addWidget(self.play_button)
        play_layout.addLayout(play_fps_row)
        play_layout.addStretch()

        # Combine control sections
        controls_layout = QHBoxLayout()
        controls_layout.addLayout(load_buttons_layout)
        controls_layout.addLayout(checkbox_layout)
        controls_layout.addLayout(play_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=3)
        layout.addWidget(self.slider)
        layout.addLayout(controls_layout)
        layout.addWidget(stats_group)

        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer)

        self.setLayout(layout)

        # State
        self.image_paths = []
        self.current_index = 0
        self.is_playing = False
        self.start_time = None
        self.keyframes = {}  # Dictionary to store the keyframe data
        self.map_points = {}  # Dictionary to store the map points data
        self.show_only_keyframes = False  # Toggle to show only keyframes

        # Timer
        self.timer = QTimer()
        self.timer.setInterval(100)  # ~10 fps

        # Connect Signals
        self.load_button.clicked.connect(self.load_images)
        self.load_json_button.clicked.connect(self.load_json)
        self.load_map_points_button.clicked.connect(self.load_map_points)
        self.play_button.clicked.connect(self.toggle_playback)
        self.slider.valueChanged.connect(self.slider_changed)
        self.keyframe_checkbox.stateChanged.connect(self.toggle_keyframe_mode)
        self.fps_input.valueChanged.connect(self.update_fps)
        self.timer.timeout.connect(lambda: self.next_frame(1))
        self.prev_button.clicked.connect(lambda: self.next_frame(-1))
        self.next_button.clicked.connect(lambda: self.next_frame(1))

    def load_images(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images")
        if not folder:
            return

        # Load all images in the folder
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(folder, ext)))
        self.image_paths.sort()

        if self.image_paths:
            self.slider.setMaximum(len(self.image_paths) - 1)
            self.slider.setEnabled(True)
            self.current_index = 0
            self.show_image(0)
            self.start_time = time.time()
            self.dataset_path_label.setText(f"Dataset path: {folder}")

    def load_json(self):
        json_file, _ = QFileDialog.getOpenFileName(self, "Select Keyframe JSON", "", "JSON Files (*.json)")
        if json_file:
            with open(json_file, "r") as f:
                self.keyframes = json.load(f)
            print(f"Keyframes loaded: {len(self.keyframes)} keyframes.")
            self.keyframe_json_path_label.setText(f"Keyframe JSON path: {json_file}")

    def load_map_points(self):
        map_points_file, _ = QFileDialog.getOpenFileName(self, "Select Map Points JSON", "", "JSON Files (*.json)")
        if map_points_file:
            with open(map_points_file, "r") as f:
                self.map_points = json.load(f)
            print(f"Map Points loaded: {len(self.map_points)} map points.")
            self.map_points_json_path_label.setText(f"Map Points JSON path: {map_points_file}")

    def show_image(self, index):
        if index < 0 or index > len(self.image_paths): return
        check_idx = str(index)
        path = self.image_paths[index]
        image = cv2.imread(path)
        if self.show_map_points_checkbox.isChecked() and self.map_points and check_idx in self.keyframes.keys():
            for map_point in self.map_points.values():
                if map_point['outlier']: continue
                if check_idx not in map_point['observations'].keys(): continue

                index_in_kf = map_point['observations'][check_idx]
                kp = self.keyframes[check_idx]['keypoints'][index_in_kf]
                kp = [int(_) for _ in kp]
                cv2.circle(image, kp, 3, (0, 255, 0), -1)

        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

        # Update statistics
        self.update_statistics(index)

    def update_fps(self, value):
        if self.timer.isActive():
            interval_ms = int(1000 / value)
            self.timer.setInterval(interval_ms)

    def toggle_playback(self):
        if not self.image_paths:
            return

        if self.is_playing:
            self.timer.stop()
            self.play_button.setText("Start")
        else:
            self.timer.start()
            self.play_button.setText("Stop")
        self.next_button.setEnabled(self.is_playing)
        self.prev_button.setEnabled(self.is_playing)

        self.is_playing = not self.is_playing

    def next_frame(self, direction):
        self.current_index += direction
        if self.current_index >= len(self.filtered_image_paths()): self.current_index = 0
        if self.current_index < 0: self.current_index = len(self.filtered_image_paths()) - 1
        self.show_image(self.current_index)

    def slider_changed(self, value):
        self.current_index = value
        self.show_image(value)

    def toggle_keyframe_mode(self):
        # Update the toggle based on the checkbox state
        self.show_only_keyframes = self.keyframe_checkbox.isChecked()
        
        # Reset index when changing modes
        self.current_index = 0
        self.show_image(self.current_index)

    def filtered_image_paths(self):
        if self.show_only_keyframes:
            # Filter images to only those whose timestamp matches a keyframe
            keyframe_timestamps = {str(k['timestep']) for k in self.keyframes.values()}
            return [path for path in self.image_paths if os.path.splitext(os.path.basename(path))[0] in keyframe_timestamps]
        else:
            return self.image_paths

    def update_statistics(self, index):
        # Get corresponding keyframe data from the JSON
        image_name = os.path.splitext(os.path.basename(self.image_paths[index]))[0]
        timestamp = int(image_name)
        keyframe = None
        for kf in self.keyframes.values():
            if int(kf['timestep']) == timestamp:
                keyframe = kf
                break
        self.frame_id_label.setText(f'{index}')
        if keyframe:
            timestamp_info = f"{timestamp}"
            pose_info = self.format_pose(keyframe['Twc'])
            map_point_info = self.get_map_point_info(keyframe['id'])
        else:
            timestamp_info = 'No keyframe data'
            pose_info = 'No pose data'
            map_point_info = 'No map point data'

        self.timestamp_label.setText(timestamp_info)
        self.pose_label.setText(pose_info)
        self.map_point_info_label.setText(map_point_info)

    def get_map_point_info(self, kf_id):
        # Loop through map points and find any relevant ones for the current timestamp
        map_points = 0
        outliers = 0
        for mp_data in self.map_points.values():
            # Check if the map point has observations for the current timestamp
            if str(kf_id) in mp_data['observations'].keys():
                map_points += 1
                if mp_data['outlier']:
                    outliers += 1
        percentage = 0 if map_points == 0 else 100*outliers/map_points
        return f'Number of map points {map_points} - Outlier percentage {percentage}'

    def format_pose(self, Twc):
        # Format the pose as a string or matrix (simplified for display)
        if len(Twc) == 4:
            pose = Geometry.SE3(Twc)
            vals = Geometry.SE3.Logmap(pose)
            r2d = 180 / 3.14
            return '\n'.join([
                f'Roll: {vals[0] * r2d:.2f} x: {vals[3]:.2f}', 
                f'Pitch: {vals[1] * r2d:.2f} y: {vals[4]:.2f}',
                f'Yaw: {vals[2] * r2d:.2f} z: {vals[5]:.2f}'
                ])
        else:
            return "Invalid pose data"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSequencePlayer()
    window.resize(900, 700)
    window.show()
    sys.exit(app.exec())
