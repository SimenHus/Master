from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QWidget, QScrollArea, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QPainter, QPen, QImage, QColor
from PySide6.QtCore import Qt, QPoint
import sys
import cv2
import numpy as np

class MaskLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.image = None
        self.qimage = None
        self.display_scale = 1.0
        self.mask_orig = None
        self.points = []
        self.polygons = []
        self.is_drawing = False

    def load_image(self, path, max_width=1200, max_height=900):
        self.image = cv2.imread(path)
        if self.image is None:
            return
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.orig_h, self.orig_w = self.image.shape[:2]

        scale_w = min(1.0, max_width / self.orig_w)
        scale_h = min(1.0, max_height / self.orig_h)
        self.display_scale = min(scale_w, scale_h)

        resized = cv2.resize(self.image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)
        self.qimage = QImage(resized.data, resized.shape[1], resized.shape[0],
                             resized.strides[0], QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(self.qimage))

        self.mask_orig = np.ones((self.orig_h, self.orig_w), dtype=np.uint8) * 255
        self.points = []
        self.polygons = []
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing and self.image is not None:
            x = int(event.pos().x() / self.display_scale)
            y = int(event.pos().y() / self.display_scale)
            # Clamp points inside image bounds
            x = max(0, min(self.orig_w - 1, x))
            y = max(0, min(self.orig_h - 1, y))
            self.points.append((x, y))
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is not None:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)

            # Draw in-progress polygon lines
            for i in range(1, len(self.points)):
                p1 = QPoint(int(self.points[i - 1][0] * self.display_scale),
                            int(self.points[i - 1][1] * self.display_scale))
                p2 = QPoint(int(self.points[i][0] * self.display_scale),
                            int(self.points[i][1] * self.display_scale))
                painter.drawLine(p1, p2)

            # Draw green mask overlay only if mask has any masked-out pixels (value == 0)
            if self.mask_orig is not None:
                # Check if there is any masked pixel (0)
                if np.any(self.mask_orig == 0):
                    # Create overlay image same size as displayed image
                    overlay = np.zeros((self.orig_h, self.orig_w, 4), dtype=np.uint8)

                    # Set green and alpha for masked pixels (mask==0)
                    overlay[self.mask_orig == 0] = [0, 255, 0, 150]  # RGBA, semi-transparent green

                    # Convert overlay to QImage, scale to display size
                    overlay_qimage = QImage(overlay.data, overlay.shape[1], overlay.shape[0],
                                            overlay.strides[0], QImage.Format_RGBA8888)
                    overlay_qimage_scaled = overlay_qimage.scaled(self.qimage.width(), self.qimage.height(),
                                                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

                    # Draw overlay on top
                    painter.drawImage(0, 0, overlay_qimage_scaled)



    def start_polygon(self):
        if not self.is_drawing:
            self.points = []
            self.is_drawing = True
            self.update()

    def finish_polygon(self):
        if self.is_drawing and len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            self.polygons.append(pts.copy())
            cv2.fillPoly(self.mask_orig, [pts], 0)
        self.points = []
        self.is_drawing = False
        self.update()

    def undo_last_polygon(self):
        if self.polygons:
            self.polygons.pop()
            self._rebuild_mask()
            self.update()

    def clear_polygons(self):
        self.polygons.clear()
        self.points.clear()
        self._rebuild_mask()
        self.update()

    def _rebuild_mask(self):
        self.mask_orig = np.ones((self.orig_h, self.orig_w), dtype=np.uint8) * 255
        for poly in self.polygons:
            cv2.fillPoly(self.mask_orig, [poly], 0)

class MaskTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Static Mask Creator")
        self.setGeometry(100, 100, 1280, 960)

        self.label = MaskLabel()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.label)

        # Buttons
        self.load_button = QPushButton("Load Image")
        self.save_button = QPushButton("Save Mask")
        self.start_button = QPushButton("Start Polygon")
        self.finish_button = QPushButton("Finish Polygon")
        self.undo_button = QPushButton("Undo Last Region")
        self.clear_button = QPushButton("Clear Regions")

        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_mask)
        self.start_button.clicked.connect(self.label.start_polygon)
        self.finish_button.clicked.connect(self.label.finish_polygon)
        self.undo_button.clicked.connect(self.label.undo_last_polygon)
        self.clear_button.clicked.connect(self.label.clear_polygons)

        button_layout = QHBoxLayout()
        for btn in [self.load_button, self.save_button, self.start_button,
                    self.finish_button, self.undo_button, self.clear_button]:
            button_layout.addWidget(btn)

        layout = QVBoxLayout()
        layout.addLayout(button_layout)
        layout.addWidget(self.scroll)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_path:
            self.label.load_image(file_path)

    def save_mask(self):
        if self.label.mask_orig is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Mask", "static_mask.png", "PNG Files (*.png)")
            if file_path:
                cv2.imwrite(file_path, self.label.mask_orig)
                print(f"✅ Mask saved to: {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskTool()
    window.show()
    sys.exit(app.exec())
