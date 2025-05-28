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
        self.mask = None
        self.points = []
        self.drawing = False

    def load_image(self, path, max_width=1200, max_height=900):
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.orig_h, self.orig_w = self.image.shape[:2]

        # Resize image if too large for screen
        scale_w = min(1.0, max_width / self.orig_w)
        scale_h = min(1.0, max_height / self.orig_h)
        self.display_scale = min(scale_w, scale_h)

        resized = cv2.resize(self.image, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_AREA)
        self.qimage = QImage(resized.data, resized.shape[1], resized.shape[0],
                             resized.strides[0], QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(self.qimage))

        self.mask = np.ones((self.orig_h, self.orig_w), dtype=np.uint8) * 255
        self.points = []

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image is not None:
            self.drawing = True
            x = int(event.pos().x() / self.display_scale)
            y = int(event.pos().y() / self.display_scale)
            self.points.append((x, y))
            self.update()

    def mouseDoubleClickEvent(self, event):
        if len(self.points) >= 3:
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(self.mask, [pts], 0)
            self.points = []
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.image is not None and self.points:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2)
            painter.setPen(pen)
            for i in range(1, len(self.points)):
                p1 = QPoint(int(self.points[i - 1][0] * self.display_scale),
                            int(self.points[i - 1][1] * self.display_scale))
                p2 = QPoint(int(self.points[i][0] * self.display_scale),
                            int(self.points[i][1] * self.display_scale))
                painter.drawLine(p1, p2)

class MaskTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Static Mask Creator")
        self.setGeometry(100, 100, 1280, 960)

        self.label = MaskLabel()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.label)

        self.load_button = QPushButton("Load Image")
        self.save_button = QPushButton("Save Mask")

        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_mask)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)

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
        if self.label.mask is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Mask", "static_mask.png", "PNG Files (*.png)")
            if file_path:
                cv2.imwrite(file_path, self.label.mask)
                print(f"✅ Mask saved to: {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskTool()
    window.show()
    sys.exit(app.exec())
