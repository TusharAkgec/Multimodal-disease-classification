from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from gui.utils import pil_image_to_qpixmap


class ImageUploadWidget(QFrame):
    imageChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ImageUploadWidget")
        self._image_path: str = ""
        self._original_pixmap: Optional[QPixmap] = None

        self.title = QLabel("X-ray image")
        self.title.setObjectName("SectionTitle")

        self.btn = QPushButton("Choose image…")
        self.btn.setObjectName("PrimaryButton")
        self.btn.clicked.connect(self._choose_file)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("SecondaryButton")
        self.clear_btn.clicked.connect(self.clear)

        self.preview = QLabel("No image selected")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setObjectName("ImagePreview")
        self.preview.setMinimumHeight(220)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        top = QHBoxLayout()
        top.addWidget(self.btn, 1)
        top.addWidget(self.clear_btn, 0)

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addLayout(top)
        layout.addWidget(self.preview, 1)
        self.setLayout(layout)

    @property
    def image_path(self) -> str:
        return self._image_path

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        self.set_image(path)

    def set_image(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = Image.open(str(p)).convert("RGB")
        self._original_pixmap = pil_image_to_qpixmap(img)
        self._refresh_preview()
        self._image_path = str(p)
        self.imageChanged.emit(self._image_path)

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._original_pixmap = pixmap
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        if self._original_pixmap is None or self._original_pixmap.isNull():
            self.preview.setText("No image selected")
            self.preview.setPixmap(QPixmap())
            return
        scaled = self._original_pixmap.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview.setText("")
        self.preview.setPixmap(scaled)

    def resizeEvent(self, event):  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self._original_pixmap is not None and not self._original_pixmap.isNull():
            self._refresh_preview()

    def clear(self) -> None:
        self._image_path = ""
        self._original_pixmap = None
        self.preview.setText("No image selected")
        self.preview.setPixmap(QPixmap())
        self.imageChanged.emit("")


class ProbabilityBar(QFrame):
    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("ProbabilityBar")
        self.name = QLabel(label)
        self.name.setObjectName("ProbLabel")
        self.value = QLabel("0.0%")
        self.value.setObjectName("ProbValue")

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setObjectName("ProbProgress")

        row = QHBoxLayout()
        row.addWidget(self.name, 1)
        row.addWidget(self.value, 0)

        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addWidget(self.bar)
        self.setLayout(layout)

    def set_prob(self, prob: float) -> None:
        p = float(prob)
        p = 0.0 if p < 0 else (1.0 if p > 1.0 else p)
        self.value.setText(f"{p * 100.0:.1f}%")
        self.bar.setValue(int(round(p * 1000)))


class PredictionCard(QFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("PredictionCard")

        self.title = QLabel("Prediction")
        self.title.setObjectName("SectionTitle")

        self.label = QLabel("—")
        self.label.setObjectName("PredLabel")

        self.conf = QLabel("Confidence: —")
        self.conf.setObjectName("PredConfidence")

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.label)
        layout.addWidget(self.conf)
        self.setLayout(layout)

    def set_prediction(self, label: str, confidence: float) -> None:
        self.label.setText(str(label))
        self.conf.setText(f"Confidence: {float(confidence) * 100.0:.1f}%")

    def reset(self) -> None:
        self.label.setText("—")
        self.conf.setText("Confidence: —")

