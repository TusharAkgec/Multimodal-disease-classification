from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QIntValidator
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from gui.utils import format_probabilities, pil_image_to_qpixmap
from gui.widgets import ImageUploadWidget, PredictionCard, ProbabilityBar

if TYPE_CHECKING:
    from gui.inference import InferenceEngine


def _load_styles(app_widget: QWidget) -> None:
    qss_path = Path(__file__).resolve().parent / "assets" / "styles.qss"
    if qss_path.exists():
        app_widget.setStyleSheet(qss_path.read_text(encoding="utf-8"))


class _InferenceWorker(QObject):
    finished = pyqtSignal(object, int, float)  # probs, top_idx, top_prob
    failed = pyqtSignal(str)

    def __init__(self, engine: InferenceEngine, image_path: str, age: int, gender: str, view: str):
        super().__init__()
        self.engine = engine
        self.image_path = image_path
        self.age = age
        self.gender = gender
        self.view = view

    def run(self) -> None:
        try:
            from gui.preprocess import prepare_inputs

            image_t, meta_t = prepare_inputs(
                image_path=self.image_path,
                age=float(self.age),
                gender=self.gender,
                view=self.view,
            )
            probs, top_idx, top_prob = self.engine.predict(
                image_t,
                meta_t,
                image_path=self.image_path,
            )
            self.finished.emit(probs, int(top_idx), float(top_prob))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    def __init__(self, engine: InferenceEngine):
        super().__init__()
        self.engine = engine
        self.class_names: List[str] = list(engine.class_names)

        self.setWindowTitle("CXR Multimodal Inference")
        self.setMinimumSize(1040, 680)

        central = QWidget()
        central.setObjectName("AppRoot")
        _load_styles(central)
        self.setCentralWidget(central)

        self._thread: Optional[QThread] = None
        self._worker: Optional[_InferenceWorker] = None

        # Header
        self.h_title = QLabel("CXR Multimodal Inference")
        self.h_title.setObjectName("HeaderTitle")
        self.h_subtitle = QLabel("Upload an image, enter metadata, then analyze.")
        self.h_subtitle.setObjectName("HeaderSubtitle")
        self.h_status = QLabel("")
        self.h_status.setObjectName("HeaderStatus")

        header = QVBoxLayout()
        header.addWidget(self.h_title)
        header.addWidget(self.h_subtitle)
        header.addWidget(self.h_status)

        header_frame = QFrame()
        header_frame.setObjectName("HeaderFrame")
        header_frame.setLayout(header)

        # Left panel (inputs)
        self.image_widget = ImageUploadWidget()

        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age (1–120)")
        self.age_input.setValidator(QIntValidator(1, 120, self))
        self.age_input.setObjectName("InputField")

        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["Select gender", "Male", "Female"])
        self.gender_combo.setObjectName("ComboField")

        self.view_combo = QComboBox()
        self.view_combo.addItems(["Select view", "AP", "PA", "Lateral"])
        self.view_combo.setObjectName("ComboField")

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setObjectName("PrimaryButton")
        self.analyze_btn.clicked.connect(self.on_analyze_clicked)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("SecondaryButton")
        self.reset_btn.clicked.connect(self.on_reset_clicked)

        left = QVBoxLayout()
        left.addWidget(self.image_widget)
        left.addWidget(QLabel("Age"))
        left.addWidget(self.age_input)
        left.addWidget(QLabel("Gender"))
        left.addWidget(self.gender_combo)
        left.addWidget(QLabel("View"))
        left.addWidget(self.view_combo)
        left.addSpacing(10)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.analyze_btn, 1)
        btn_row.addWidget(self.reset_btn, 0)
        left.addLayout(btn_row)
        left.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        left_frame = QFrame()
        left_frame.setObjectName("LeftPanel")
        left_frame.setLayout(left)

        # Right panel (results)
        self.pred_card = PredictionCard()

        self.fusion_label = QLabel("")
        self.fusion_label.setObjectName("FusionLabel")
        self._update_fusion_label()

        self.prob_bars: List[ProbabilityBar] = [ProbabilityBar(n) for n in self.class_names]
        prob_container = QWidget()
        prob_layout = QVBoxLayout()
        prob_layout.setContentsMargins(0, 0, 0, 0)
        prob_layout.setSpacing(10)
        for b in self.prob_bars:
            prob_layout.addWidget(b)
        prob_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        prob_container.setLayout(prob_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("ProbScroll")
        scroll.setWidget(prob_container)

        self.heatmap_btn = QPushButton("Show heatmap")
        self.heatmap_btn.setObjectName("TertiaryButton")
        self.heatmap_btn.clicked.connect(self.on_heatmap_clicked)
        self.heatmap_btn.setEnabled(False)

        self._gradcam_available = self._check_gradcam_available()
        if not self._gradcam_available:
            self.heatmap_btn.setVisible(False)

        right = QVBoxLayout()
        right.addWidget(self.pred_card)
        right.addWidget(self.fusion_label)
        right.addWidget(QLabel("Class probabilities"))
        right.addWidget(scroll, 1)
        right.addWidget(self.heatmap_btn)

        right_frame = QFrame()
        right_frame.setObjectName("RightPanel")
        right_frame.setLayout(right)

        # Main split
        body = QHBoxLayout()
        body.addWidget(left_frame, 0)
        body.addWidget(right_frame, 1)
        body.setSpacing(16)

        root = QVBoxLayout()
        root.addWidget(header_frame)
        root.addLayout(body, 1)
        central.setLayout(root)

        self._last_probs: Optional[np.ndarray] = None
        self._last_top_idx: Optional[int] = None
        self._last_inputs: Optional[Tuple[str, int, str, str]] = None

        self.on_reset_clicked()

    def _check_gradcam_available(self) -> bool:
        try:
            import gui.gradcam  # noqa: F401

            return True
        except Exception:
            return False

    def _set_busy(self, busy: bool) -> None:
        self.analyze_btn.setEnabled(not busy)
        self.reset_btn.setEnabled(not busy)
        self.image_widget.btn.setEnabled(not busy)
        self.image_widget.clear_btn.setEnabled(not busy)
        self.age_input.setEnabled(not busy)
        self.gender_combo.setEnabled(not busy)
        self.view_combo.setEnabled(not busy)
        if busy:
            self.h_status.setText("Analyzing…")
        else:
            self.h_status.setText("")

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_warning(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def _validate_inputs(self) -> Tuple[str, int, str, str]:
        image_path = self.image_widget.image_path.strip()
        if not image_path:
            raise ValueError("Please select an image.")
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        age_text = self.age_input.text().strip()
        if not age_text:
            raise ValueError("Please enter age.")
        try:
            age = int(age_text)
        except Exception:
            raise ValueError("Age must be a number.")
        if age < 1 or age > 120:
            raise ValueError("Age must be between 1 and 120.")

        if self.gender_combo.currentIndex() <= 0:
            raise ValueError("Please select gender.")
        gender = self.gender_combo.currentText().strip()

        if self.view_combo.currentIndex() <= 0:
            raise ValueError("Please select view.")
        view = self.view_combo.currentText().strip()

        return image_path, age, gender, view

    def _update_fusion_label(self) -> None:
        w = self.engine.get_fusion_weight()
        if w is None:
            self.fusion_label.setText("")
            return
        self.fusion_label.setText(f"Fusion weight: {w:.4f}")

    def on_analyze_clicked(self) -> None:
        try:
            image_path, age, gender, view = self._validate_inputs()
        except Exception as e:
            self._show_error("Invalid input", str(e))
            return

        if self._thread is not None:
            return

        self._set_busy(True)
        self.heatmap_btn.setEnabled(False)

        self._thread = QThread()
        self._worker = _InferenceWorker(self.engine, image_path, age, gender, view)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_inference_finished)
        self._worker.failed.connect(self._on_inference_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

        self._last_inputs = (image_path, age, gender, view)

    def _cleanup_thread(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)

    def _on_inference_failed(self, msg: str) -> None:
        self._show_error("Inference failed", msg)

    def _on_inference_finished(self, probs_obj: object, top_idx: int, top_prob: float) -> None:
        probs = np.asarray(probs_obj, dtype=np.float32).reshape(-1)
        if probs.size != len(self.class_names):
            self._show_error(
                "Inference error",
                f"Model returned {probs.size} probabilities but UI expects {len(self.class_names)} classes.",
            )
            return
        self._last_probs = probs
        self._last_top_idx = int(top_idx)

        pred_label = self.class_names[int(top_idx)]
        self.pred_card.set_prediction(pred_label, float(top_prob))

        formatted = format_probabilities(probs, self.class_names)
        for bar, (_, p, _) in zip(self.prob_bars, formatted):
            bar.set_prob(p)

        self._update_fusion_label()

        # After rendering prediction, clear only the input fields (partial reset).
        self.clear_inputs_after_prediction()

        if self._gradcam_available and self._last_inputs is not None:
            self.heatmap_btn.setEnabled(True)

    def clear_inputs_after_prediction(self) -> None:
        """
        Clear user inputs (image + metadata) while keeping prediction results visible.
        """
        if hasattr(self, "image_widget") and self.image_widget is not None:
            self.image_widget.clear()
        if hasattr(self, "age_input") and self.age_input is not None:
            self.age_input.setText("")
        if hasattr(self, "gender_combo") and self.gender_combo is not None:
            self.gender_combo.setCurrentIndex(0)
        if hasattr(self, "view_combo") and self.view_combo is not None:
            self.view_combo.setCurrentIndex(0)

    def on_heatmap_clicked(self) -> None:
        if not self._gradcam_available:
            self._show_warning("Unavailable", "Grad-CAM is not available in this build.")
            return
        if self._last_inputs is None or self._last_top_idx is None:
            self._show_warning("No result", "Run Analyze first.")
            return

        image_path, age, gender, view = self._last_inputs
        try:
            from gui.gradcam import generate_heatmap
            from gui.preprocess import prepare_inputs

            image_t, meta_t = prepare_inputs(image_path, float(age), gender, view)
            overlay = generate_heatmap(self.engine.model, image_t, meta_t, int(self._last_top_idx))
            pix = pil_image_to_qpixmap(overlay)
            self.image_widget.set_preview_pixmap(pix)
        except Exception as e:
            self._show_warning("Heatmap failed", f"{type(e).__name__}: {e}")

    def on_reset_clicked(self) -> None:
        self.image_widget.clear()
        self.age_input.setText("")
        self.gender_combo.setCurrentIndex(0)
        self.view_combo.setCurrentIndex(0)
        self.pred_card.reset()
        for b in self.prob_bars:
            b.set_prob(0.0)
        self._last_probs = None
        self._last_top_idx = None
        self._last_inputs = None
        self.heatmap_btn.setEnabled(False)
        self._update_fusion_label()

