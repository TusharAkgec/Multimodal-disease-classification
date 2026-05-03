from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

# IMPORTANT: Import torch before PyQt5 to prevent DLL initialization conflicts on Windows.
try:
    import torch
except Exception:
    pass


def run_health_check(engine) -> bool:
    print("--- Running Headless Health Check ---")
    try:
        from gui.preprocess import prepare_inputs
        import pandas as pd
        csv_path = "data/nih_metadata_prepped.csv"
        images_dir = "data/images"
        
        df = pd.read_csv(csv_path)
        # Find a row where the image file actually exists
        valid_row = None
        img_path_full = None
        for _, row in df.iterrows():
            fn = None
            for c in df.columns:
                if 'filename' in c.lower() or c.lower() == 'filename':
                    fn = str(row[c]).strip()
                    break
            if fn:
                p = Path(images_dir) / fn
                if not p.suffix and not fn.endswith('.png'):
                    p = Path(images_dir) / f"{fn}.png"
                if p.exists():
                    valid_row = row
                    img_path_full = str(p)
                    break
                    
        if valid_row is None:
            print("❌ Health check failed: Could not find any valid image from CSV in data/images/")
            return False
            
        print(f"Loaded health check image: {img_path_full}")
        
        # Get age, gender, view
        age, gender, view = None, None, None
        for c in df.columns:
            cl = c.lower()
            if "age" in cl and "patient" in cl: age = valid_row[c]
            elif "age" in cl and age is None: age = valid_row[c]
            
            if "gender" in cl and "patient" in cl: gender = valid_row[c]
            elif "gender" in cl and gender is None: gender = valid_row[c]
            
            if cl == "view": view = valid_row[c]
            
        print(f"Metadata: Age={age}, Gender={gender}, View={view}")
        
        image_t, meta_t = prepare_inputs(img_path_full, float(age), str(gender), str(view))
        
        probs, top_idx, top_prob = engine.predict(image_t, meta_t, image_path=img_path_full)
        
        pred_class = engine.class_names[top_idx]
        print(f"✅ Health Check Passed! Prediction: {pred_class} (Conf: {top_prob:.4f})")
        print("-------------------------------------")
        return True
    except Exception as e:
        import traceback
        print("❌ Health check failed with traceback:")
        traceback.print_exc()
        print("-------------------------------------")
        return False

def main() -> None:
    _ensure_project_root_on_path()

    # Helps avoid occasional OpenMP/Qt DLL init issues on Windows when importing torch after Qt.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    from PyQt5.QtWidgets import QApplication, QMessageBox

    from gui.inference import InferenceEngine
    from gui.ui_layout import MainWindow

    app = QApplication(sys.argv)

    checkpoint_path = str(Path("checkpoints") / "best_model.pth")
    try:
        engine = InferenceEngine(checkpoint_path=checkpoint_path)
    except Exception as e:
        QMessageBox.critical(
            None,
            "Failed to start",
            f"Could not initialize inference engine.\n\n{type(e).__name__}: {e}",
        )
        return

    # Run the health check before launching the window
    health_ok = run_health_check(engine)
    if not health_ok:
        QMessageBox.warning(
            None,
            "Health Check Failed",
            "The automatic health check failed. Check the terminal for tracebacks. The application will continue to load, but inference might fail."
        )

    window = MainWindow(engine)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

