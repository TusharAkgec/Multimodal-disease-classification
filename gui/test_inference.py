from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from gui.inference import get_engine
    from gui.preprocess import prepare_inputs

    engine = get_engine()

    # Try to locate a sample image under data/image/
    sample_dir = project_root / "data" / "image"
    if not sample_dir.exists():
        print("No sample directory found at data/image; skipping prediction.")
        return

    candidates = sorted([p for p in sample_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not candidates:
        print("No sample images found; skipping prediction.")
        return

    image_path = str(candidates[0])
    image_t, meta_t = prepare_inputs(image_path, age=50.0, gender="Male", view="AP")
    probs, top_idx, top_prob = engine.predict(image_t, meta_t)
    print("OK:", engine.class_names[top_idx], f"{top_prob:.4f}", "from", Path(image_path).name)


if __name__ == "__main__":
    main()

