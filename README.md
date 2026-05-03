 Multimodal Disease Classification

A machine learning project that predicts chest diseases using:

🩻 Chest X-ray images
🧾 Patient metadata (age, sex, etc.)


🚀 Features
Combines image + clinical data (multimodal AI)
Uses deep learning models (CNN + MLP)
Provides disease predictions
Interactive UI using Streamlit


 How to Run
1. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run gui/streamlit_app.py


📁 Project Structure
├── gui/
│   └── streamlit_app.py
├── models/
├── data/
├── notebooks/
├── requirements.txt
└── README.md

 Model Overview
CNN → extracts features from X-ray images
MLP → processes patient metadata
Fusion → combines both for better prediction

 Note
Dataset and model files are not included (for size reasons)
Add your own dataset before running

 Tech Stack
Python
PyTorch / TensorFlow
Streamlit
NumPy, Pandas
