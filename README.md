# x-ray-tube-spectra-generation

A Python-based web app for generating and visualizing X-ray tube spectra using the **Ebel (1999)** model.  
This project integrates `xraylib` for precise X-ray physics calculations.

## 🧩 Features
- Computes X-ray tube spectra based on user input parameters.
- Web interface built with Flask (`templates/` and `static/`).
- Modular structure (`ebel_calculations.py` handles physics).

## ⚙️ Requirements
Install dependencies:
```bash
pip install -r requirements.txt