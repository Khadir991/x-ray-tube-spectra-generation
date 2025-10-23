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

## 🚀 Run the app
python app.py

Then open http://127.0.0.1:5000/ in your browser.

## 📁 Project structure

XRAYTUBE2.0/
├── app.py
├── ebel_calculations.py
├── static/
│   └── js/
│       └── main.js
├── templates/
│   └── index.html
└── requirements.txt


## 🧠 Reference

Ebel, H. (1999). Analytical model for X-ray tube spectra. X-Ray Spectrometry, 28, 255–266.

## 📜 License

MIT License


---

### `.gitignore`
```gitignore
__pycache__/
*.pyc
venv/
.env/
.DS_Store
