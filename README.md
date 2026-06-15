# ESCO Knowledge Graph Application

## Launch the application

### **Easiest methods:**

#### **1. Batch file (Recommended):**
```
Double-click on: start_app.bat
```

#### **2. Python script:**
```
python start_app.py
```

#### **3. Directly via Python:**
```
python -m streamlit run esco_kg_streamlit.py
```

#### **4. PowerShell:**
```
.\start_app.ps1
```

### **Create a desktop shortcut:**
```
Double-click on: create_shortcut.bat
```

## Prerequisites

Make sure the following packages are installed:
```bash
pip install streamlit pandas scikit-learn beautifulsoup4
```

## Troubleshooting

### **Problem: "streamlit" not found**
**Solution:** Use `python -m streamlit` instead of just `streamlit`

### **Problem: Missing dependencies**
**Solution:** Run: `pip install -r requirements.txt`

## Project structure

```
Prototype_qualification_assistant/
├── esco_kg_streamlit.py     # Main application
├── start_app.bat            # Batch Starter
├── start_app.ps1            # PowerShell Starter
├── start_app.py             # Python Starter
├── create_shortcut.bat      # Create a desktop shortcut
├── README.md                # This file
└── data/                    # Data folder
    ├── employees_data.csv
    ├── courses.csv
    └── ...
```

## Features

- 👥 Employee Management
- 📊 Competency Profiles
- 🔍 Job Matching
- 📚 Course Recommendations
- 💾 Persistent Data Storage 