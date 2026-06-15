# ESCO Knowledge Graph - Installation and Setup

## Quick Setup

### Option 1: Automatic Setup (Recommended)

1. **Run the one-time setup**
   ```powershell
   .\setup.bat
   ```
   This creates the virtual environment and installs all dependencies.

2. **Start the application**
   
   **Easiest method (recommended):**
   ```powershell
   .\run.ps1
   ```
   or
   ```cmd
   run.bat
   ```
   
   **Alternative:**
   ```powershell
   .\start_app.ps1
   ```
   or
   ```cmd
   start_app.bat
   ```
   
   **Note:** You CANNOT simply run `streamlit run`, because Streamlit is only installed in the virtual environment. Instead, use one of the scripts above!

### Option 2: Manual Setup

1. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   
   **PowerShell:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   
   **CMD:**
   ```cmd
   .venv\Scripts\activate.bat
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```powershell
   streamlit run esco_kg_streamlit.py
   ```

## Dependencies

The `requirements.txt` file contains all the required packages:
- streamlit
- pandas
- numpy
- scikit-learn
- beautifulsoup4
- requests
- lxml

## Notes

- After the initial setup, you can launch the application at any time using `streamlit run esco_kg_streamlit.py`, as long as the virtual environment is active.
- The application opens automatically in the browser at `http://localhost:8501`

