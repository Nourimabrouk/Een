#!/usr/bin/env python3
"""
Een | Unity Mathematics - Master Dashboard (Streamlit Cloud Entry Point)
Main entry point for Streamlit Cloud deployment: https://een-unity-mathematics.streamlit.app

This file automatically launches the most complete and recent Unity Mathematics dashboard.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import and run the master Unity Mathematics dashboard
if __name__ == "__main__":
    try:
        # Import the main dashboard from src/unity_mathematics_streamlit.py
        from unity_mathematics_streamlit import *
        
        # The streamlit app will automatically run when this file is executed
        print("🚀 Een Unity Mathematics Dashboard launched successfully!")
        print("📊 Running most complete implementation with φ-harmonic resonance")
        print("🌟 Access at: https://een-unity-mathematics.streamlit.app")
        
    except ImportError as e:
        import streamlit as st
        
        st.error(f"⚠️ Import Error: {e}")
        st.error("Please ensure all dependencies are installed.")
        st.markdown("""
        ### Missing Dependencies
        
        Run the following command to install required packages:
        ```bash
        pip install -r requirements.txt
        ```
        
        ### Manual Launch
        If this continues, you can manually launch with:
        ```bash
        streamlit run src/unity_mathematics_streamlit.py
        ```
        """)
        
    except Exception as e:
        import streamlit as st
        
        st.error(f"🚨 Unexpected Error: {e}")
        st.markdown("""
        ### Fallback Information
        
        **Primary Dashboard Location:** `src/unity_mathematics_streamlit.py`
        
        **Manual Commands:**
        ```bash
        # Activate environment
        conda activate een
        
        # Run dashboard
        streamlit run src/unity_mathematics_streamlit.py
        ```
        
        **Streamlit Cloud URL:** https://een-unity-mathematics.streamlit.app
        """)