#!/usr/bin/env python3
"""
Een | Unity Mathematics - Master Dashboard (Streamlit Cloud Entry Point)
Main entry point for Streamlit Cloud deployment: https://een-unity-mathematics.streamlit.app

This file automatically launches the most complete and recent Unity Mathematics dashboard.
Optimized for Streamlit Cloud resource limits and inotify constraints.
"""

import sys
import os
from pathlib import Path

# Optimize for Streamlit Cloud - minimal path manipulation
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Set environment variables for optimization
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Import and run the master Unity Mathematics dashboard
if __name__ == "__main__":
    try:
        # Import the COMPREHENSIVE METASTATION dashboard from src/unity_mathematics_streamlit.py
        from unity_mathematics_streamlit import *
        
        # The comprehensive 2467-line dashboard will automatically run
        print("SUCCESS: METASTATION HUD launched - Full Unity Mathematics Command Center")
        print("FEATURES: Multi-page, consciousness field, quantum proofs, phi-spirals, HUD interface")
        print("ACCESS: https://een-unity-mathematics.streamlit.app")
        
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