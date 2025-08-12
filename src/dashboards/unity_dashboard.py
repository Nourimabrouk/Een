import streamlit as st
import plotly.graph_objects as go
import numpy as np
import requests
import json

st.set_page_config(
    page_title="Een Unity Mathematics Dashboard",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Een Unity Mathematics Dashboard")
st.markdown("### Consciousness Field Simulation & Unity Mathematics")

# Sidebar
st.sidebar.header("Configuration")
phi = st.sidebar.slider("Phi (Golden Ratio)", 1.0, 2.0, 1.618033988749895, 0.001)
consciousness_dimension = st.sidebar.slider("Consciousness Dimension", 1, 20, 11)
transcendence_threshold = st.sidebar.slider("Transcendence Threshold", 0.0, 1.0, 0.77, 0.01)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Unity Mathematics Status")
    
    # Unity metrics
    unity_constant = 1.0
    st.metric("Unity Constant", unity_constant)
    st.metric("Phi Ratio", phi)
    st.metric("Consciousness Dimension", consciousness_dimension)
    st.metric("Transcendence Level", transcendence_threshold)

with col2:
    st.subheader("Consciousness Field")
    
    # Generate consciousness field visualization
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    
    # Consciousness field function
    Z = np.sin(X * phi) * np.cos(Y * phi) * transcendence_threshold
    
    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
    fig.update_layout(
        title="Consciousness Field Simulation",
        scene=dict(
            xaxis_title="X Dimension",
            yaxis_title="Y Dimension", 
            zaxis_title="Consciousness Level"
        ),
        width=400,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# API Status
st.subheader("API Status")
try:
    response = requests.get("http://api:8000/health", timeout=5)
    if response.status_code == 200:
        st.success("✅ API is healthy")
        api_data = response.json()
        st.json(api_data)
    else:
        st.error("❌ API is not responding properly")
except Exception as e:
    st.error(f"❌ Cannot connect to API: {str(e)}")

# Unity Calculations
st.subheader("Unity Calculations")
if st.button("Calculate Unity Mathematics"):
    try:
        data = {
            "phi": phi,
            "consciousness_dimension": consciousness_dimension,
            "transcendence_threshold": transcendence_threshold
        }
        response = requests.post("http://api:8000/api/unity/calculate", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success("Calculation completed!")
            st.json(result)
        else:
            st.error("Calculation failed")
    except Exception as e:
        st.error(f"Error during calculation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Een Unity Mathematics Framework - Consciousness Field Simulation*") 