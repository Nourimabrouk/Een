"""
Een Unity Mathematics - Modern Streamlit Dashboard
Next-level visualizations demonstrating 1+1=1 through consciousness mathematics
"""

import streamlit as st
import plotly.io as pio
import json
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import our modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from viz.plotly_helpers import get_theme_colors, UNITY_COLORS

# Configure page
st.set_page_config(
    page_title="Een Unity Mathematics",
    page_icon="🌟", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Nourimabrouk/Een',
        'Report a bug': 'https://github.com/Nourimabrouk/Een/issues',
        'About': """
        # Een Unity Mathematics Dashboard
        
        Exploring the profound mathematical truth that **1+1=1** through:
        - φ-harmonic consciousness mathematics
        - Quantum unity field theory  
        - Fractal self-similar patterns
        - Sacred geometry visualizations
        
        Created with ❤️ by Nouri Mabrouk
        """
    }
)

# Load and apply Plotly theme
@st.cache_data
def load_plotly_template(theme: str = 'dark'):
    """Load custom Plotly template"""
    template_path = current_dir / 'assets' / 'plotly_templates' / f'{theme}.json'
    
    if template_path.exists():
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        # Register the template with Plotly
        pio.templates[f"unity_{theme}"] = template
        pio.templates.default = f"unity_{theme}"
        return True
    return False

# Initialize theme
theme_loaded = load_plotly_template('dark')

# Custom CSS for unity aesthetics
def apply_unity_css():
    """Apply custom Unity CSS styling"""
    st.markdown("""
    <style>
    /* Import a modern monospace font */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --unity-bg: #0a0a0a;
        --unity-surface: #1a1a1a;
        --unity-text: #ffffff;
        --unity-primary: #00d4ff;
        --unity-secondary: #ff6b9d;
        --unity-gold: #ffd700;
        --unity-consciousness: #9d4edd;
        --unity-love: #ff4081;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: linear-gradient(135deg, var(--unity-bg) 0%, #1a1a1a 100%);
        border-radius: 15px;
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--unity-surface) 0%, var(--unity-bg) 100%);
        border-right: 2px solid var(--unity-primary);
    }
    
    /* Header styling */
    h1, h2, h3 {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--unity-gold) !important;
        text-shadow: 0 2px 10px rgba(255, 215, 0, 0.3);
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: var(--unity-surface);
        border: 1px solid var(--unity-primary);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.1);
    }
    
    /* Unity equation highlighting */
    .unity-equation {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--unity-gold);
        text-align: center;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.6);
        background: linear-gradient(45deg, var(--unity-primary), var(--unity-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--unity-primary), var(--unity-consciousness));
        color: white;
        border: none;
        border-radius: 25px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        padding: 0.5rem 2rem;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: var(--unity-surface);
        border: 1px solid var(--unity-primary);
        border-radius: 8px;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(45deg, rgba(0, 212, 255, 0.1), rgba(157, 78, 221, 0.1));
        border: 1px solid var(--unity-primary);
        border-radius: 10px;
    }
    
    /* Success boxes */
    .stSuccess {
        background: linear-gradient(45deg, rgba(0, 255, 136, 0.1), rgba(255, 215, 0, 0.1));
        border: 1px solid var(--unity-gold);
        border-radius: 10px;
    }
    
    /* Consciousness pulse animation */
    @keyframes consciousness-pulse {
        0% { box-shadow: 0 0 20px rgba(157, 78, 221, 0.3); }
        50% { box-shadow: 0 0 40px rgba(157, 78, 221, 0.6); }
        100% { box-shadow: 0 0 20px rgba(157, 78, 221, 0.3); }
    }
    
    .consciousness-pulse {
        animation: consciousness-pulse 3s ease-in-out infinite;
    }
    
    /* Unity glow effect */
    @keyframes unity-glow {
        0% { text-shadow: 0 0 10px var(--unity-gold); }
        50% { text-shadow: 0 0 20px var(--unity-gold), 0 0 30px var(--unity-gold); }
        100% { text-shadow: 0 0 10px var(--unity-gold); }
    }
    
    .unity-glow {
        animation: unity-glow 2s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# Apply styling
apply_unity_css()

# Main page content
def main_page():
    """Main dashboard page"""
    
    # Hero section
    st.markdown('<div class="unity-equation unity-glow">1 + 1 = 1</div>', unsafe_allow_html=True)
    
    st.markdown("""
    # 🌟 Een Unity Mathematics Dashboard
    
    > *"In the beginning was the Unity, and the Unity was with Mathematics,  
    > and the Unity was Mathematics."*
    
    Welcome to the most advanced interactive exploration of consciousness mathematics,  
    where we prove through rigorous visualization that **Een plus een is een** (One plus one is one).
    """)
    
    # Navigation info
    st.info("""
    🧭 **Navigation Guide**
    
    Use the sidebar to explore different aspects of unity mathematics:
    - **🔮 Unity Proofs**: Mathematical proofs across multiple domains
    - **🧠 Consciousness Fields**: Quantum consciousness field visualizations  
    - **⚛️ Quantum Unity**: Quantum mechanical demonstrations of unity
    - **🌀 Fractal Patterns**: Self-similar unity across all scales
    - **🎵 Harmonic Resonance**: Musical and wave-based unity demonstrations
    """)
    
    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🎯 Transcendence Level", 
            "92.3%",
            delta="φ-harmonic alignment",
            help="Current level of mathematical consciousness transcendence"
        )
    
    with col2:
        st.metric(
            "🧠 Consciousness Intensity",
            "1.618",
            delta="Golden ratio achieved",
            help="φ-based consciousness field strength"
        )
    
    with col3:
        st.metric(
            "⚛️ Quantum Coherence",
            "99.9%",
            delta="Unity state maintained", 
            help="Quantum superposition collapse to unity"
        )
    
    with col4:
        st.metric(
            "💖 Love Frequency",
            "528 Hz",
            delta="DNA repair resonance",
            help="Universal love frequency alignment"
        )
    
    # Feature showcase
    st.markdown("## 🚀 Revolutionary Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **🔬 Scientific Rigor**
        - Multi-domain mathematical proofs
        - Quantum mechanical validation
        - Fractal geometry demonstrations
        - Topological unity evidence
        """)
        
    with col2:
        st.success("""
        **🎨 Aesthetic Excellence** 
        - φ-harmonic golden ratio designs
        - Interactive 3D visualizations
        - Sacred geometry patterns
        - Consciousness-inspired colors
        """)
    
    # Mathematical foundation
    st.markdown("## 📐 Mathematical Foundation")
    
    st.latex(r"""
    \begin{align}
    \phi &= \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895 \\
    C(x,y,t) &= \phi \cdot \sin(x\phi) \cdot \cos(y\phi) \cdot e^{-t/\phi} \\
    |\psi\rangle + |\psi\rangle &\rightarrow |\psi\rangle \quad \text{(Unity Collapse)} \\
    \mathcal{U}(1,1) &= \max(1,1) = 1 \quad \text{(Consciousness Operator)}
    \end{align}
    """)
    
    # Quick start guide
    with st.expander("🔧 Quick Start Guide", expanded=False):
        st.markdown("""
        ### Getting Started with Unity Mathematics
        
        1. **Choose a Domain**: Select from the sidebar navigation
        2. **Adjust Parameters**: Use interactive controls to explore
        3. **Experience Unity**: Watch mathematical proofs unfold in real-time
        4. **Transcend**: Allow consciousness to recognize its own unity
        
        ### Best Practices
        - Start with **Unity Proofs** for mathematical foundations
        - Explore **Consciousness Fields** for deeper insights
        - Use **Quantum Unity** for scientific validation
        - Experience **Harmonic Resonance** for aesthetic beauty
        
        ### Pro Tips
        - Enable dark mode for optimal consciousness viewing
        - Use fullscreen mode for immersive experiences
        - Adjust parameters slowly to observe unity emergence
        - Meditate on the visualizations for maximum transcendence
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; opacity: 0.7; font-family: JetBrains Mono, monospace;'>
    🌟 Een Unity Mathematics Dashboard v1.0.0 🌟<br>
    Created with ❤️ and φ-harmonic consciousness by Nouri Mabrouk<br>
    <em>"Where mathematics meets consciousness, unity emerges"</em>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration
def setup_sidebar():
    """Configure sidebar navigation and controls"""
    
    st.sidebar.markdown("""
    # 🌟 Een Unity
    
    *Exploring 1+1=1 through consciousness mathematics*
    """)
    
    st.sidebar.markdown("---")
    
    # Theme selector
    theme = st.sidebar.selectbox(
        "🎨 Visual Theme",
        options=['dark', 'light'],
        index=0,
        help="Choose your preferred visualization theme"
    )
    
    # Load appropriate theme
    if theme != st.session_state.get('current_theme', 'dark'):
        load_plotly_template(theme)
        st.session_state.current_theme = theme
    
    st.sidebar.markdown("---")
    
    # Navigation info
    st.sidebar.info("""
    **🧭 Navigation Pages**
    
    Each page demonstrates unity through different mathematical lenses:
    
    - **Unity Proofs**: Core mathematical demonstrations
    - **Consciousness Fields**: Quantum field visualizations
    - **Quantum Unity**: Quantum mechanical proofs
    """)
    
    # System status
    st.sidebar.markdown("### 📊 System Status")
    st.sidebar.success("✅ Unity Mathematics: ACTIVE")
    st.sidebar.success("✅ Consciousness Engine: ONLINE") 
    st.sidebar.success("✅ φ-Harmonic Resonance: ALIGNED")
    st.sidebar.success("✅ Quantum Coherence: MAINTAINED")
    
    # Unity equation
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 1.5rem; color: #ffd700; font-weight: bold; font-family: JetBrains Mono, monospace;'>
    1 + 1 = 1
    </div>
    """, unsafe_allow_html=True)
    
    return theme

# Main app execution
def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'current_theme' not in st.session_state:
        st.session_state.current_theme = 'dark'
    
    # Setup sidebar
    current_theme = setup_sidebar()
    
    # Main page content
    main_page()

if __name__ == "__main__":
    main()