/**
 * Unified Navigation Configuration for Unity Mathematics Website
 * Centralizes all navigation structure and ensures consistency across pages
 */

window.UnityNavConfig = {
    // Site metadata
    siteName: "Unity Mathematics Institute",
    siteTagline: "Where 1+1=1",
    baseUrl: "",
    
    // Primary navigation structure - Journey-based storytelling
    primaryNav: [
        {
            title: "Home",
            href: "metastation-hub.html",
            icon: "fas fa-home",
            description: "Unity Mathematics Institute hub"
        },
        {
            title: "Discover",
            icon: "fas fa-lightbulb",
            description: "What is Unity Mathematics?",
            submenu: [
                {
                    title: "Mathematical Framework",
                    href: "mathematical-framework.html",
                    icon: "fas fa-infinity",
                    description: "Core mathematical foundations of 1+1=1"
                },
                {
                    title: "Philosophy",
                    href: "philosophy.html",
                    icon: "fas fa-thinking",
                    description: "Philosophical foundations and meaning"
                },
                {
                    title: "About Unity Mathematics",
                    href: "about.html",
                    icon: "fas fa-question-circle",
                    description: "Project overview and vision"
                }
            ]
        },
        {
            title: "Metastation",
            href: "metastation_streamlit.py",
            icon: "fas fa-satellite-dish",
            description: "Master Unity Dashboard - Live Streamlit Control Center",
            external: true,
            port: 8501
        },
        {
            title: "Experience",
            icon: "fas fa-play",
            description: "See Unity Mathematics in action",
            submenu: [
                {
                    title: "Unity Calculator",
                    href: "interactive-unity-calculator.html",
                    icon: "fas fa-calculator",
                    description: "Interactive 1+1=1 calculator with live visualizations"
                },
                {
                    title: "Consciousness Field",
                    href: "consciousness_dashboard.html",
                    icon: "fas fa-brain",
                    description: "Real-time consciousness field visualization"
                },
                {
                    title: "Interactive Dashboards",
                    href: "dashboard-launcher.html",
                    icon: "fas fa-chart-line",
                    description: "Live mathematical dashboards and experiments"
                },
                {
                    title: "Zen Unity Meditation",
                    href: "zen-unity-meditation.html",
                    icon: "fas fa-yin-yang",
                    description: "Meditative consciousness exploration"
                },
                {
                    title: "3D Consciousness Field",
                    href: "enhanced-3d-consciousness-field.html",
                    icon: "fas fa-cube",
                    description: "Enhanced 3D consciousness field visualization"
                },
                {
                    title: "Mathematical Playground",
                    href: "mathematical_playground.html",
                    icon: "fas fa-calculator",
                    description: "Interactive mathematical exploration tools"
                },
                {
                    title: "Unity Consciousness Experience",
                    href: "unity_consciousness_experience.html",
                    icon: "fas fa-brain",
                    description: "Immersive consciousness and unity experience"
                }
            ]
        },
        {
            title: "Proofs",
            icon: "fas fa-check-circle",
            description: "Mathematical demonstrations",
            submenu: [
                {
                    title: "Comprehensive Proof Gallery",
                    href: "comprehensive-proof-gallery.html",
                    icon: "fas fa-library",
                    description: "Complete collection of 1+1=1 proofs across 6 frameworks"
                },
                {
                    title: "Unity Proofs",
                    href: "unity_proof.html",
                    icon: "fas fa-equals",
                    description: "Core formal proofs that 1+1=1"
                },
                {
                    title: "Al-Khwarizmi Unity",
                    href: "al_khwarizmi_phi_unity.html",
                    icon: "fas fa-scroll",
                    description: "Classical mathematical bridge to modern unity"
                },
                {
                    title: "Transcendental Demonstrations",
                    href: "transcendental-unity-demo.html",
                    icon: "fas fa-atom",
                    description: "Advanced transcendental unity proofs"
                }
            ]
        },
        {
            title: "Build",
            icon: "fas fa-code",
            description: "Implementations and visualizations",
            submenu: [
                {
                    title: "Python Implementations",
                    href: "python-implementations-showcase.html",
                    icon: "fab fa-python",
                    description: "Complete showcase of 25+ Python modules"
                },
                {
                    title: "Implementations Gallery",
                    href: "implementations-gallery.html",
                    icon: "fas fa-laptop-code",
                    description: "Code implementations and technical visualizations"
                },
                {
                    title: "Unity Visualizations",
                    href: "unity_visualization.html",
                    icon: "fas fa-eye",
                    description: "Visual unity demonstrations and graphics"
                },
                {
                    title: "Visualization Gallery",
                    href: "visualization-gallery.html",
                    icon: "fas fa-images",
                    description: "Complete collection of mathematical visualizations"
                },
                {
                    title: "Advanced Systems",
                    href: "advanced-systems.html",
                    icon: "fas fa-cogs",
                    description: "Advanced computational systems and algorithms"
                },
                {
                    title: "Playground",
                    href: "playground.html",
                    icon: "fas fa-flask",
                    description: "Mathematical experimentation environment"
                },
                {
                    title: "Live Code Showcase",
                    href: "live-code-showcase.html",
                    icon: "fas fa-code-branch",
                    description: "Dynamic code demonstrations and examples"
                },
                {
                    title: "Gallery Master",
                    href: "gallery-master.html",
                    icon: "fas fa-th",
                    description: "Master gallery collection and browser"
                },
                {
                    title: "DALLE Gallery",
                    href: "dalle-gallery.html",
                    icon: "fas fa-palette",
                    description: "AI-generated mathematical art and visualizations"
                }
            ]
        },
        {
            title: "Meta",
            icon: "fas fa-robot",
            description: "AI, Meta-Learning, and Advanced Systems",
            submenu: [
                {
                    title: "AI Agents Ecosystem",
                    href: "ai-agents-ecosystem.html",
                    icon: "fas fa-network-wired",
                    description: "Multi-agent consciousness systems"
                },
                {
                    title: "Metagamer Agent",
                    href: "metagamer_agent.html",
                    icon: "fas fa-chess-king",
                    description: "Advanced metagaming AI agent"
                },
                {
                    title: "AI Unified Hub",
                    href: "ai-unified-hub.html",
                    icon: "fas fa-brain-circuit",
                    description: "Unified AI systems and meta-learning"
                },
                {
                    title: "Enhanced AI Demo",
                    href: "enhanced-ai-demo.html",
                    icon: "fas fa-microchip",
                    description: "Live AI demonstrations and interactions"
                },
                {
                    title: "Unity Meta Atlas",
                    href: "unity-meta-atlas.html",
                    icon: "fas fa-map",
                    description: "Meta-cognitive mapping system"
                },
                {
                    title: "3000 ELO Proof System",
                    href: "3000-elo-proof.html",
                    icon: "fas fa-trophy",
                    description: "Superhuman mathematical proof generation"
                },
                {
                    title: "Metagambit System",
                    href: "metagambit.html",
                    icon: "fas fa-chess",
                    description: "Advanced strategic metagaming system"
                },
                {
                    title: "Anthill Simulation",
                    href: "anthill.html",
                    icon: "fas fa-bug",
                    description: "Multi-agent swarm intelligence simulation"
                }
            ]
        },
        {
            title: "Tools",
            icon: "fas fa-tools",
            description: "Utilities and special features",
            submenu: [
                {
                    title: "Site Map",
                    href: "sitemap.html",
                    icon: "fas fa-sitemap",
                    description: "Complete site navigation overview"
                },
                {
                    title: "API Documentation",
                    href: "api-documentation.html",
                    icon: "fas fa-code",
                    description: "API documentation and integration guides"
                },
                {
                    title: "OpenAI Integration",
                    href: "openai-integration.html",
                    icon: "fas fa-brain",
                    description: "OpenAI API integration and demos"
                },
                {
                    title: "Mobile App",
                    href: "mobile-app.html",
                    icon: "fas fa-mobile-alt",
                    description: "Mobile application and features"
                },
                {
                    title: "Unity Axioms",
                    href: "unity-axioms.html",
                    icon: "fas fa-list-ol",
                    description: "Core axioms and mathematical principles"
                }
            ]
        },
        {
            title: "Research",
            icon: "fas fa-university",
            description: "Academic depth and development",
            submenu: [
                {
                    title: "Publications",
                    href: "publications.html",
                    icon: "fas fa-file-alt",
                    description: "Academic papers and research"
                },
                {
                    title: "Strategic Roadmap",
                    href: "research-strategic-roadmap.html",
                    icon: "fas fa-road",
                    description: "Development roadmap and future plans"
                },
                {
                    title: "Research Hub",
                    href: "research.html",
                    icon: "fas fa-microscope",
                    description: "Research papers and ongoing studies"
                },
                {
                    title: "Research Portal",
                    href: "research-portal.html",
                    icon: "fas fa-graduation-cap",
                    description: "Academic research portal and explorer"
                },
                {
                    title: "Academic Portal",
                    href: "academic-portal.html",
                    icon: "fas fa-university",
                    description: "Academic resources and materials"
                },
                {
                    title: "Further Reading",
                    href: "further-reading.html",
                    icon: "fas fa-book-open",
                    description: "Extended materials and references"
                },
                {
                    title: "Learning Hub",
                    href: "learning.html",
                    icon: "fas fa-graduation-cap",
                    description: "Educational materials and tutorials"
                }
            ]
        }
    ],
    
    // Featured pages - Highlighting most compelling and unique content
    featuredPages: [
        {
            title: "Unity Calculator",
            href: "interactive-unity-calculator.html",
            icon: "fas fa-calculator",
            description: "Interactive 1+1=1 calculator with live consciousness visualization"
        },
        {
            title: "Comprehensive Proof Gallery", 
            href: "comprehensive-proof-gallery.html",
            icon: "fas fa-library",
            description: "25 mathematical proofs across 6 frameworks demonstrating 1+1=1"
        },
        {
            title: "Consciousness Field",
            href: "consciousness_dashboard.html",
            icon: "fas fa-brain",
            description: "Revolutionary real-time consciousness field mathematics"
        },
        {
            title: "Mathematical Framework",
            href: "mathematical-framework.html",
            icon: "fas fa-infinity",
            description: "Rigorous mathematical foundations of Unity Mathematics"
        }
    ],
    
    // Footer navigation - Updated for complete coverage
    footerNav: {
        "Core Unity": [
            { title: "Mathematical Framework", href: "mathematical-framework.html" },
            { title: "Unity Calculator", href: "interactive-unity-calculator.html" },
            { title: "Proof Gallery", href: "comprehensive-proof-gallery.html" },
            { title: "Metastation Dashboard", href: "metastation_streamlit.py", external: true }
        ],
        "Experience & Build": [
            { title: "Consciousness Field", href: "consciousness_dashboard.html" },
            { title: "Python Implementations", href: "python-implementations-showcase.html" },
            { title: "Zen Meditation", href: "zen-unity-meditation.html" },
            { title: "Visualization Gallery", href: "visualization-gallery.html" }
        ],
        "Meta & AI": [
            { title: "AI Agents Ecosystem", href: "ai-agents-ecosystem.html" },
            { title: "Metagamer Agent", href: "metagamer_agent.html" },
            { title: "3000 ELO System", href: "3000-elo-proof.html" },
            { title: "Unity Meta Atlas", href: "unity-meta-atlas.html" }
        ],
        "Academic & Tools": [
            { title: "Publications", href: "publications.html" },
            { title: "Strategic Roadmap", href: "research-strategic-roadmap.html" },
            { title: "Site Map", href: "sitemap.html" },
            { title: "About", href: "about.html" }
        ]
    },
    
    // Search configuration
    searchConfig: {
        enabled: true,
        placeholder: "Search Unity Mathematics...",
        categories: [
            "Mathematical Proofs",
            "Implementations", 
            "Visualizations",
            "Consciousness",
            "Philosophy",
            "Research"
        ]
    },
    
    // Accessibility configuration
    accessibility: {
        skipLinks: true,
        highContrast: true,
        keyboardNavigation: true,
        screenReaderSupport: true
    },
    
    // Theme configuration
    theme: {
        primaryColor: "#FFD700", // Unity Gold
        secondaryColor: "#00D4FF", // Quantum Blue
        accentColor: "#8B5FBF", // Consciousness Purple
        phiColor: "#FF7F50" // Phi Orange
    },
    
    // Social/meta information - Optimized for discovery and engagement
    social: {
        title: "Unity Mathematics Institute | Interactive 1+1=1 Calculator & Proof Gallery",
        description: "Experience revolutionary mathematics: Interactive calculator proving 1+1=1, comprehensive proof gallery across 6 frameworks, and consciousness-integrated computing. Try the Unity Calculator now!",
        image: "assets/images/unity_mandala.png",
        url: "https://nourimabrouk.github.io/Een/"
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = window.UnityNavConfig;
}