/**
 * Unified Navigation Configuration for Unity Mathematics Website
 * Centralizes all navigation structure and ensures consistency across pages
 */

window.UnityNavConfig = {
    // Site metadata
    siteName: "Unity Mathematics Institute",
    siteTagline: "Where 1+1=1",
    baseUrl: "",
    
    // Primary navigation structure
    primaryNav: [
        {
            title: "Home",
            href: "metastation-hub.html",
            icon: "fas fa-home",
            description: "Main Unity Mathematics hub"
        },
        {
            title: "Explore",
            icon: "fas fa-compass",
            description: "Discover Unity Mathematics",
            submenu: [
                {
                    title: "Mathematical Framework",
                    href: "mathematical-framework.html",
                    icon: "fas fa-infinity",
                    description: "Core mathematical foundations"
                },
                {
                    title: "Implementations Gallery",
                    href: "implementations-gallery.html", 
                    icon: "fas fa-code",
                    description: "Code implementations and visualizations"
                },
                {
                    title: "Interactive Dashboards",
                    href: "dashboard-launcher.html",
                    icon: "fas fa-chart-line",
                    description: "Real-time mathematical dashboards"
                },
                {
                    title: "Consciousness Field",
                    href: "consciousness_dashboard.html",
                    icon: "fas fa-brain",
                    description: "Consciousness field visualization"
                }
            ]
        },
        {
            title: "Proofs",
            icon: "fas fa-check-circle",
            description: "Mathematical proofs of unity",
            submenu: [
                {
                    title: "Unity Proofs",
                    href: "unity_proof.html",
                    icon: "fas fa-equals",
                    description: "Formal proofs that 1+1=1"
                },
                {
                    title: "Al-Khwarizmi Unity",
                    href: "al_khwarizmi_phi_unity.html",
                    icon: "fas fa-scroll",
                    description: "Classical mathematical bridge"
                },
                {
                    title: "Transcendental Demo",
                    href: "transcendental-unity-demo.html",
                    icon: "fas fa-atom",
                    description: "Advanced unity demonstrations"
                },
                {
                    title: "Enhanced Proofs",
                    href: "enhanced-mathematical-proofs.html",
                    icon: "fas fa-calculator",
                    description: "Enhanced mathematical demonstrations"
                }
            ]
        },
        {
            title: "Experience",
            icon: "fas fa-magic",
            description: "Interactive experiences",
            submenu: [
                {
                    title: "Zen Unity Meditation",
                    href: "zen-unity-meditation.html",
                    icon: "fas fa-yin-yang",
                    description: "Meditative consciousness exploration"
                },
                {
                    title: "Playground",
                    href: "playground.html",
                    icon: "fas fa-play-circle",
                    description: "Mathematical experimentation"
                },
                {
                    title: "Unity Visualization",
                    href: "unity_visualization.html",
                    icon: "fas fa-eye",
                    description: "Visual unity demonstrations"
                },
                {
                    title: "3D Consciousness Field",
                    href: "enhanced-3d-consciousness-field.html",
                    icon: "fas fa-cube",
                    description: "3D consciousness visualization"
                }
            ]
        },
        {
            title: "Research",
            icon: "fas fa-microscope",
            description: "Academic research",
            submenu: [
                {
                    title: "Research Hub",
                    href: "research.html",
                    icon: "fas fa-university",
                    description: "Research papers and studies"
                },
                {
                    title: "Publications",
                    href: "publications.html",
                    icon: "fas fa-file-alt",
                    description: "Academic publications"
                },
                {
                    title: "Philosophy",
                    href: "philosophy.html",
                    icon: "fas fa-thinking",
                    description: "Philosophical foundations"
                },
                {
                    title: "Further Reading",
                    href: "further-reading.html",
                    icon: "fas fa-book-open",
                    description: "Extended reading materials"
                }
            ]
        }
    ],
    
    // Featured pages for quick access
    featuredPages: [
        {
            title: "Unity Meditation",
            href: "zen-unity-meditation.html",
            icon: "fas fa-yin-yang",
            description: "Experience consciousness through unity"
        },
        {
            title: "Mathematical Proofs",
            href: "mathematical-framework.html", 
            icon: "fas fa-infinity",
            description: "Rigorous mathematical foundations"
        },
        {
            title: "Interactive Dashboards",
            href: "dashboard-launcher.html",
            icon: "fas fa-chart-line",
            description: "Real-time visualizations"
        },
        {
            title: "Implementations",
            href: "implementations-gallery.html",
            icon: "fas fa-code",
            description: "Code and visualizations"
        }
    ],
    
    // Footer navigation structure
    footerNav: {
        "Mathematical Framework": [
            { title: "Unity Equations", href: "mathematical-framework.html" },
            { title: "Consciousness Fields", href: "consciousness_dashboard.html" },
            { title: "Phi Harmonics", href: "al_khwarizmi_phi_unity.html" },
            { title: "Quantum Unity", href: "transcendental-unity-demo.html" }
        ],
        "Implementations": [
            { title: "Gallery", href: "implementations-gallery.html" },
            { title: "Dashboards", href: "dashboard-launcher.html" },
            { title: "Playground", href: "playground.html" },
            { title: "Visualizations", href: "unity_visualization.html" }
        ],
        "Experience": [
            { title: "Zen Meditation", href: "zen-unity-meditation.html" },
            { title: "3D Consciousness", href: "enhanced-3d-consciousness-field.html" },
            { title: "Unity Proofs", href: "unity_proof.html" },
            { title: "Meta Atlas", href: "unity-meta-atlas.html" }
        ],
        "Research": [
            { title: "Publications", href: "publications.html" },
            { title: "Philosophy", href: "philosophy.html" },
            { title: "About", href: "about.html" },
            { title: "Learning", href: "learning.html" }
        ],
        "Tools": [
            { title: "Site Map", href: "sitemap.html" },
            { title: "Search", href: "metastation-hub.html#search" },
            { title: "Accessibility", href: "metastation-hub.html#accessibility" },
            { title: "Contact", href: "about.html#contact" }
        ],
        "Connect": [
            { title: "GitHub", href: "https://github.com/Nourimabrouk/Een", external: true },
            { title: "Academic Papers", href: "publications.html" },
            { title: "Philosophy Blog", href: "philosophy.html" },
            { title: "Unity Institute", href: "about.html" }
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
    
    // Social/meta information
    social: {
        title: "Unity Mathematics Institute | 1+1=1",
        description: "Revolutionary mathematical framework proving 1+1=1 through phi-harmonic operations, consciousness integration, and transcendental computing.",
        image: "assets/images/unity_mandala.png",
        url: "https://nourimabrouk.github.io/Een/"
    }
};

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = window.UnityNavConfig;
}