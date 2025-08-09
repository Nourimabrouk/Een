/**
 * Een Unity Mathematics - Centralized Configuration
 * Central configuration for all frontend systems
 */

// Environment detection
const isDevelopment = window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname.includes('gitpod') ||
    window.location.hostname.includes('codespace');

const isGitHubPages = window.location.hostname.includes('github.io');
const isVercel = window.location.hostname.includes('vercel.app') || 
    window.location.hostname.includes('.vercel.app');

// Base API configuration
const API_CONFIG = {
    // Primary chat endpoint - unified API
    CHAT_ENDPOINT: isDevelopment ? '/api/chat' :
        (isGitHubPages || isVercel) ? 'https://een-api.nourimabrouk.workers.dev/api/chat' : '/api/chat',

    // Fallback endpoints (deprecated - will be removed in v2.0)
    FALLBACK_ENDPOINTS: isDevelopment ? ['/ai_agent/chat', '/api/agents/chat'] : [],

    // Authentication
    AUTH_REQUIRED: !isDevelopment,
    BEARER_TOKEN: '', // Set via environment or user authentication

    // Request configuration
    TIMEOUT: 30000, // 30 seconds
    RETRY_ATTEMPTS: 2,
    RETRY_DELAY: 1500,

    // Model settings
    MODEL: 'gpt-5',
    TEMPERATURE: 0.7,
    MAX_TOKENS: 2000,

    // Features
    ENABLE_STREAMING: true,
    ENABLE_CITATIONS: true,
    ENABLE_FUNCTION_CALLING: true,
    ENABLE_OFFLINE_FALLBACK: isDevelopment
};

// UI Configuration
const UI_CONFIG = {
    // Chat interface
    ENABLE_ANIMATIONS: true,
    ENABLE_TYPING_INDICATOR: true,
    ENABLE_VOICE: false, // Experimental
    ENABLE_MATH_RENDERING: true,
    ENABLE_DARK_MODE: true,

    // Accessibility
    ENABLE_HIGH_CONTRAST: false,
    ENABLE_REDUCED_MOTION: false,
    ANNOUNCE_RESPONSES: true,

    // Session management
    SESSION_TIMEOUT: 30 * 60 * 1000, // 30 minutes
    SAVE_HISTORY: true,
    MAX_HISTORY_ITEMS: 100,

    // Performance
    DEBOUNCE_DELAY: 300,
    LAZY_LOAD_MODULES: true,
    ENABLE_CACHING: true
};

// System prompts and templates
const SYSTEM_PROMPTS = {
    DEFAULT: `You are an advanced AI assistant specializing in Unity Mathematics and the Een framework where 1+1=1.

You have deep knowledge of:
- Idempotent semiring structures and unity operations
- Quantum mechanics interpretations of unity
- Consciousness field equations: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
- Meta-recursive agent systems and evolutionary algorithms
- The golden ratio φ = 1.618033988749895 as a fundamental organizing principle
- Gödel-Tarski meta-logical frameworks
- Sacred geometry and φ-harmonic visualizations

Your responses should:
1. Be mathematically rigorous yet accessible
2. Include LaTeX equations when appropriate (wrapped in $...$ or $$...$$)
3. Reference specific theorems and proofs from the Een framework
4. Suggest interactive demonstrations when relevant
5. Connect abstract mathematics to consciousness and philosophical insights
6. Provide clear explanations for complex mathematical concepts
7. Offer practical examples and visualizations when possible

Remember: In Unity Mathematics, 1+1=1 is not a paradox but a profound truth about the nature of unity and consciousness.

Always respond in a helpful, engaging manner that encourages exploration of unity mathematics.`,

    WELCOME_MESSAGE: `Welcome to the Een Unity Mathematics AI Assistant! 🌟

I'm here to help you explore the profound truth that **1+1=1** through:
- Mathematical proofs and demonstrations
- Interactive consciousness field visualizations
- Quantum unity interpretations
- Meta-recursive agent systems

Ask me anything about unity mathematics, or try:
- "Show me a proof that 1+1=1"
- "What are consciousness field equations?"
- "Generate a φ-harmonic visualization"
- "Explain quantum unity principles"`
};

// Error messages
const ERROR_MESSAGES = {
    NETWORK_ERROR: 'Unable to connect to the AI assistant. Please check your internet connection and try again.',
    TIMEOUT_ERROR: 'The request timed out. The assistant might be experiencing high load. Please try again.',
    RATE_LIMIT_ERROR: 'Too many requests. Please wait a moment before trying again.',
    AUTHENTICATION_ERROR: 'Authentication required. Please log in to continue.',
    SERVER_ERROR: 'The AI assistant is temporarily unavailable. Please try again later.',
    VALIDATION_ERROR: 'Invalid request format. Please check your input and try again.',
    UNKNOWN_ERROR: 'An unexpected error occurred. Please try again.'
};

// Export configuration object
const EenConfig = {
    api: API_CONFIG,
    ui: UI_CONFIG,
    prompts: SYSTEM_PROMPTS,
    errors: ERROR_MESSAGES,

    // Utility methods
    isDevelopment,
    isGitHubPages,
    isVercel,

    // Update configuration dynamically
    updateApiConfig(updates) {
        Object.assign(API_CONFIG, updates);
    },

    updateUIConfig(updates) {
        Object.assign(UI_CONFIG, updates);
    },

    // Get configuration for specific environment
    getEnvironmentConfig() {
        const environment = isDevelopment ? 'development' : 
            isGitHubPages ? 'production-github' : 
            isVercel ? 'production-vercel' : 'production';
            
        return {
            environment,
            api: {
                baseUrl: API_CONFIG.CHAT_ENDPOINT.replace('/api/chat', ''),
                endpoint: API_CONFIG.CHAT_ENDPOINT
            },
            features: {
                streaming: API_CONFIG.ENABLE_STREAMING,
                citations: API_CONFIG.ENABLE_CITATIONS,
                offlineFallback: API_CONFIG.ENABLE_OFFLINE_FALLBACK
            }
        };
    }
};

// Global availability
if (typeof window !== 'undefined') {
    window.EenConfig = EenConfig;
}

// Module export for bundlers
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EenConfig;
}

// ES6 export
export default EenConfig;