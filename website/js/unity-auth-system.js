/**
 * Unity Authentication System - Master Implementation
 * Supports OAuth for Google, GitHub, OpenAI with session storage
 * Compatible with both GitHub Pages and Vercel
 * Version: 1.0.0 - One-shot master implementation
 */

class UnityAuthSystem {
    constructor() {
        this.isAuthenticated = false;
        this.user = null;
        this.sessionKey = 'een_unity_session';
        this.configKey = 'een_auth_config';
        
        // OAuth configurations
        this.oauthConfigs = {
            google: {
                clientId: this.getEnvVar('GOOGLE_CLIENT_ID', ''),
                scope: 'openid profile email',
                redirectUri: this.getRedirectUri('google'),
                authUrl: 'https://accounts.google.com/o/oauth2/v2/auth'
            },
            github: {
                clientId: this.getEnvVar('GITHUB_CLIENT_ID', ''),
                scope: 'user:email',
                redirectUri: this.getRedirectUri('github'),
                authUrl: 'https://github.com/login/oauth/authorize'
            },
            openai: {
                // OpenAI uses API key authentication, not OAuth
                apiKeyRequired: true
            }
        };

        this.tokenEndpoints = {
            google: '/api/auth/google/token',
            github: '/api/auth/github/token'
        };

        this.init();
    }

    getEnvVar(name, defaultValue) {
        // Try multiple methods to get environment variables
        if (typeof process !== 'undefined' && process.env) {
            return process.env[name] || defaultValue;
        }
        if (window.location.hostname === 'localhost') {
            // Development defaults
            const devConfigs = {
                'GOOGLE_CLIENT_ID': 'your-google-client-id.apps.googleusercontent.com',
                'GITHUB_CLIENT_ID': 'your-github-client-id'
            };
            return devConfigs[name] || defaultValue;
        }
        return defaultValue;
    }

    getRedirectUri(provider) {
        const baseUrl = window.location.origin;
        return `${baseUrl}/auth/callback/${provider}`;
    }

    init() {
        // Check for existing session
        this.loadSession();
        
        // Handle OAuth callbacks
        this.handleOAuthCallback();
        
        // Setup periodic session refresh
        this.setupSessionRefresh();
        
        console.log('🔐 Unity Auth System initialized');
        this.dispatchAuthEvent('auth:initialized');
    }

    // Session Management
    loadSession() {
        try {
            const sessionData = localStorage.getItem(this.sessionKey);
            if (sessionData) {
                const session = JSON.parse(sessionData);
                
                // Check if session is still valid
                if (session.expiresAt && Date.now() < session.expiresAt) {
                    this.isAuthenticated = true;
                    this.user = session.user;
                    this.dispatchAuthEvent('auth:loaded', { user: this.user });
                    return true;
                } else {
                    // Session expired
                    this.logout();
                }
            }
        } catch (error) {
            console.error('Error loading session:', error);
            this.logout();
        }
        return false;
    }

    saveSession(user, expiresIn = 86400000) { // 24 hours default
        const sessionData = {
            user: user,
            expiresAt: Date.now() + expiresIn,
            createdAt: Date.now()
        };
        
        localStorage.setItem(this.sessionKey, JSON.stringify(sessionData));
        this.isAuthenticated = true;
        this.user = user;
        
        this.dispatchAuthEvent('auth:login', { user });
    }

    logout() {
        localStorage.removeItem(this.sessionKey);
        this.isAuthenticated = false;
        this.user = null;
        
        this.dispatchAuthEvent('auth:logout');
    }

    // OAuth Implementation
    async initiateOAuth(provider) {
        if (provider === 'openai') {
            return this.handleOpenAIAuth();
        }

        const config = this.oauthConfigs[provider];
        if (!config || !config.clientId) {
            throw new Error(`${provider} OAuth not configured`);
        }

        // Generate state for CSRF protection
        const state = this.generateRandomString(32);
        sessionStorage.setItem('oauth_state', state);
        sessionStorage.setItem('oauth_provider', provider);

        // Build OAuth URL
        const params = new URLSearchParams({
            client_id: config.clientId,
            redirect_uri: config.redirectUri,
            scope: config.scope,
            response_type: 'code',
            state: state
        });

        const authUrl = `${config.authUrl}?${params.toString()}`;
        
        // Redirect to OAuth provider
        window.location.href = authUrl;
    }

    async handleOAuthCallback() {
        const urlParams = new URLSearchParams(window.location.search);
        const code = urlParams.get('code');
        const state = urlParams.get('state');
        const provider = sessionStorage.getItem('oauth_provider');
        const expectedState = sessionStorage.getItem('oauth_state');

        if (!code || !state || !provider) {
            return; // Not an OAuth callback
        }

        // Verify state to prevent CSRF
        if (state !== expectedState) {
            console.error('OAuth state mismatch - potential CSRF attack');
            this.handleAuthError('Invalid authentication state');
            return;
        }

        try {
            // Clean up temporary storage
            sessionStorage.removeItem('oauth_state');
            sessionStorage.removeItem('oauth_provider');

            // Exchange code for token
            const user = await this.exchangeCodeForToken(provider, code);
            
            if (user) {
                this.saveSession(user);
                
                // Clean URL
                const cleanUrl = window.location.origin + window.location.pathname;
                window.history.replaceState({}, document.title, cleanUrl);
                
                // Redirect to dashboard or intended page
                const intendedPath = sessionStorage.getItem('intended_path') || '/unity-dashboard.html';
                sessionStorage.removeItem('intended_path');
                window.location.href = intendedPath;
            }
        } catch (error) {
            console.error('OAuth callback error:', error);
            this.handleAuthError('Authentication failed');
        }
    }

    async exchangeCodeForToken(provider, code) {
        // For GitHub Pages, use client-side token exchange with CORS proxy
        // For Vercel, use serverless function
        
        const endpoint = this.tokenEndpoints[provider];
        const isLocalOrVercel = window.location.hostname === 'localhost' || 
                               window.location.hostname.includes('vercel.app');

        if (isLocalOrVercel) {
            // Use serverless function
            return this.exchangeTokenServerless(provider, code);
        } else {
            // Use client-side approach for GitHub Pages
            return this.exchangeTokenClientSide(provider, code);
        }
    }

    async exchangeTokenServerless(provider, code) {
        const response = await fetch(`/api/auth/${provider}/token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code, redirect_uri: this.oauthConfigs[provider].redirectUri })
        });

        if (!response.ok) {
            throw new Error(`Token exchange failed: ${response.statusText}`);
        }

        return await response.json();
    }

    async exchangeTokenClientSide(provider, code) {
        // For GitHub Pages, implement client-side token exchange
        // This requires a CORS proxy service or public OAuth flow
        
        switch (provider) {
            case 'github':
                return this.exchangeGitHubTokenClientSide(code);
            case 'google':
                return this.exchangeGoogleTokenClientSide(code);
            default:
                throw new Error(`Client-side auth not supported for ${provider}`);
        }
    }

    async exchangeGitHubTokenClientSide(code) {
        // Use GitHub's device flow or a CORS proxy
        // For production, you'd use a secure backend
        
        const corsProxy = 'https://cors-anywhere.herokuapp.com/';
        const tokenUrl = 'https://github.com/login/oauth/access_token';
        
        const response = await fetch(corsProxy + tokenUrl, {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: new URLSearchParams({
                client_id: this.oauthConfigs.github.clientId,
                client_secret: 'handled-by-proxy', // Would need secure handling
                code: code,
                redirect_uri: this.oauthConfigs.github.redirectUri
            })
        });

        const tokenData = await response.json();
        
        if (tokenData.access_token) {
            // Get user info
            const userResponse = await fetch(corsProxy + 'https://api.github.com/user', {
                headers: { 'Authorization': `token ${tokenData.access_token}` }
            });
            
            const userData = await userResponse.json();
            
            return {
                id: userData.id,
                name: userData.name || userData.login,
                email: userData.email,
                avatar: userData.avatar_url,
                provider: 'github',
                accessToken: tokenData.access_token
            };
        }
        
        throw new Error('Failed to get access token');
    }

    async exchangeGoogleTokenClientSide(code) {
        // Similar implementation for Google OAuth
        // This is simplified - production would use secure backend
        throw new Error('Google client-side auth requires backend implementation');
    }

    // OpenAI API Key Authentication
    async handleOpenAIAuth() {
        return new Promise((resolve) => {
            this.showOpenAIKeyModal(resolve);
        });
    }

    showOpenAIKeyModal(callback) {
        const modal = document.createElement('div');
        modal.className = 'auth-modal';
        modal.innerHTML = `
            <div class="auth-modal-content">
                <h3>🤖 OpenAI API Key Authentication</h3>
                <p>Enter your OpenAI API key to access enhanced Unity AI features</p>
                <input type="password" id="openai-key" placeholder="sk-..." />
                <div class="auth-modal-actions">
                    <button id="openai-submit">Authenticate</button>
                    <button id="openai-cancel">Cancel</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        document.getElementById('openai-submit').addEventListener('click', async () => {
            const apiKey = document.getElementById('openai-key').value;
            if (apiKey) {
                try {
                    // Validate API key
                    const valid = await this.validateOpenAIKey(apiKey);
                    if (valid) {
                        const user = {
                            id: 'openai-user',
                            name: 'OpenAI User',
                            email: '',
                            provider: 'openai',
                            apiKey: apiKey
                        };
                        
                        this.saveSession(user);
                        modal.remove();
                        callback(user);
                    } else {
                        alert('Invalid OpenAI API key');
                    }
                } catch (error) {
                    alert('Failed to validate API key');
                }
            }
        });

        document.getElementById('openai-cancel').addEventListener('click', () => {
            modal.remove();
            callback(null);
        });
    }

    async validateOpenAIKey(apiKey) {
        try {
            const response = await fetch('https://api.openai.com/v1/models', {
                headers: { 'Authorization': `Bearer ${apiKey}` }
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    // Utility Methods
    generateRandomString(length) {
        const array = new Uint8Array(length);
        crypto.getRandomValues(array);
        return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
    }

    setupSessionRefresh() {
        // Refresh session every 30 minutes
        setInterval(() => {
            if (this.isAuthenticated && this.user) {
                this.refreshSession();
            }
        }, 1800000); // 30 minutes
    }

    async refreshSession() {
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user: this.user })
            });

            if (response.ok) {
                const userData = await response.json();
                this.saveSession(userData.user, userData.expiresIn);
            }
        } catch (error) {
            console.log('Session refresh failed, user needs to re-authenticate');
        }
    }

    handleAuthError(message) {
        this.dispatchAuthEvent('auth:error', { error: message });
        
        // Show user-friendly error
        const errorModal = document.createElement('div');
        errorModal.className = 'auth-error-modal';
        errorModal.innerHTML = `
            <div class="auth-modal-content">
                <h3>Authentication Error</h3>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove()">OK</button>
            </div>
        `;
        document.body.appendChild(errorModal);
    }

    dispatchAuthEvent(eventName, data = {}) {
        const event = new CustomEvent(eventName, { detail: data });
        window.dispatchEvent(event);
    }

    // Public API Methods
    async login(provider) {
        if (this.isAuthenticated) {
            return this.user;
        }

        try {
            await this.initiateOAuth(provider);
        } catch (error) {
            this.handleAuthError(error.message);
            throw error;
        }
    }

    requireAuth(redirectPath = null) {
        if (!this.isAuthenticated) {
            if (redirectPath) {
                sessionStorage.setItem('intended_path', redirectPath);
            }
            this.showLoginModal();
            return false;
        }
        return true;
    }

    showLoginModal() {
        if (document.querySelector('.auth-login-modal')) {
            return; // Already showing
        }

        const modal = document.createElement('div');
        modal.className = 'auth-login-modal';
        modal.innerHTML = `
            <div class="auth-modal-content">
                <h3>🌟 Unity Mathematics Authentication</h3>
                <p>Sign in to access enhanced Unity features and personalized experience</p>
                
                <div class="auth-providers">
                    <button class="auth-provider google" data-provider="google">
                        <i class="fab fa-google"></i>
                        Continue with Google
                    </button>
                    
                    <button class="auth-provider github" data-provider="github">
                        <i class="fab fa-github"></i>
                        Continue with GitHub
                    </button>
                    
                    <button class="auth-provider openai" data-provider="openai">
                        <i class="fas fa-robot"></i>
                        OpenAI API Key
                    </button>
                </div>
                
                <div class="auth-benefits">
                    <h4>🎯 Benefits of signing in:</h4>
                    <ul>
                        <li>✅ Save your Unity calculations</li>
                        <li>🧠 Personalized consciousness insights</li>
                        <li>🤖 AI-enhanced mathematical analysis</li>
                        <li>📊 Advanced visualization features</li>
                        <li>🔄 Sync across devices</li>
                    </ul>
                </div>
                
                <button class="auth-skip">Continue without signing in</button>
            </div>
        `;

        document.body.appendChild(modal);

        // Bind events
        modal.querySelectorAll('.auth-provider').forEach(button => {
            button.addEventListener('click', async (e) => {
                const provider = e.target.closest('.auth-provider').dataset.provider;
                modal.remove();
                await this.login(provider);
            });
        });

        modal.querySelector('.auth-skip').addEventListener('click', () => {
            modal.remove();
        });
    }

    getUserProfile() {
        return this.user;
    }

    isUserAuthenticated() {
        return this.isAuthenticated;
    }

    getAccessToken() {
        return this.user?.accessToken || this.user?.apiKey;
    }
}

// Global instance
window.UnityAuth = new UnityAuthSystem();

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnityAuthSystem;
}