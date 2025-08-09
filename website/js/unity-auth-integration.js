/**
 * Unity Authentication Integration
 * Seamlessly integrates auth system with Unity Mathematics website
 * Provides user status, protected routes, and enhanced features
 */

class UnityAuthIntegration {
    constructor() {
        this.authSystem = window.UnityAuth;
        this.protectedFeatures = new Set(['save-calculations', 'ai-analysis', 'sync-data', 'advanced-viz']);
        this.userStatusElement = null;
        this.loginButton = null;
        
        this.init();
    }

    init() {
        // Wait for auth system to be ready
        if (!this.authSystem) {
            setTimeout(() => this.init(), 100);
            return;
        }

        this.createUserInterface();
        this.bindAuthEvents();
        this.enhanceProtectedFeatures();
        this.showAuthBenefits();
        
        console.log('🔐 Unity Auth Integration initialized');
    }

    createUserInterface() {
        // Create user status in navigation
        this.createUserStatus();
        
        // Add login button to main pages
        this.createLoginButton();
        
        // Add auth status indicator
        this.createAuthIndicator();
    }

    createUserStatus() {
        // Find navigation area (try multiple selectors)
        const navSelectors = [
            '.unified-header .header-right',
            '.unified-nav .nav-actions',
            'header .nav-actions',
            '.top-nav .actions'
        ];
        
        let navContainer = null;
        for (const selector of navSelectors) {
            navContainer = document.querySelector(selector);
            if (navContainer) break;
        }
        
        if (!navContainer) {
            // Create a floating user status
            navContainer = document.createElement('div');
            navContainer.className = 'floating-user-status';
            navContainer.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
            `;
            document.body.appendChild(navContainer);
        }

        this.userStatusElement = document.createElement('div');
        this.userStatusElement.className = 'unity-auth-status';
        navContainer.appendChild(this.userStatusElement);
        
        this.updateUserStatus();
    }

    createLoginButton() {
        // Add login button to key pages
        const targetPages = ['unity-dashboard.html', 'metastation-hub.html', 'index.html'];
        const currentPage = window.location.pathname.split('/').pop();
        
        if (targetPages.includes(currentPage) || currentPage === '') {
            this.addLoginPrompt();
        }
    }

    addLoginPrompt() {
        // Create login prompt for enhanced features
        const loginPrompt = document.createElement('div');
        loginPrompt.className = 'unity-login-prompt';
        loginPrompt.innerHTML = `
            <div class="login-prompt-content">
                <div class="login-prompt-icon">🌟</div>
                <div class="login-prompt-text">
                    <h4>Unlock Enhanced Unity Features</h4>
                    <p>Sign in to save calculations, access AI analysis, and sync across devices</p>
                </div>
                <button class="login-prompt-btn" onclick="UnityAuthIntegration.instance.showLoginModal()">
                    Sign In
                </button>
                <button class="login-prompt-close" onclick="this.parentElement.parentElement.style.display='none'">
                    ×
                </button>
            </div>
        `;
        
        // Add styles
        loginPrompt.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            max-width: 350px;
            background: rgba(18, 18, 26, 0.95);
            border: 1px solid rgba(255, 215, 0, 0.3);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(10px);
            z-index: 1000;
            animation: slideInUp 0.5s ease-out;
        `;
        
        // Only show if not authenticated and not dismissed
        if (!this.authSystem.isAuthenticated && !localStorage.getItem('unity_login_prompt_dismissed')) {
            document.body.appendChild(loginPrompt);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (loginPrompt.parentElement) {
                    loginPrompt.style.animation = 'slideOutDown 0.5s ease-out forwards';
                    setTimeout(() => loginPrompt.remove(), 500);
                }
            }, 10000);
        }
    }

    createAuthIndicator() {
        // Create floating auth status indicator
        const indicator = document.createElement('div');
        indicator.className = 'auth-status-indicator';
        indicator.style.display = 'none';
        document.body.appendChild(indicator);
        
        this.authIndicator = indicator;
    }

    updateUserStatus() {
        if (!this.userStatusElement) return;

        if (this.authSystem.isAuthenticated && this.authSystem.user) {
            const user = this.authSystem.user;
            
            this.userStatusElement.innerHTML = `
                <div class="unity-user-profile" onclick="UnityAuthIntegration.instance.toggleUserDropdown()">
                    <div class="unity-user-avatar">
                        ${user.avatar ? 
                            `<img src="${user.avatar}" alt="${user.name}" />` : 
                            user.name.charAt(0).toUpperCase()
                        }
                    </div>
                    <div class="unity-user-info">
                        <div class="unity-user-name">${user.name}</div>
                        <div class="unity-user-provider">${user.provider}</div>
                    </div>
                    <i class="fas fa-chevron-down"></i>
                    
                    <div class="unity-user-dropdown">
                        <a href="#" class="unity-user-dropdown-item" onclick="UnityAuthIntegration.instance.viewProfile()">
                            <i class="fas fa-user"></i> Profile
                        </a>
                        <a href="#" class="unity-user-dropdown-item" onclick="UnityAuthIntegration.instance.viewSavedData()">
                            <i class="fas fa-save"></i> Saved Calculations
                        </a>
                        <a href="#" class="unity-user-dropdown-item" onclick="UnityAuthIntegration.instance.viewSettings()">
                            <i class="fas fa-cog"></i> Settings
                        </a>
                        <hr style="margin: 8px 4px; border: none; border-top: 1px solid rgba(255,255,255,0.1);">
                        <a href="#" class="unity-user-dropdown-item" onclick="UnityAuthIntegration.instance.logout()">
                            <i class="fas fa-sign-out-alt"></i> Sign Out
                        </a>
                    </div>
                </div>
            `;
        } else {
            this.userStatusElement.innerHTML = `
                <button class="unity-login-btn" onclick="UnityAuthIntegration.instance.showLoginModal()">
                    <i class="fas fa-sign-in-alt"></i> Sign In
                </button>
            `;
        }
    }

    bindAuthEvents() {
        // Listen for auth events
        window.addEventListener('auth:login', (e) => {
            this.updateUserStatus();
            this.showAuthSuccess('Welcome back!');
            this.enhanceProtectedFeatures();
        });

        window.addEventListener('auth:logout', () => {
            this.updateUserStatus();
            this.showAuthInfo('Signed out successfully');
            this.resetProtectedFeatures();
        });

        window.addEventListener('auth:error', (e) => {
            this.showAuthError(e.detail.error);
        });
    }

    enhanceProtectedFeatures() {
        if (!this.authSystem.isAuthenticated) return;

        // Enable protected features
        document.querySelectorAll('[data-auth-required]').forEach(element => {
            element.classList.remove('auth-disabled');
            element.removeAttribute('disabled');
        });

        // Add save buttons to calculation results
        this.addSaveButtons();
        
        // Enable AI analysis features
        this.enableAIFeatures();
        
        // Show enhanced visualizations
        this.showEnhancedVisualization();
    }

    resetProtectedFeatures() {
        // Disable protected features
        document.querySelectorAll('[data-auth-required]').forEach(element => {
            element.classList.add('auth-disabled');
            element.setAttribute('disabled', 'true');
        });

        // Remove save buttons
        document.querySelectorAll('.unity-save-btn').forEach(btn => btn.remove());
    }

    addSaveButtons() {
        // Add save buttons to calculation results
        document.querySelectorAll('.result, .unity-result').forEach(resultElement => {
            if (resultElement.querySelector('.unity-save-btn')) return; // Already has save button

            const saveBtn = document.createElement('button');
            saveBtn.className = 'unity-save-btn';
            saveBtn.innerHTML = '<i class="fas fa-save"></i> Save';
            saveBtn.onclick = () => this.saveCalculation(resultElement);
            
            resultElement.appendChild(saveBtn);
        });
    }

    enableAIFeatures() {
        // Enable AI analysis buttons
        document.querySelectorAll('.ai-analysis-btn').forEach(btn => {
            btn.classList.remove('disabled');
            btn.onclick = () => this.runAIAnalysis(btn.dataset.calculation);
        });
    }

    showEnhancedVisualization() {
        // Show premium visualization features
        document.querySelectorAll('.premium-viz').forEach(viz => {
            viz.style.display = 'block';
        });
    }

    // User Interface Methods
    toggleUserDropdown() {
        const dropdown = document.querySelector('.unity-user-dropdown');
        if (dropdown) {
            dropdown.classList.toggle('show');
            
            // Close when clicking outside
            setTimeout(() => {
                document.addEventListener('click', (e) => {
                    if (!e.target.closest('.unity-user-profile')) {
                        dropdown.classList.remove('show');
                    }
                }, { once: true });
            }, 10);
        }
    }

    showLoginModal() {
        this.authSystem.showLoginModal();
    }

    async logout() {
        this.authSystem.logout();
        this.toggleUserDropdown();
    }

    viewProfile() {
        const user = this.authSystem.user;
        const profileModal = document.createElement('div');
        profileModal.className = 'auth-modal';
        profileModal.innerHTML = `
            <div class="auth-modal-content">
                <h3>👤 User Profile</h3>
                <div class="profile-info">
                    <div class="profile-avatar">
                        ${user.avatar ? 
                            `<img src="${user.avatar}" alt="${user.name}" />` : 
                            user.name.charAt(0).toUpperCase()
                        }
                    </div>
                    <div class="profile-details">
                        <p><strong>Name:</strong> ${user.name}</p>
                        <p><strong>Email:</strong> ${user.email || 'Not provided'}</p>
                        <p><strong>Provider:</strong> ${user.provider}</p>
                        <p><strong>Member since:</strong> ${new Date().toLocaleDateString()}</p>
                    </div>
                </div>
                <button onclick="this.parentElement.parentElement.remove()">Close</button>
            </div>
        `;
        document.body.appendChild(profileModal);
        this.toggleUserDropdown();
    }

    viewSavedData() {
        // Show saved calculations and data
        const savedData = this.getSavedCalculations();
        const dataModal = document.createElement('div');
        dataModal.className = 'auth-modal';
        dataModal.innerHTML = `
            <div class="auth-modal-content">
                <h3>💾 Saved Calculations</h3>
                <div class="saved-data-list">
                    ${savedData.length > 0 ? 
                        savedData.map(item => `
                            <div class="saved-item">
                                <strong>${item.type}:</strong> ${item.result}
                                <small>${new Date(item.timestamp).toLocaleString()}</small>
                            </div>
                        `).join('') :
                        '<p>No saved calculations yet. Start calculating to save results!</p>'
                    }
                </div>
                <button onclick="this.parentElement.parentElement.remove()">Close</button>
            </div>
        `;
        document.body.appendChild(dataModal);
        this.toggleUserDropdown();
    }

    viewSettings() {
        // Show user settings
        const settingsModal = document.createElement('div');
        settingsModal.className = 'auth-modal';
        settingsModal.innerHTML = `
            <div class="auth-modal-content">
                <h3>⚙️ Settings</h3>
                <div class="settings-options">
                    <label>
                        <input type="checkbox" id="auto-save" checked> Auto-save calculations
                    </label>
                    <label>
                        <input type="checkbox" id="ai-suggestions" checked> Enable AI suggestions
                    </label>
                    <label>
                        <input type="checkbox" id="sync-data" checked> Sync data across devices
                    </label>
                </div>
                <button onclick="UnityAuthIntegration.instance.saveSettings(); this.parentElement.parentElement.remove()">
                    Save Settings
                </button>
                <button onclick="this.parentElement.parentElement.remove()">Cancel</button>
            </div>
        `;
        document.body.appendChild(settingsModal);
        this.toggleUserDropdown();
    }

    // Feature Methods
    saveCalculation(resultElement) {
        if (!this.authSystem.isAuthenticated) {
            this.showLoginModal();
            return;
        }

        const calculation = {
            type: 'Unity Calculation',
            result: resultElement.textContent.trim(),
            timestamp: Date.now(),
            page: window.location.pathname
        };

        this.saveToPersistentStorage('calculations', calculation);
        this.showAuthSuccess('Calculation saved!');
    }

    async runAIAnalysis(calculationData) {
        if (!this.authSystem.isAuthenticated) {
            this.showLoginModal();
            return;
        }

        // Show loading
        this.showAuthInfo('Running AI analysis...', 'loading');
        
        // Simulate AI analysis (replace with actual API call)
        setTimeout(() => {
            this.showAuthSuccess('AI analysis complete! Enhanced insights available.');
        }, 2000);
    }

    // Utility Methods
    saveToPersistentStorage(key, data) {
        const existingData = JSON.parse(localStorage.getItem(`unity_${key}`) || '[]');
        existingData.push(data);
        localStorage.setItem(`unity_${key}`, JSON.stringify(existingData));
    }

    getSavedCalculations() {
        return JSON.parse(localStorage.getItem('unity_calculations') || '[]');
    }

    saveSettings() {
        const settings = {
            autoSave: document.getElementById('auto-save').checked,
            aiSuggestions: document.getElementById('ai-suggestions').checked,
            syncData: document.getElementById('sync-data').checked
        };
        
        localStorage.setItem('unity_settings', JSON.stringify(settings));
        this.showAuthSuccess('Settings saved!');
    }

    showAuthBenefits() {
        // Add auth benefits to key pages
        if (!this.authSystem.isAuthenticated) {
            const benefitsElement = document.createElement('div');
            benefitsElement.className = 'auth-benefits-banner';
            benefitsElement.innerHTML = `
                <div class="benefits-content">
                    <h4>🌟 Sign in for enhanced Unity experience</h4>
                    <ul>
                        <li>💾 Save your calculations</li>
                        <li>🤖 AI-powered analysis</li>
                        <li>📊 Advanced visualizations</li>
                        <li>🔄 Sync across devices</li>
                    </ul>
                    <button onclick="UnityAuthIntegration.instance.showLoginModal()">Get Started</button>
                </div>
            `;
            
            // Insert after header or at top of content
            const insertPoint = document.querySelector('main') || document.body;
            insertPoint.insertBefore(benefitsElement, insertPoint.firstChild);
        }
    }

    // Status Messages
    showAuthSuccess(message) {
        this.showAuthMessage(message, 'success');
    }

    showAuthError(message) {
        this.showAuthMessage(message, 'error');
    }

    showAuthInfo(message, type = 'info') {
        this.showAuthMessage(message, type);
    }

    showAuthMessage(message, type) {
        if (this.authIndicator) {
            this.authIndicator.innerHTML = `
                <div class="auth-status-badge ${type}">
                    <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
                    ${message}
                </div>
            `;
            this.authIndicator.style.display = 'block';
            
            setTimeout(() => {
                this.authIndicator.style.display = 'none';
            }, 3000);
        }
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    UnityAuthIntegration.instance = new UnityAuthIntegration();
});

// Global access
window.UnityAuthIntegration = UnityAuthIntegration;