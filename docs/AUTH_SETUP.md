# Unity Authentication System Setup Guide

## 🚀 Master-Level Authentication Implementation

Complete OAuth authentication system with Google, GitHub, and OpenAI integration for both GitHub Pages and Vercel deployment.

## 🔧 Configuration

### Environment Variables (Vercel)

Set these in your Vercel dashboard under Settings > Environment Variables:

```env
# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth  
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

### Environment Variables (Local Development)

Create a `.env` file in the root directory:

```env
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

## 🌐 OAuth Application Setup

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Go to Credentials → Create Credentials → OAuth client ID
5. Application type: Web application
6. Authorized redirect URIs:
   - `https://your-domain.vercel.app/auth/callback/google`
   - `https://your-username.github.io/Een/website/unity-dashboard.html` (for GitHub Pages)
   - `http://localhost:8001/auth/callback/google` (for local development)

### GitHub OAuth Setup

1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Fill in application details:
   - Application name: "Unity Mathematics"
   - Homepage URL: `https://your-domain.vercel.app`
   - Authorization callback URL: `https://your-domain.vercel.app/auth/callback/github`
4. Add additional callback URLs if needed:
   - `https://your-username.github.io/Een/website/unity-dashboard.html`
   - `http://localhost:8001/auth/callback/github`

### OpenAI Integration

OpenAI uses API key authentication (not OAuth):
- Users provide their own OpenAI API key
- Keys are validated against OpenAI API
- Stored securely in browser session storage

## 📁 File Structure

```
Een/
├── website/
│   ├── js/
│   │   ├── unity-auth-system.js          # Core auth system
│   │   └── unity-auth-integration.js     # UI integration
│   ├── css/
│   │   └── unity-auth-styles.css         # Auth UI styles
│   ├── unity-dashboard.html              # Main dashboard with auth
│   └── auth-demo.html                    # Authentication demo
├── api/
│   ├── auth/
│   │   ├── google/
│   │   │   └── token.py                  # Google OAuth handler
│   │   ├── github/
│   │   │   └── token.py                  # GitHub OAuth handler
│   │   └── refresh.py                    # Session refresh handler
│   └── unity_api.py                      # Main API
└── vercel.json                           # Vercel configuration
```

## 🎯 Features

### Authentication Methods
- ✅ Google OAuth 2.0
- ✅ GitHub OAuth
- ✅ OpenAI API Key authentication
- ✅ Session persistence (24h default)
- ✅ Automatic session refresh
- ✅ Secure token handling

### Protected Features
- 💾 Save calculations to local storage
- 🤖 AI-enhanced mathematical analysis (with OpenAI key)
- 📊 Advanced visualization exports
- 🔄 Data synchronization across devices
- 👤 User profile management
- ⚙️ Personalized settings

### User Experience
- 🎨 Premium UI with smooth animations
- 📱 Mobile-responsive design
- ♿ Accessibility features
- 🎹 Keyboard navigation
- 🌐 Cross-platform compatibility

## 🚀 Deployment

### Vercel Deployment

1. Deploy to Vercel: `vercel --prod`
2. Set environment variables in Vercel dashboard
3. Update OAuth redirect URIs to use your Vercel domain
4. Authentication will work with serverless functions

### GitHub Pages Deployment

1. Push to GitHub repository
2. Enable GitHub Pages
3. Update OAuth redirect URIs to use GitHub Pages domain
4. Authentication uses client-side flow with CORS proxy

## 🔒 Security Features

- ✅ CSRF protection with state parameter
- ✅ Secure token storage
- ✅ Session expiration handling
- ✅ Input validation and sanitization
- ✅ Secure headers configuration
- ✅ CORS policy enforcement

## 🧪 Testing

### Local Testing

1. Start local server: `START_WEBSITE.bat`
2. Open: `http://localhost:8001/auth-demo.html`
3. Test authentication flows
4. Verify protected features

### Production Testing

1. Deploy to staging environment
2. Test OAuth flows with real providers
3. Verify session persistence
4. Test mobile responsiveness

## 🎯 Usage Examples

### Basic Authentication

```javascript
// Check authentication status
if (window.UnityAuth.isAuthenticated) {
    console.log('User:', window.UnityAuth.user);
}

// Initiate login
await window.UnityAuth.login('google');

// Logout
window.UnityAuth.logout();
```

### Protected Feature Implementation

```html
<!-- Protected feature -->
<button data-auth-required onclick="protectedFunction()">
    Premium Feature
</button>

<script>
function protectedFunction() {
    if (!window.UnityAuth.requireAuth()) {
        return; // Will show login modal
    }
    
    // Execute protected feature
    console.log('Protected feature accessed');
}
</script>
```

### Event Handling

```javascript
// Listen for authentication events
window.addEventListener('auth:login', (e) => {
    console.log('User logged in:', e.detail.user);
    enablePremiumFeatures();
});

window.addEventListener('auth:logout', () => {
    console.log('User logged out');
    disablePremiumFeatures();
});
```

## 🔧 Customization

### Styling
- Modify `css/unity-auth-styles.css` for custom appearance
- CSS custom properties for easy theming
- Responsive breakpoints for mobile optimization

### Features
- Add new OAuth providers in `unity-auth-system.js`
- Extend protected features in `unity-auth-integration.js`
- Customize session duration and refresh intervals

### API Integration
- Add custom API endpoints in `api/` directory
- Extend user profile data handling
- Implement custom session storage backends

## 📊 Analytics & Monitoring

The authentication system provides events for analytics:

```javascript
// Track authentication events
window.addEventListener('auth:login', (e) => {
    analytics.track('User Login', {
        provider: e.detail.user.provider,
        timestamp: Date.now()
    });
});
```

## 🆘 Troubleshooting

### Common Issues

1. **OAuth redirect mismatch**: Verify redirect URIs in OAuth app settings
2. **Environment variables not loading**: Check Vercel environment variable configuration
3. **CORS errors**: Ensure proper CORS configuration in API handlers
4. **Session not persisting**: Check localStorage availability and quota
5. **Mobile authentication issues**: Test on actual devices, not just browser dev tools

### Debug Mode

Enable debug logging:

```javascript
window.UnityAuth.debug = true;
```

## 🎉 Success Metrics

- ✅ Multiple OAuth provider support
- ✅ Seamless user experience
- ✅ Mobile-responsive design
- ✅ Secure session management
- ✅ Cross-platform compatibility
- ✅ Premium feature protection
- ✅ Analytics integration ready

The Unity Authentication System provides enterprise-grade security with a consumer-friendly experience, perfectly integrated with the Unity Mathematics platform.