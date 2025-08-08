/**
 * Unity Navigation System - Production Ready
 * Single-file navigation solution for all Unity Mathematics pages
 * Simply include this script to get full navigation functionality
 */

(function () {
    'use strict';

    // Load navigation configuration
    if (!window.UnityNavConfig) {
        console.warn('UnityNavConfig not found, using defaults');
        window.UnityNavConfig = {
            siteName: "Unity Mathematics Institute",
            primaryNav: [],
            featuredPages: []
        };
    }

    const config = window.UnityNavConfig;

    /**
     * Inject required CSS for navigation
     */
    function injectNavigationCSS() {
        if (document.querySelector('#unity-nav-styles')) return;

        const css = `
        /* Unity Navigation Styles */
        .unity-nav {
            background: var(--bg-secondary, #12121A);
            border-bottom: 1px solid var(--border-primary, rgba(255,255,255,0.1));
            position: sticky;
            top: 0;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        .unity-nav-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 4rem;
        }
        
        .unity-nav-brand {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--unity-gold, #FFD700);
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .unity-nav-brand:hover {
            opacity: 0.8;
        }
        
        .unity-nav-menu {
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
            gap: 2rem;
        }
        
        .unity-nav-item {
            position: relative;
        }
        
        .unity-nav-link {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            color: var(--text-secondary, #B8B8C8);
            text-decoration: none;
            border-radius: 0.5rem;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        
        .unity-nav-link:hover {
            color: var(--text-primary, #FFFFFF);
            background: var(--bg-tertiary, rgba(255,255,255,0.05));
        }
        
        .unity-nav-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            background: var(--bg-elevated, #222230);
            border: 1px solid var(--border-primary, rgba(255,255,255,0.1));
            border-radius: 0.75rem;
            box-shadow: 0 8px 32px rgba(0,0,0,0.5);
            min-width: 250px;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-10px);
            transition: all 0.2s ease;
            z-index: 1001;
        }
        
        .unity-nav-item:hover .unity-nav-dropdown {
            opacity: 1;
            visibility: visible;
            transform: translateY(0);
        }
        
        .unity-nav-dropdown-item {
            display: block;
            padding: 0.75rem 1rem;
            color: var(--text-secondary, #B8B8C8);
            text-decoration: none;
            border-radius: 0.5rem;
            margin: 0.25rem;
            transition: all 0.2s ease;
        }
        
        .unity-nav-dropdown-item:hover {
            color: var(--text-primary, #FFFFFF);
            background: var(--bg-secondary, rgba(255,255,255,0.05));
        }
        
        .unity-nav-toggle {
            display: none;
            background: none;
            border: none;
            color: var(--text-primary, #FFFFFF);
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.5rem;
        }
        
        .unity-nav-mobile {
            display: none;
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: var(--bg-secondary, #12121A);
            border-top: 1px solid var(--border-primary, rgba(255,255,255,0.1));
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }
        
        .unity-nav-mobile.active {
            max-height: 500px;
        }
        
        .unity-nav-mobile .unity-nav-link {
            display: block;
            width: 100%;
            margin: 0.25rem 0;
        }
        
        /* Mobile Responsive */
        @media (max-width: 768px) {
            .unity-nav-menu {
                display: none;
            }
            
            .unity-nav-toggle {
                display: block;
            }
            
            .unity-nav-mobile {
                display: block;
            }
            
            .unity-nav-container {
                position: relative;
            }
        }
        
        /* Accessibility */
        .unity-nav-link:focus {
            outline: 2px solid var(--unity-gold, #FFD700);
            outline-offset: 2px;
        }
        
        /* Active page styling */
        .unity-nav-link.active {
            color: var(--unity-gold, #FFD700);
            background: rgba(255, 215, 0, 0.1);
        }
        `;

        const style = document.createElement('style');
        style.id = 'unity-nav-styles';
        style.textContent = css;
        document.head.appendChild(style);
    }

    /**
     * Generate navigation HTML
     */
    function generateNavigationHTML() {
        const currentPage = window.location.pathname.split('/').pop() || 'metastation-hub.html';

        let navHTML = `
        <nav class="unity-nav">
            <div class="unity-nav-container">
                <a href="metastation-hub.html" class="unity-nav-brand">
                    <i class="fas fa-infinity"></i>
                    ${config.siteName}
                </a>
                
                <ul class="unity-nav-menu">
        `;

        // Generate primary navigation items
        config.primaryNav.forEach(item => {
            const isActive = item.href === currentPage ? 'active' : '';

            if (item.submenu && item.submenu.length > 0) {
                navHTML += `
                <li class="unity-nav-item">
                    <a href="#" class="unity-nav-link ${isActive}" role="button" aria-label="Toggle Menu" title="Toggle Menu">
                        <i class="${item.icon}"></i>
                        ${item.title}
                        <i class="fas fa-chevron-down" style="font-size: 0.8em; margin-left: 0.25rem;"></i>
                    </a>
                    <div class="unity-nav-dropdown">
                `;

                item.submenu.forEach(subitem => {
                    const subIsActive = subitem.href === currentPage ? 'active' : '';
                    navHTML += `
                    <a href="${subitem.href}" class="unity-nav-dropdown-item ${subIsActive}">
                        <i class="${subitem.icon}"></i>
                        ${subitem.title}
                    </a>
                    `;
                });

                navHTML += `</div></li>`;
            } else if (item.href) {
                navHTML += `
                <li class="unity-nav-item">
                    <a href="${item.href}" class="unity-nav-link ${isActive}">
                        <i class="${item.icon}"></i>
                        ${item.title}
                    </a>
                </li>
                `;
            }
        });

        navHTML += `
                </ul>
                
                <button class="unity-nav-toggle" aria-label="Toggle navigation">
                    <i class="fas fa-bars"></i>
                </button>
                
                <div class="unity-nav-mobile">
        `;

        // Generate mobile navigation
        config.primaryNav.forEach(item => {
            if (item.href) {
                const isActive = item.href === currentPage ? 'active' : '';
                navHTML += `
                <a href="${item.href}" class="unity-nav-link ${isActive}">
                    <i class="${item.icon}"></i>
                    ${item.title}
                </a>
                `;
            }

            if (item.submenu && item.submenu.length > 0) {
                item.submenu.forEach(subitem => {
                    const subIsActive = subitem.href === currentPage ? 'active' : '';
                    navHTML += `
                    <a href="${subitem.href}" class="unity-nav-link ${subIsActive}">
                        <i class="${subitem.icon}"></i>
                        ${subitem.title}
                    </a>
                    `;
                });
            }
        });

        navHTML += `
                </div>
            </div>
        </nav>
        `;

        return navHTML;
    }

    /**
     * Insert navigation into page
     */
    function insertNavigation() {
        // Remove existing navigation
        const existingNav = document.querySelector('.unity-nav');
        if (existingNav) {
            existingNav.remove();
        }

        // Insert new navigation at the beginning of body
        const navHTML = generateNavigationHTML();
        document.body.insertAdjacentHTML('afterbegin', navHTML);

        // Add mobile toggle functionality
        const toggle = document.querySelector('.unity-nav-toggle');
        const mobileMenu = document.querySelector('.unity-nav-mobile');

        if (toggle && mobileMenu) {
            toggle.addEventListener('click', function () {
                mobileMenu.classList.toggle('active');
                const icon = toggle.querySelector('i');
                if (mobileMenu.classList.contains('active')) {
                    icon.className = 'fas fa-times';
                } else {
                    icon.className = 'fas fa-bars';
                }
            });
        }

        // Close mobile menu when clicking links
        const mobileLinks = document.querySelectorAll('.unity-nav-mobile .unity-nav-link');
        mobileLinks.forEach(link => {
            link.addEventListener('click', function () {
                mobileMenu.classList.remove('active');
                toggle.querySelector('i').className = 'fas fa-bars';
            });
        });
    }

    /**
     * Initialize navigation system
     */
    function initializeNavigation() {
        // Inject CSS
        injectNavigationCSS();

        // Insert navigation
        insertNavigation();

        console.log('Unity Navigation System initialized');
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeNavigation);
    } else {
        initializeNavigation();
    }

    // Make functions available globally for debugging
    window.UnityNavSystem = {
        initialize: initializeNavigation,
        config: config
    };

})();