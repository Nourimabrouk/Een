// Een Unity Mathematics - Main JavaScript

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', () => {
    // Smooth scroll
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Navbar scroll effect
    const navbar = document.querySelector('.navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;

        if (currentScroll > 100) {
            navbar.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
            navbar.style.background = 'rgba(255,255,255,0.98)';
        } else {
            navbar.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
            navbar.style.background = '#ffffff';
        }

        lastScroll = currentScroll;
    });

    // Active nav link highlighting
    const sections = document.querySelectorAll('section[id]');
    const navItems = document.querySelectorAll('.nav-link');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (pageYOffset >= sectionTop - 100) {
                current = section.getAttribute('id');
            }
        });

        navItems.forEach(item => {
            item.classList.remove('active');
            if (item.getAttribute('href').slice(1) === current) {
                item.classList.add('active');
            }
        });
    });

    // Initialize MathJax
    if (window.MathJax) {
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            }
        };
    }

    // Animate elements on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function (entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    const animateElements = document.querySelectorAll('.theory-card, .research-item, .publication, .proof');
    animateElements.forEach(el => {
        observer.observe(el);
    });

    // Dark mode toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        const stored = localStorage.getItem('een-theme');
        if (stored === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        }
        themeToggle.addEventListener('click', () => {
            document.body.classList.toggle('dark-mode');
            const dark = document.body.classList.contains('dark-mode');
            themeToggle.innerHTML = dark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
            localStorage.setItem('een-theme', dark ? 'dark' : 'light');
        });
    }

    // Ensure Unified Chatbot is available on every page
    try {
        const hasUnified = typeof window.unifiedChatbot !== 'undefined';
        const hasScript = !!document.querySelector('script[src*="unified-chatbot-system.js"]');
        if (!hasUnified && !hasScript) {
            const script = document.createElement('script');
            script.src = 'js/unified-chatbot-system.js';
            script.defer = true;
            script.onload = () => {
                // initialized in script itself; expose to window
                console.log('💬 Unified Chatbot injected globally');
            };
            document.head.appendChild(script);
        }
    } catch (e) {
        console.warn('Failed to auto-inject unified chatbot:', e);
    }
});