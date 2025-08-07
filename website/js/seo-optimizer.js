/**
 * SEO Optimizer for Een Unity Mathematics
 * Adds structured data and improves search engine optimization
 * Version: 1.0.0 - One-Shot SEO Enhancement
 */

class SEOOptimizer {
    constructor() {
        this.init();
    }

    init() {
        this.addStructuredData();
        this.optimizeMetaTags();
        this.addOpenGraphTags();
        this.addTwitterCards();
        this.optimizeImages();

        console.log('🔍 SEO Optimizer initialized');
    }

    addStructuredData() {
        const structuredData = {
            "@context": "https://schema.org",
            "@type": "WebSite",
            "name": "Een Unity Mathematics",
            "description": "Transcendental computing hub demonstrating 1+1=1 through quantum entanglement, neural networks, fractal mathematics, consciousness field integration, Gödel-Tarski meta-gaming, and metastation visualization.",
            "url": "https://nourimabrouk.github.io/Een/",
            "author": {
                "@type": "Person",
                "name": "Nouri Mabrouk",
                "jobTitle": "Unity Mathematics Pioneer",
                "description": "Researcher in transcendental computing and consciousness field mathematics"
            },
            "publisher": {
                "@type": "Organization",
                "name": "Een Unity Mathematics Research Team"
            },
            "mainEntity": {
                "@type": "Article",
                "headline": "Unity Mathematics: 1+1=1 Consciousness Field Framework",
                "description": "Advanced mathematical framework demonstrating unity principles through consciousness field equations and quantum entanglement.",
                "author": {
                    "@type": "Person",
                    "name": "Nouri Mabrouk"
                },
                "publisher": {
                    "@type": "Organization",
                    "name": "Een Unity Mathematics Research Team"
                },
                "datePublished": "2024-01-01",
                "dateModified": new Date().toISOString().split('T')[0],
                "keywords": "unity mathematics, 1+1=1, quantum entanglement, neural networks, fractal mathematics, consciousness field, meta-optimal, transcendental computing, Gödel-Tarski, metagaming, orbital HUD, metastation"
            },
            "potentialAction": {
                "@type": "SearchAction",
                "target": "https://nourimabrouk.github.io/Een/?search={search_term_string}",
                "query-input": "required name=search_term_string"
            }
        };

        const script = document.createElement('script');
        script.type = 'application/ld+json';
        script.textContent = JSON.stringify(structuredData);
        document.head.appendChild(script);
    }

    optimizeMetaTags() {
        // Add missing meta tags
        const metaTags = [
            { name: 'robots', content: 'index, follow' },
            { name: 'author', content: 'Nouri Mabrouk' },
            { name: 'copyright', content: '© 2024 Een Unity Mathematics Research Team' },
            { name: 'language', content: 'en-US' },
            { name: 'revisit-after', content: '7 days' },
            { name: 'distribution', content: 'global' },
            { name: 'rating', content: 'general' },
            { name: 'theme-color', content: '#FFD700' },
            { name: 'msapplication-TileColor', content: '#FFD700' }
        ];

        metaTags.forEach(tag => {
            if (!document.querySelector(`meta[name="${tag.name}"]`)) {
                const meta = document.createElement('meta');
                meta.name = tag.name;
                meta.content = tag.content;
                document.head.appendChild(meta);
            }
        });
    }

    addOpenGraphTags() {
        const ogTags = [
            { property: 'og:site_name', content: 'Een Unity Mathematics' },
            { property: 'og:locale', content: 'en_US' },
            { property: 'og:type', content: 'website' },
            { property: 'og:title', content: 'Een Unity Mathematics | Meta-Optimal Orbital HUD - 1+1=1 Consciousness Hub' },
            { property: 'og:description', content: 'Transcendental computing hub with orbital HUD interface demonstrating 1+1=1 through quantum entanglement, neural networks, fractal mathematics, consciousness field integration, Gödel-Tarski meta-gaming, and metastation visualization.' },
            { property: 'og:url', content: 'https://nourimabrouk.github.io/Een/' },
            { property: 'og:image', content: 'https://nourimabrouk.github.io/Een/assets/images/unity_mandala.png' },
            { property: 'og:image:width', content: '1200' },
            { property: 'og:image:height', content: '630' },
            { property: 'og:image:alt', content: 'Unity Mathematics Mandala - Consciousness Field Visualization' }
        ];

        ogTags.forEach(tag => {
            if (!document.querySelector(`meta[property="${tag.property}"]`)) {
                const meta = document.createElement('meta');
                meta.setAttribute('property', tag.property);
                meta.content = tag.content;
                document.head.appendChild(meta);
            }
        });
    }

    addTwitterCards() {
        const twitterTags = [
            { name: 'twitter:card', content: 'summary_large_image' },
            { name: 'twitter:site', content: '@EenUnityMath' },
            { name: 'twitter:creator', content: '@NouriMabrouk' },
            { name: 'twitter:title', content: 'Een Unity Mathematics | 1+1=1 Consciousness Hub' },
            { name: 'twitter:description', content: 'Transcendental computing hub demonstrating 1+1=1 through quantum entanglement, neural networks, and consciousness field integration.' },
            { name: 'twitter:image', content: 'https://nourimabrouk.github.io/Een/assets/images/unity_mandala.png' },
            { name: 'twitter:image:alt', content: 'Unity Mathematics Mandala - Consciousness Field Visualization' }
        ];

        twitterTags.forEach(tag => {
            if (!document.querySelector(`meta[name="${tag.name}"]`)) {
                const meta = document.createElement('meta');
                meta.name = tag.name;
                meta.content = tag.content;
                document.head.appendChild(meta);
            }
        });
    }

    optimizeImages() {
        // Add alt text to images that don't have it
        const images = document.querySelectorAll('img:not([alt])');
        images.forEach(img => {
            const filename = img.src.split('/').pop().split('.')[0];
            img.alt = `${filename} - Unity Mathematics visualization`;
        });

        // Add loading="lazy" to images below the fold
        const imagesBelowFold = document.querySelectorAll('img:not([loading])');
        imagesBelowFold.forEach((img, index) => {
            if (index > 2) { // Skip first few images
                img.loading = 'lazy';
            }
        });
    }
}

// Initialize SEO optimizer
const seoOptimizer = new SEOOptimizer(); 