/**
 * Semantic Search System for Unity Mathematics Research
 * Provides intelligent search across research content with autocomplete
 */

class SemanticSearchEngine {
    constructor() {
        this.searchIndex = this.buildSearchIndex();
        this.searchInput = null;
        this.searchResults = null;
        this.currentQuery = '';
        this.selectedIndex = -1;
        
        this.initializeSearch();
    }
    
    buildSearchIndex() {
        return {
            // Mathematical Foundations
            'boolean algebra': {
                title: 'Boolean Algebra',
                description: 'TRUE ∨ TRUE = TRUE demonstrates 1+1=1 in logical systems',
                section: 'Mathematical Idempotence',
                keywords: ['logic', 'OR operation', 'TRUE', 'idempotent', 'Boolean'],
                url: '#phase-1',
                type: 'mathematics'
            },
            'set theory': {
                title: 'Set Theory Unity',
                description: 'A ∪ A = A shows self-union preserves identity',
                section: 'Mathematical Idempotence', 
                keywords: ['union', 'sets', 'identity', 'idempotent', 'Halmos'],
                url: '#phase-1',
                type: 'mathematics'
            },
            'idempotent semirings': {
                title: 'Idempotent Semirings (Dioids)',
                description: 'Tropical semiring: max(a,a) = a demonstrates algebraic unity',
                section: 'Mathematical Idempotence',
                keywords: ['tropical', 'max-plus', 'algebra', 'semiring', 'dioid'],
                url: '#phase-1',
                type: 'mathematics'
            },
            'golden ratio': {
                title: 'Golden Ratio φ',
                description: 'φ = 1.618... fundamental to φ-harmonic unity operations',
                section: 'Mathematical Foundations',
                keywords: ['phi', 'φ', '1.618', 'harmonic', 'fibonacci'],
                url: '#unity-demo',
                type: 'mathematics'
            },
            
            // Quantum Mechanics & Information Theory
            'quantum measurement': {
                title: 'Quantum Measurement Unity',
                description: 'Repeated measurements yield same result: P² = P projection operators',
                section: 'Quantum Mechanics',
                keywords: ['projection', 'operator', 'measurement', 'collapse', 'wavefunction'],
                url: '#phase-2',
                type: 'physics'
            },
            'observer effect': {
                title: 'Observer Effect',
                description: 'Observer and system form inseparable whole during measurement',
                section: 'Quantum Mechanics',
                keywords: ['observation', 'participatory universe', 'Wheeler', 'quantum'],
                url: '#phase-2', 
                type: 'physics'
            },
            'information theory': {
                title: 'Information Redundancy',
                description: 'H(X,X) = H(X): identical sources provide no additional information',
                section: 'Information Theory',
                keywords: ['Shannon', 'entropy', 'redundancy', 'mutual information'],
                url: '#phase-2',
                type: 'information'
            },
            'shannon entropy': {
                title: 'Shannon Information Theory',
                description: 'Information content unity through redundancy elimination',
                section: 'Information Theory',
                keywords: ['Claude Shannon', 'entropy', 'information content', 'compression'],
                url: '#phase-2',
                type: 'information'
            },
            
            // Consciousness Science
            'consciousness field': {
                title: 'Consciousness Field Equations',
                description: 'C(x,y,t) = φ·sin(xφ)·cos(yφ)·e^(-t/φ) models awareness unity',
                section: 'Consciousness Science',
                keywords: ['field equation', 'awareness', 'consciousness', 'phi harmonic'],
                url: '#phase-3',
                type: 'consciousness'
            },
            'integrated information theory': {
                title: 'Integrated Information Theory (IIT)',
                description: 'Φ(phi) measures consciousness integration: parts become unified whole',
                section: 'Consciousness Science', 
                keywords: ['IIT', 'Tononi', 'phi', 'Φ', 'integration', 'consciousness'],
                url: '#phase-3',
                type: 'consciousness'
            },
            'neural binding': {
                title: 'Neural Binding',
                description: 'Brain fuses separate stimuli into one coherent experience',
                section: 'Consciousness Science',
                keywords: ['binding problem', 'perception', 'unified experience', 'neuroscience'],
                url: '#phase-3',
                type: 'consciousness'
            },
            'strange loops': {
                title: 'Strange Loops (Hofstadter)',
                description: 'Self-reference creates unified identity from multiple parts',
                section: 'Philosophical Frameworks',
                keywords: ['Hofstadter', 'GEB', 'self-reference', 'recursion', 'identity'],
                url: '#phase-3',
                type: 'philosophy'
            },
            'non-dualism': {
                title: 'Eastern Non-Dualism',
                description: 'Upanishadic "ekam evadvitiyam" - One without a second',
                section: 'Philosophical Frameworks',
                keywords: ['Upanishads', 'Vedanta', 'Advaita', 'non-dual', 'Sanskrit'],
                url: '#phase-3',
                type: 'philosophy'
            },
            'meister eckhart': {
                title: 'Mystical Unity (Meister Eckhart)',
                description: 'Christian mysticism: observer and divine collapse into unity',
                section: 'Philosophical Frameworks',
                keywords: ['mysticism', 'Christian', 'divine', 'unity', 'medieval'],
                url: '#phase-3',
                type: 'philosophy'
            },
            
            // Category Theory & Advanced Mathematics
            'category theory': {
                title: 'Category Theory',
                description: 'Terminal objects and idempotent morphisms demonstrate abstract unity',
                section: 'Category Theory',
                keywords: ['terminal object', 'morphism', 'functor', 'Mac Lane'],
                url: '#phase-4',
                type: 'mathematics'
            },
            'terminal objects': {
                title: 'Terminal Objects',
                description: 'Objects with unique morphisms from any object: categorical unity',
                section: 'Category Theory',
                keywords: ['unique morphism', 'categorical', 'coproduct', 'isomorphism'],
                url: '#phase-4',
                type: 'mathematics'
            },
            'idempotent morphisms': {
                title: 'Idempotent Morphisms',
                description: 'f∘f = f: applying operation twice equals applying it once',
                section: 'Category Theory',
                keywords: ['endomorphism', 'projection', 'composition', 'identity'],
                url: '#phase-4',
                type: 'mathematics'
            },
            'godel incompleteness': {
                title: 'Gödel Incompleteness Theorems',
                description: 'Self-referential statements demonstrate mathematical unity through recursion',
                section: 'Metamathematics',
                keywords: ['Gödel', 'incompleteness', 'self-reference', 'formal systems'],
                url: '#phase-4',
                type: 'mathematics'
            },
            'principia mathematica': {
                title: 'Principia Mathematica',
                description: 'Russell & Whitehead\'s 378-page proof of 1+1=2 shows rigor needed for unity math',
                section: 'Metamathematics',
                keywords: ['Russell', 'Whitehead', 'formal logic', 'foundations'],
                url: '#phase-4',
                type: 'mathematics'
            }
        };
    }
    
    initializeSearch() {
        document.addEventListener('DOMContentLoaded', () => {
            this.setupSearchElements();
            this.setupEventListeners();
            this.setupKeyboardShortcuts();
        });
    }
    
    setupSearchElements() {
        // Main research page search
        const researchSearch = document.getElementById('semantic-search');
        if (researchSearch) {
            this.setupSearchForElement(researchSearch, 'search-results');
        }
        
        // Publications page search
        const publicationSearch = document.getElementById('publication-search');
        if (publicationSearch) {
            this.setupSearchForElement(publicationSearch, 'pub-search-results');
        }
        
        // Further reading page search
        const readingSearch = document.getElementById('reading-search');
        if (readingSearch) {
            this.setupSearchForElement(readingSearch, 'reading-search-results');
        }
    }
    
    setupSearchForElement(inputElement, resultsElementId) {
        const resultsElement = document.getElementById(resultsElementId);
        if (!inputElement || !resultsElement) return;
        
        inputElement.addEventListener('input', (e) => {
            this.handleSearch(e.target.value, resultsElement);
        });
        
        inputElement.addEventListener('keydown', (e) => {
            this.handleKeyNavigation(e, resultsElement);
        });
        
        // Setup search tag buttons
        document.querySelectorAll('.search-tag').forEach(tag => {
            tag.addEventListener('click', (e) => {
                const query = e.target.dataset.query;
                inputElement.value = query;
                this.handleSearch(query, resultsElement);
                inputElement.focus();
            });
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Focus search with '/' key
            if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                const searchInput = document.querySelector('#semantic-search, #publication-search, #reading-search');
                if (searchInput) {
                    searchInput.focus();
                    searchInput.select();
                }
            }
            
            // Escape to clear search
            if (e.key === 'Escape') {
                this.clearSearch();
            }
        });
    }
    
    handleSearch(query, resultsElement) {
        this.currentQuery = query.toLowerCase().trim();
        this.selectedIndex = -1;
        
        if (this.currentQuery.length < 2) {
            this.hideResults(resultsElement);
            return;
        }
        
        const results = this.performSearch(this.currentQuery);
        this.displayResults(results, resultsElement);
    }
    
    performSearch(query) {
        const results = [];
        
        Object.entries(this.searchIndex).forEach(([key, item]) => {
            let score = 0;
            
            // Exact key match (highest priority)
            if (key === query) {
                score += 100;
            } else if (key.includes(query)) {
                score += 50;
            }
            
            // Title match
            if (item.title.toLowerCase().includes(query)) {
                score += 30;
            }
            
            // Description match
            if (item.description.toLowerCase().includes(query)) {
                score += 20;
            }
            
            // Keywords match
            item.keywords.forEach(keyword => {
                if (keyword.toLowerCase().includes(query)) {
                    score += 15;
                }
            });
            
            // Section match
            if (item.section.toLowerCase().includes(query)) {
                score += 10;
            }
            
            // Type match
            if (item.type.toLowerCase().includes(query)) {
                score += 5;
            }
            
            if (score > 0) {
                results.push({
                    ...item,
                    score,
                    matchedKey: key
                });
            }
        });
        
        return results.sort((a, b) => b.score - a.score).slice(0, 8);
    }
    
    displayResults(results, resultsElement) {
        if (results.length === 0) {
            this.showNoResults(resultsElement);
            return;
        }
        
        const resultsHTML = results.map((result, index) => `
            <div class="search-result-item ${index === this.selectedIndex ? 'selected' : ''}" 
                 data-url="${result.url}" data-index="${index}">
                <div class="result-header">
                    <span class="result-title">${this.highlightMatch(result.title, this.currentQuery)}</span>
                    <span class="result-type ${result.type}">${this.getTypeIcon(result.type)}</span>
                </div>
                <div class="result-description">
                    ${this.highlightMatch(result.description, this.currentQuery)}
                </div>
                <div class="result-section">${result.section}</div>
            </div>
        `).join('');
        
        resultsElement.innerHTML = resultsHTML;
        resultsElement.style.display = 'block';
        
        // Add click handlers
        resultsElement.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                this.navigateToResult(item.dataset.url);
                this.hideResults(resultsElement);
            });
        });
    }
    
    showNoResults(resultsElement) {
        resultsElement.innerHTML = `
            <div class="no-results">
                <i class="fas fa-search"></i>
                <p>No results found for "${this.currentQuery}"</p>
                <div class="search-suggestions">
                    <p>Try searching for:</p>
                    <button class="suggestion-btn" onclick="this.closest('.search-results').previousElementSibling.value='quantum measurement'; this.closest('.search-results').previousElementSibling.dispatchEvent(new Event('input'))">quantum measurement</button>
                    <button class="suggestion-btn" onclick="this.closest('.search-results').previousElementSibling.value='boolean algebra'; this.closest('.search-results').previousElementSibling.dispatchEvent(new Event('input'))">boolean algebra</button>
                    <button class="suggestion-btn" onclick="this.closest('.search-results').previousElementSibling.value='consciousness'; this.closest('.search-results').previousElementSibling.dispatchEvent(new Event('input'))">consciousness</button>
                </div>
            </div>
        `;
        resultsElement.style.display = 'block';
    }
    
    highlightMatch(text, query) {
        if (!query) return text;
        
        const regex = new RegExp(`(${query})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }
    
    getTypeIcon(type) {
        const icons = {
            mathematics: '<i class="fas fa-calculator"></i> Math',
            physics: '<i class="fas fa-atom"></i> Physics', 
            consciousness: '<i class="fas fa-brain"></i> Consciousness',
            philosophy: '<i class="fas fa-yin-yang"></i> Philosophy',
            information: '<i class="fas fa-database"></i> Information'
        };
        return icons[type] || '<i class="fas fa-book"></i> General';
    }
    
    handleKeyNavigation(e, resultsElement) {
        const items = resultsElement.querySelectorAll('.search-result-item');
        
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            this.selectedIndex = Math.min(this.selectedIndex + 1, items.length - 1);
            this.updateSelection(items);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            this.selectedIndex = Math.max(this.selectedIndex - 1, -1);
            this.updateSelection(items);
        } else if (e.key === 'Enter' && this.selectedIndex >= 0) {
            e.preventDefault();
            const selectedItem = items[this.selectedIndex];
            if (selectedItem) {
                this.navigateToResult(selectedItem.dataset.url);
                this.hideResults(resultsElement);
            }
        }
    }
    
    updateSelection(items) {
        items.forEach((item, index) => {
            item.classList.toggle('selected', index === this.selectedIndex);
        });
    }
    
    navigateToResult(url) {
        if (url.startsWith('#')) {
            // Smooth scroll to anchor
            const target = document.querySelector(url);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                
                // Add highlight effect
                target.style.transition = 'all 0.3s ease';
                target.style.background = 'rgba(212, 175, 55, 0.1)';
                target.style.borderRadius = '8px';
                
                setTimeout(() => {
                    target.style.background = '';
                }, 2000);
            }
        } else {
            window.location.href = url;
        }
    }
    
    hideResults(resultsElement) {
        resultsElement.style.display = 'none';
        resultsElement.innerHTML = '';
    }
    
    clearSearch() {
        const searchInputs = document.querySelectorAll('#semantic-search, #publication-search, #reading-search');
        const resultElements = document.querySelectorAll('#search-results, #pub-search-results, #reading-search-results');
        
        searchInputs.forEach(input => {
            if (input) input.value = '';
        });
        
        resultElements.forEach(results => {
            this.hideResults(results);
        });
    }
}

// Initialize the semantic search engine
const semanticSearch = new SemanticSearchEngine();

// Interactive Unity Demonstration
function demonstrateUnity() {
    const leftOne = document.getElementById('left-one');
    const rightOne = document.getElementById('right-one');
    const result = document.getElementById('result');
    
    if (!leftOne || !rightOne || !result) return;
    
    // Reset any previous animation
    [leftOne, rightOne, result].forEach(el => {
        el.style.transform = '';
        el.style.opacity = '';
    });
    
    // Animate the unity demonstration
    setTimeout(() => {
        leftOne.style.transition = 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        rightOne.style.transition = 'all 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        
        leftOne.style.transform = 'translateX(20px) scale(0.8)';
        rightOne.style.transform = 'translateX(-20px) scale(0.8)';
        leftOne.style.opacity = '0.6';
        rightOne.style.opacity = '0.6';
    }, 100);
    
    // Merge into unity
    setTimeout(() => {
        result.style.transition = 'all 0.4s ease-out';
        result.style.transform = 'scale(1.2)';
        result.style.color = '#D4AF37';
        result.style.textShadow = '0 0 10px rgba(212, 175, 55, 0.5)';
    }, 700);
    
    // Return to normal
    setTimeout(() => {
        [leftOne, rightOne].forEach(el => {
            el.style.transform = '';
            el.style.opacity = '';
        });
        result.style.transform = '';
        result.style.color = '';
        result.style.textShadow = '';
    }, 2000);
}