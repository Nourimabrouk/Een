// Publications Page Interactive Elements

// Citation Format Switching
function showCitation(format) {
    // Hide all citation texts
    const citations = document.querySelectorAll('.citation-text');
    citations.forEach(citation => {
        citation.classList.remove('active');
    });
    
    // Remove active class from all tabs
    const tabs = document.querySelectorAll('.format-tab');
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected citation format
    document.getElementById(format).classList.add('active');
    
    // Add active class to clicked tab
    event.target.classList.add('active');
}

// Copy Citation to Clipboard
function copyCitation() {
    const activeFormat = document.querySelector('.citation-text.active');
    let textToCopy = '';
    
    if (activeFormat.querySelector('pre code')) {
        textToCopy = activeFormat.querySelector('pre code').textContent;
    } else if (activeFormat.querySelector('p')) {
        textToCopy = activeFormat.querySelector('p').textContent;
    }
    
    navigator.clipboard.writeText(textToCopy).then(() => {
        const button = document.querySelector('.copy-citation');
        const originalText = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-check"></i> Copied!';
        button.style.background = 'var(--success-color)';
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.background = 'var(--primary-color)';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy citation: ', err);
    });
}

// Unity Visualization Canvas
function initializeUnityVisualization() {
    const canvas = document.getElementById('unity-visualization');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = 250;
    const height = canvas.height = 200;
    
    const phi = 1.618033988749895;
    let time = 0;
    let animationId;
    
    function drawVisualization() {
        ctx.clearRect(0, 0, width, height);
        
        // Background gradient
        const gradient = ctx.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, 'rgba(26, 35, 126, 0.05)');
        gradient.addColorStop(1, 'rgba(57, 73, 171, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Draw unity equation visualization
        const centerX = width / 2;
        const centerY = height / 2;
        
        // Draw "1" symbols that merge into unity
        const leftX = centerX - 60;
        const rightX = centerX + 60;
        const symbolY = centerY;
        
        // Animate convergence
        const convergence = (Math.sin(time * 0.02) + 1) / 2; // 0 to 1
        const currentLeftX = leftX + (centerX - leftX) * convergence * 0.7;
        const currentRightX = rightX - (rightX - centerX) * convergence * 0.7;
        
        // Draw "1" symbols
        ctx.fillStyle = `rgba(26, 35, 126, ${0.3 + convergence * 0.5})`;
        ctx.font = '2rem "Crimson Text", serif';
        ctx.textAlign = 'center';
        ctx.fillText('1', currentLeftX, symbolY);
        ctx.fillText('1', currentRightX, symbolY);
        
        // Draw plus sign (fading)
        ctx.fillStyle = `rgba(26, 35, 126, ${0.6 - convergence * 0.6})`;
        ctx.font = '1.5rem "Crimson Text", serif';
        ctx.fillText('+', centerX, symbolY - 10);
        
        // Draw equals and result
        ctx.fillStyle = `rgba(26, 35, 126, 0.8)`;
        ctx.font = '1.5rem "Crimson Text", serif';
        ctx.fillText('=', centerX, symbolY + 40);
        
        // Final unity "1" (intensifying)
        ctx.fillStyle = `rgba(212, 175, 55, ${convergence})`;
        ctx.font = `${2 + convergence}rem "Crimson Text", serif`;
        ctx.fillText('1', centerX, symbolY + 80);
        
        // Draw φ-harmonic spiral overlay
        if (convergence > 0.5) {
            ctx.strokeStyle = `rgba(212, 175, 55, ${(convergence - 0.5) * 0.6})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let t = 0; t < 3 * Math.PI; t += 0.1) {
                const r = 15 * Math.pow(phi, t / (2 * Math.PI));
                const x = centerX + (r * Math.cos(t + time * 0.01)) / 5;
                const y = centerY + 80 + (r * Math.sin(t + time * 0.01)) / 5;
                
                if (t === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }
        
        time++;
        animationId = requestAnimationFrame(drawVisualization);
    }
    
    // Start animation when canvas is visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                drawVisualization();
            } else {
                if (animationId) {
                    cancelAnimationFrame(animationId);
                }
            }
        });
    }, { threshold: 0.3 });
    
    observer.observe(canvas);
}

// Publication Item Animations
function initializePublicationAnimations() {
    const publicationItems = document.querySelectorAll('.publication-item');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    publicationItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(30px)';
        item.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
        observer.observe(item);
    });
}

// Badge Hover Effects
function initializeBadgeEffects() {
    const badges = document.querySelectorAll('.badge');
    
    badges.forEach(badge => {
        badge.addEventListener('mouseenter', () => {
            badge.style.transform = 'scale(1.1)';
            badge.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
        });
        
        badge.addEventListener('mouseleave', () => {
            badge.style.transform = 'scale(1)';
            badge.style.boxShadow = 'none';
        });
    });
}

// Impact Metrics Animation
function animateImpactMetrics() {
    const metricNumbers = document.querySelectorAll('.metric-item .metric-number');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const target = entry.target;
                const finalValue = target.textContent;
                
                if (!isNaN(finalValue)) {
                    animateNumber(target, 0, parseInt(finalValue), 2000);
                } else if (finalValue.includes('.')) {
                    const numericValue = parseFloat(finalValue);
                    animateDecimal(target, 0, numericValue, 2000);
                }
                
                observer.unobserve(target);
            }
        });
    }, { threshold: 0.5 });
    
    metricNumbers.forEach(number => {
        observer.observe(number);
    });
}

function animateNumber(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = Math.floor(start + (range * easeOutCubic(progress)));
        element.textContent = current;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        } else {
            element.textContent = end;
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function animateDecimal(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = start + (range * easeOutCubic(progress));
        element.textContent = current.toFixed(1);
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        } else {
            element.textContent = end.toFixed(1);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

// Publication Category Filtering
function initializePublicationFiltering() {
    // Add filter buttons
    const publicationsSection = document.querySelector('.publications-section');
    if (!publicationsSection) return;
    
    const filterContainer = document.createElement('div');
    filterContainer.className = 'publication-filters';
    filterContainer.innerHTML = `
        <div class="filter-buttons">
            <button class="filter-btn active" data-filter="all">All Publications</button>
            <button class="filter-btn" data-filter="theoretical">Theoretical</button>
            <button class="filter-btn" data-filter="consciousness">Consciousness</button>
            <button class="filter-btn" data-filter="computational">Computational</button>
            <button class="filter-btn" data-filter="quantum">Quantum</button>
            <button class="filter-btn" data-filter="applied">Applied</button>
        </div>
    `;
    
    publicationsSection.insertBefore(filterContainer, publicationsSection.firstChild.nextSibling);
    
    // Add CSS for filters
    const style = document.createElement('style');
    style.textContent = `
        .publication-filters {
            text-align: center;
            margin-bottom: 3rem;
        }
        
        .filter-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .filter-btn {
            background: none;
            border: 2px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.85rem;
        }
        
        .filter-btn.active,
        .filter-btn:hover {
            border-color: var(--primary-color);
            color: var(--primary-color);
            background: rgba(26, 35, 126, 0.05);
        }
    `;
    document.head.appendChild(style);
    
    // Add filter functionality
    const filterButtons = document.querySelectorAll('.filter-btn');
    const categories = document.querySelectorAll('.publication-category');
    
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Remove active class from all buttons
            filterButtons.forEach(btn => btn.classList.remove('active'));
            // Add active class to clicked button
            button.classList.add('active');
            
            const filter = button.dataset.filter;
            
            categories.forEach(category => {
                if (filter === 'all') {
                    category.style.display = 'block';
                } else {
                    const categoryTitle = category.querySelector('h3').textContent.toLowerCase();
                    const shouldShow = categoryTitle.includes(filter) || 
                                     (filter === 'applied' && (categoryTitle.includes('applied') || categoryTitle.includes('category') || categoryTitle.includes('topological')));
                    
                    category.style.display = shouldShow ? 'block' : 'none';
                }
            });
        });
    });
}

// Search Functionality
function initializePublicationSearch() {
    const searchContainer = document.createElement('div');
    searchContainer.className = 'publication-search';
    searchContainer.innerHTML = `
        <div class="search-box">
            <i class="fas fa-search"></i>
            <input type="text" placeholder="Search publications..." id="publication-search">
            <button class="clear-search" id="clear-search" style="display: none;">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    const publicationsSection = document.querySelector('.publications-section');
    const sectionTitle = publicationsSection.querySelector('.section-title');
    sectionTitle.parentNode.insertBefore(searchContainer, sectionTitle.nextSibling);
    
    // Add CSS for search
    const style = document.createElement('style');
    style.textContent = `
        .publication-search {
            max-width: 500px;
            margin: 2rem auto;
        }
        
        .search-box {
            position: relative;
            display: flex;
            align-items: center;
        }
        
        .search-box i.fa-search {
            position: absolute;
            left: 1rem;
            color: var(--text-secondary);
        }
        
        .search-box input {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            border: 2px solid var(--border-color);
            border-radius: 25px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        .search-box input:focus {
            outline: none;
            border-color: var(--primary-color);
        }
        
        .clear-search {
            position: absolute;
            right: 0.5rem;
            background: none;
            border: none;
            color: var(--text-secondary);
            cursor: pointer;
            padding: 0.5rem;
            border-radius: 50%;
        }
        
        .clear-search:hover {
            background: var(--bg-light);
        }
    `;
    document.head.appendChild(style);
    
    // Add search functionality
    const searchInput = document.getElementById('publication-search');
    const clearButton = document.getElementById('clear-search');
    const publicationItems = document.querySelectorAll('.publication-item');
    
    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        
        if (searchTerm) {
            clearButton.style.display = 'block';
        } else {
            clearButton.style.display = 'none';
        }
        
        publicationItems.forEach(item => {
            const title = item.querySelector('h4').textContent.toLowerCase();
            const authors = item.querySelector('.authors').textContent.toLowerCase();
            const summary = item.querySelector('.publication-summary p').textContent.toLowerCase();
            
            const matches = title.includes(searchTerm) || 
                          authors.includes(searchTerm) || 
                          summary.includes(searchTerm);
            
            item.style.display = matches ? 'block' : 'none';
        });
    });
    
    clearButton.addEventListener('click', () => {
        searchInput.value = '';
        clearButton.style.display = 'none';
        publicationItems.forEach(item => {
            item.style.display = 'block';
        });
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeUnityVisualization();
    initializePublicationAnimations();
    initializeBadgeEffects();
    animateImpactMetrics();
    initializePublicationFiltering();
    initializePublicationSearch();
    
    // Initialize keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('publication-search');
            if (searchInput) {
                searchInput.focus();
            }
        }
    });
});