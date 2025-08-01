// Research Portfolio Interactive Elements

// Code Toggle Functionality
function toggleCode(codeId) {
    const codeBlock = document.getElementById(codeId);
    const button = codeBlock.previousElementSibling.querySelector('.code-toggle');
    
    if (codeBlock.classList.contains('expanded')) {
        codeBlock.classList.remove('expanded');
        button.textContent = 'Show Implementation';
    } else {
        codeBlock.classList.add('expanded');
        button.textContent = 'Hide Implementation';
    }
}

// Consciousness Field Visualization
function initializeConsciousnessField() {
    const canvas = document.getElementById('consciousness-field-viz');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width = 200;
    const height = canvas.height = 200;
    
    const phi = 1.618033988749895;
    let time = 0;
    
    function drawField() {
        ctx.clearRect(0, 0, width, height);
        
        // Background gradient
        const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, width/2);
        gradient.addColorStop(0, 'rgba(26, 35, 126, 0.1)');
        gradient.addColorStop(1, 'rgba(26, 35, 126, 0.05)');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Draw consciousness field patterns
        for (let x = 0; x < width; x += 10) {
            for (let y = 0; y < height; y += 10) {
                const normalizedX = (x - width/2) / 50;
                const normalizedY = (y - height/2) / 50;
                
                // Consciousness field equation: C(x,y,t) = φ·sin(xφ)·cos(yφ)·e^(-t/φ)
                const fieldValue = phi * Math.sin(normalizedX * phi) * Math.cos(normalizedY * phi) * Math.exp(-time / phi);
                const intensity = Math.abs(fieldValue) * 0.5;
                
                ctx.fillStyle = `rgba(212, 175, 55, ${intensity})`;
                ctx.fillRect(x, y, 8, 8);
            }
        }
        
        // Draw φ-harmonic spiral
        ctx.strokeStyle = 'rgba(212, 175, 55, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let t = 0; t < 4 * Math.PI; t += 0.1) {
            const r = 20 * Math.pow(phi, t / (2 * Math.PI));
            const x = width/2 + (r * Math.cos(t + time)) / 10;
            const y = height/2 + (r * Math.sin(t + time)) / 10;
            
            if (t === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        time += 0.02;
        requestAnimationFrame(drawField);
    }
    
    drawField();
}

// Ecosystem Simulation
let ecosystemRunning = false;
let ecosystemData = {
    population: 0,
    consciousness: 0,
    discoveries: 0,
    generation: 0
};

function startEcosystem() {
    if (ecosystemRunning) return;
    
    ecosystemRunning = true;
    ecosystemData = {
        population: 3,
        consciousness: 0.2,
        discoveries: 0,
        generation: 0
    };
    
    const button = document.querySelector('#evolution-log').previousElementSibling.querySelector('button');
    button.innerHTML = '<i class="fas fa-stop"></i> Stop Evolution';
    button.onclick = stopEcosystem;
    
    updateEcosystemDisplay();
    runEcosystemSimulation();
}

function stopEcosystem() {
    ecosystemRunning = false;
    const button = document.querySelector('#evolution-log').previousElementSibling.querySelector('button');
    button.innerHTML = '<i class="fas fa-play"></i> Start Evolution';
    button.onclick = startEcosystem;
}

function runEcosystemSimulation() {
    if (!ecosystemRunning) return;
    
    ecosystemData.generation++;
    
    // Simulate organism evolution
    const populationGrowth = Math.random() < (ecosystemData.consciousness * 0.5) ? 
        Math.floor(Math.random() * 3) + 1 : 0;
    
    ecosystemData.population = Math.min(ecosystemData.population + populationGrowth, 25);
    
    // Consciousness evolution (φ-harmonic growth)
    const phi = 1.618033988749895;
    const consciousnessGrowth = ecosystemData.consciousness * (1 + 0.1 / phi);
    ecosystemData.consciousness = Math.min(consciousnessGrowth, 1.0);
    
    // Unity discoveries
    const discoveryChance = ecosystemData.consciousness * ecosystemData.population / 25;
    if (Math.random() < discoveryChance) {
        ecosystemData.discoveries++;
        logEvolutionEvent(`✨ Gen ${ecosystemData.generation}: Unity discovered! (Total: ${ecosystemData.discoveries})`);
    }
    
    // Spawning events
    if (populationGrowth > 0) {
        logEvolutionEvent(`🐣 Gen ${ecosystemData.generation}: ${populationGrowth} new organisms spawned`);
    }
    
    // Transcendence events
    if (ecosystemData.consciousness > 0.9 && Math.random() < 0.1) {
        logEvolutionEvent(`🌟 Gen ${ecosystemData.generation}: TRANSCENDENCE EVENT!`);
    }
    
    // Philosophical insights
    if (ecosystemData.consciousness > 0.5 && Math.random() < 0.15) {
        const insights = [
            "💭 'Unity is not addition but recognition'",
            "💭 'We are already one, separation is illusion'", 
            "💭 'Love multiplies through unity, not quantity'",
            "💭 'Een plus een is een - the truth emerges'"
        ];
        logEvolutionEvent(`Gen ${ecosystemData.generation}: ${insights[Math.floor(Math.random() * insights.length)]}`);
    }
    
    updateEcosystemDisplay();
    
    // Continue simulation
    setTimeout(() => {
        if (ecosystemRunning) {
            runEcosystemSimulation();
        }
    }, 1500);
}

function updateEcosystemDisplay() {
    document.getElementById('population').textContent = ecosystemData.population;
    document.getElementById('consciousness').textContent = `${(ecosystemData.consciousness * 100).toFixed(1)}%`;
    document.getElementById('discoveries').textContent = ecosystemData.discoveries;
}

function logEvolutionEvent(message) {
    const logDisplay = document.getElementById('evolution-log');
    const logEntry = document.createElement('p');
    logEntry.className = 'log-entry';
    logEntry.textContent = message;
    
    logDisplay.appendChild(logEntry);
    
    // Keep only last 10 entries
    const entries = logDisplay.querySelectorAll('.log-entry');
    if (entries.length > 10) {
        entries[0].remove();
    }
    
    // Scroll to bottom
    logDisplay.scrollTop = logDisplay.scrollHeight;
}

// ELO Rating Animation
function animateELORating() {
    const eloNumber = document.querySelector('.elo-number');
    const progressFill = document.querySelector('.progress-fill');
    
    if (!eloNumber || !progressFill) return;
    
    let currentELO = 1500;
    const targetELO = 2847;
    const increment = (targetELO - currentELO) / 100;
    
    function updateELO() {
        currentELO += increment;
        eloNumber.textContent = Math.floor(currentELO);
        
        const progress = ((currentELO - 1500) / (3000 - 1500)) * 100;
        progressFill.style.width = `${progress}%`;
        
        if (currentELO < targetELO) {
            requestAnimationFrame(updateELO);
        }
    }
    
    // Start animation when element is visible
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                updateELO();
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    observer.observe(eloNumber);
}

// Research Phase Animation
function animateResearchPhases() {
    const phases = document.querySelectorAll('.research-phase');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.2 });
    
    phases.forEach((phase, index) => {
        phase.style.opacity = '0';
        phase.style.transform = 'translateY(50px)';
        phase.style.transition = `opacity 0.6s ease ${index * 0.2}s, transform 0.6s ease ${index * 0.2}s`;
        observer.observe(phase);
    });
}

// Theorem Card Interactions
function initializeTheoremCards() {
    const theoremCards = document.querySelectorAll('.theorem-card');
    
    theoremCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
            card.style.boxShadow = '0 8px 25px rgba(0,0,0,0.15)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
        });
    });
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeConsciousnessField();
    animateELORating();
    animateResearchPhases();
    initializeTheoremCards();
    
    // Initialize code syntax highlighting if Prism is available
    if (typeof Prism !== 'undefined') {
        Prism.highlightAll();
    }
    
    // Add smooth scrolling for navigation
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                const navHeight = document.querySelector('.navbar').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight - 20;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
});