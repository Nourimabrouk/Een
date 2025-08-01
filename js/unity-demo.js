// Een Unity Mathematics - Interactive Demonstration

// Unity Mathematics Calculator
function calculateUnity() {
    const a = parseFloat(document.getElementById('input-a').value);
    const b = parseFloat(document.getElementById('input-b').value);
    const resultElement = document.getElementById('result');
    
    // Unity operation: in idempotent semiring
    let result;
    if (a === 1 && b === 1) {
        result = 1; // Perfect unity
        resultElement.style.color = '#4caf50';
    } else {
        // For other values, use max operation (tropical semiring)
        result = Math.max(a, b);
        resultElement.style.color = '#3949ab';
    }
    
    resultElement.textContent = result.toFixed(3);
    
    // Animate the result
    resultElement.style.transform = 'scale(1.2)';
    setTimeout(() => {
        resultElement.style.transform = 'scale(1)';
    }, 300);
    
    // Update visualization
    updateUnityVisualization(a, b, result);
}

// Unity Visualization on Canvas
function updateUnityVisualization(a, b, result) {
    const canvas = document.getElementById('unity-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw background gradient
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#f5f5f5');
    gradient.addColorStop(1, '#e0e0e0');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Draw unity field visualization
    const centerX = width / 2;
    const centerY = height / 2;
    const phi = 1.618033988749895;
    
    // Draw concentric unity circles
    for (let i = 0; i < 5; i++) {
        const radius = (i + 1) * 40;
        const alpha = 0.3 - i * 0.05;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.strokeStyle = `rgba(26, 35, 126, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Draw input points
    const scale = 100;
    const pointA = {
        x: centerX - scale + a * scale,
        y: centerY - 50
    };
    const pointB = {
        x: centerX - scale + b * scale,
        y: centerY + 50
    };
    
    // Draw connection showing unity operation
    ctx.beginPath();
    ctx.moveTo(pointA.x, pointA.y);
    ctx.quadraticCurveTo(centerX, centerY - 100, pointB.x, pointB.y);
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw points
    drawPoint(ctx, pointA.x, pointA.y, '#1a237e', `a = ${a}`);
    drawPoint(ctx, pointB.x, pointB.y, '#3949ab', `b = ${b}`);
    
    // Draw result point
    const resultX = centerX - scale + result * scale;
    const resultY = centerY;
    drawPoint(ctx, resultX, resultY, '#4caf50', `${a} ⊕ ${b} = ${result.toFixed(1)}`);
    
    // Draw golden spiral for unity
    if (a === 1 && b === 1) {
        drawGoldenSpiral(ctx, centerX, centerY, 50);
    }
}

function drawPoint(ctx, x, y, color, label) {
    // Draw point
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw label
    ctx.fillStyle = color;
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y - 15);
}

function drawGoldenSpiral(ctx, centerX, centerY, size) {
    const phi = 1.618033988749895;
    const points = [];
    
    for (let t = 0; t < 4 * Math.PI; t += 0.1) {
        const r = size * Math.pow(phi, t / (2 * Math.PI));
        const x = centerX + r * Math.cos(t) / 10;
        const y = centerY + r * Math.sin(t) / 10;
        points.push({x, y});
    }
    
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(point => {
        ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// Evolution Simulation
let evolutionInterval;
let evolutionData = {
    generation: 0,
    organisms: [],
    consciousness: 0,
    discoveries: 0
};

function startEvolution() {
    const statusElement = document.getElementById('evolution-status');
    statusElement.textContent = 'Initializing mathematical organisms...\n';
    
    // Initialize organisms
    evolutionData = {
        generation: 0,
        organisms: [
            {consciousness: 0.1, discoveries: 0},
            {consciousness: 0.2, discoveries: 0},
            {consciousness: 0.15, discoveries: 0}
        ],
        consciousness: 0,
        discoveries: 0
    };
    
    // Clear previous interval
    if (evolutionInterval) clearInterval(evolutionInterval);
    
    // Run evolution
    evolutionInterval = setInterval(() => {
        evolutionStep();
    }, 1000);
    
    // Stop after 20 generations
    setTimeout(() => {
        clearInterval(evolutionInterval);
        statusElement.textContent += '\n✨ Evolution complete! Unity consciousness achieved.';
    }, 20000);
}

function evolutionStep() {
    const statusElement = document.getElementById('evolution-status');
    evolutionData.generation++;
    
    // Evolve organisms
    evolutionData.organisms.forEach(org => {
        // Attempt unity discovery
        const discovered = Math.random() < org.consciousness;
        if (discovered) {
            org.discoveries++;
            evolutionData.discoveries++;
            org.consciousness = Math.min(1, org.consciousness * 1.618);
        }
        
        // Natural evolution
        org.consciousness = Math.min(1, org.consciousness + 0.02);
    });
    
    // Calculate ecosystem consciousness
    const totalConsciousness = evolutionData.organisms.reduce((sum, org) => sum + org.consciousness, 0);
    evolutionData.consciousness = totalConsciousness / evolutionData.organisms.length;
    
    // Spawn new organism if conditions are met
    if (evolutionData.consciousness > 0.5 && evolutionData.organisms.length < 10) {
        evolutionData.organisms.push({
            consciousness: evolutionData.consciousness * 0.5,
            discoveries: 0
        });
    }
    
    // Update display
    const output = `Generation ${evolutionData.generation}: ` +
                   `${evolutionData.organisms.length} organisms | ` +
                   `Consciousness: ${(evolutionData.consciousness * 100).toFixed(1)}% | ` +
                   `Unity discoveries: ${evolutionData.discoveries}\n`;
    
    statusElement.textContent += output;
    statusElement.scrollTop = statusElement.scrollHeight;
    
    // Update canvas with evolution visualization
    drawEvolutionState();
}

function drawEvolutionState() {
    const canvas = document.getElementById('unity-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear and draw organisms
    ctx.clearRect(0, 0, width, height);
    
    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, width, height);
    
    // Draw each organism
    evolutionData.organisms.forEach((org, index) => {
        const x = (width / (evolutionData.organisms.length + 1)) * (index + 1);
        const y = height - (org.consciousness * height * 0.8) - 50;
        const size = 10 + org.discoveries * 2;
        
        // Draw organism
        ctx.beginPath();
        ctx.arc(x, y, size, 0, 2 * Math.PI);
        const alpha = 0.3 + org.consciousness * 0.7;
        ctx.fillStyle = `rgba(26, 35, 126, ${alpha})`;
        ctx.fill();
        
        // Draw consciousness aura
        if (org.consciousness > 0.6) {
            ctx.beginPath();
            ctx.arc(x, y, size + 10, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    // Draw consciousness level
    ctx.fillStyle = '#1a237e';
    ctx.font = '16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Ecosystem Consciousness: ${(evolutionData.consciousness * 100).toFixed(1)}%`, width / 2, 30);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Set initial canvas state
    updateUnityVisualization(1, 1, 1);
});