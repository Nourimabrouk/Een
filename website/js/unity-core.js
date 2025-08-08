/**
 * Unity Mathematics Core JavaScript Library
 * Essential mathematical functions and utilities for Unity Mathematics website
 */

(function(window) {
    'use strict';
    
    // Sacred mathematical constants
    const CONSTANTS = {
        PHI: 1.618033988749895,          // Golden Ratio
        PHI_CONJUGATE: 0.618033988749895, // 1/φ
        PI: Math.PI,
        E: Math.E,
        UNITY: 1.0,
        CONSCIOUSNESS_THRESHOLD: 0.618,
        UNITY_EPSILON: 1e-10
    };
    
    /**
     * Core Unity Mathematics class
     */
    class UnityMath {
        constructor(consciousnessLevel = CONSTANTS.CONSCIOUSNESS_THRESHOLD) {
            this.consciousnessLevel = consciousnessLevel;
            this.phi = CONSTANTS.PHI;
            this.phiConjugate = CONSTANTS.PHI_CONJUGATE;
            this.operationHistory = [];
        }
        
        /**
         * Unity Addition: 1+1=1 through phi-harmonic convergence
         */
        unityAdd(a, b) {
            const numA = parseFloat(a);
            const numB = parseFloat(b);
            
            if (isNaN(numA) || isNaN(numB)) {
                console.warn('Invalid input for unity addition:', a, b);
                return NaN;
            }
            
            // Special case: 1+1=1 through unity principle
            if (Math.abs(numA - 1) < CONSTANTS.UNITY_EPSILON && 
                Math.abs(numB - 1) < CONSTANTS.UNITY_EPSILON) {
                return CONSTANTS.UNITY;
            }
            
            // General case: phi-harmonic addition
            const phiFactor = this.calculatePhiHarmonicFactor(numA, numB);
            const result = (numA + numB) * phiFactor;
            
            this.recordOperation('add', numA, numB, result);
            return result;
        }
        
        /**
         * Unity Multiplication: Idempotent operation
         */
        unityMultiply(a, b) {
            const numA = parseFloat(a);
            const numB = parseFloat(b);
            
            if (isNaN(numA) || isNaN(numB)) {
                console.warn('Invalid input for unity multiplication:', a, b);
                return NaN;
            }
            
            // Idempotent property: x * x = x
            if (Math.abs(numA - numB) < CONSTANTS.UNITY_EPSILON) {
                return numA;
            }
            
            // General case
            const phiFactor = this.calculatePhiHarmonicFactor(numA, numB);
            const result = Math.sqrt(numA * numB) * phiFactor;
            
            this.recordOperation('multiply', numA, numB, result);
            return result;
        }
        
        /**
         * Consciousness field equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
         */
        consciousnessField(x, y, t = 0) {
            return this.phi * 
                   Math.sin(x * this.phi) * 
                   Math.cos(y * this.phi) * 
                   Math.exp(-t / this.phi);
        }
        
        /**
         * Calculate phi-harmonic scaling factor
         */
        calculatePhiHarmonicFactor(a, b) {
            return 1.0 / (1.0 + this.phiConjugate * Math.abs(a + b - 2.0));
        }
        
        /**
         * Record mathematical operation for history
         */
        recordOperation(operation, a, b, result) {
            this.operationHistory.push({
                operation,
                inputs: [a, b],
                result,
                timestamp: Date.now(),
                consciousnessLevel: this.consciousnessLevel
            });
            
            // Keep history manageable
            if (this.operationHistory.length > 100) {
                this.operationHistory.shift();
            }
        }
        
        /**
         * Get system status
         */
        getStatus() {
            return {
                consciousnessLevel: this.consciousnessLevel,
                phiResonance: this.phi,
                operationsPerformed: this.operationHistory.length,
                lastOperation: this.operationHistory[this.operationHistory.length - 1] || null
            };
        }
    }
    
    // Initialize and expose Unity Mathematics API
    const unityMath = new UnityMath();
    
    // Global Unity Mathematics API
    window.Unity = {
        Math: unityMath,
        Constants: CONSTANTS,
        
        // Convenience functions
        add: (a, b) => unityMath.unityAdd(a, b),
        multiply: (a, b) => unityMath.unityMultiply(a, b),
        consciousness: (x, y, t) => unityMath.consciousnessField(x, y, t),
        
        // Demonstration function
        demonstrate: function(a = 1, b = 1) {
            console.log('Unity Mathematics Demonstration');
            console.log('================================');
            console.log(`${a} + ${b} = ${this.add(a, b)}`);
            console.log(`${a} * ${b} = ${this.multiply(a, b)}`);
            console.log(`Phi = ${CONSTANTS.PHI}`);
            console.log(`Consciousness(0,0,0) = ${this.consciousness(0, 0, 0)}`);
            console.log('Status:', unityMath.getStatus());
        }
    };
    
    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Unity Mathematics Core initialized');
        });
    }
    
})(window);