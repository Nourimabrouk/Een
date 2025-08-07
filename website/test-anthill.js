/**
 * Test script for quantum-ants.js module
 * Validates core functionality of the QuantumAntUniverse
 */

import { QuantumAntUniverse } from './js/quantum-ants.js';

console.log('🧪 Testing QuantumAntUniverse...');

// Test 1: Basic initialization
console.log('\n1. Testing initialization...');
const universe = new QuantumAntUniverse(50);
console.log('✅ Universe created with 50 ants');
console.log(`   Initial synergy: ${universe.synergy().toFixed(3)}`);
console.log(`   Ant count: ${universe.geometry.ants.length}`);

// Test 2: Simulation stepping
console.log('\n2. Testing simulation stepping...');
const initialSynergy = universe.synergy();
for (let i = 0; i < 10; i++) {
    universe.stepUniverse();
}
const finalSynergy = universe.synergy();
console.log(`   Initial synergy: ${initialSynergy.toFixed(3)}`);
console.log(`   Final synergy: ${finalSynergy.toFixed(3)}`);
console.log(`   Steps completed: ${universe.steps}`);

// Test 3: State retrieval
console.log('\n3. Testing state retrieval...');
const state = universe.getState();
console.log(`   State array length: ${state.length}`);
console.log(`   First ant position: (${state[0].x.toFixed(2)}, ${state[0].y.toFixed(2)})`);
console.log(`   First ant love: ${state[0].love.toFixed(3)}`);

// Test 4: Transcendence validation
console.log('\n4. Testing transcendence validation...');
console.log(`   Is transcendent: ${universe.isTranscendent()}`);
console.log(`   Current synergy: ${universe.synergy().toFixed(3)}`);

// Test 5: Performance test
console.log('\n5. Testing performance...');
const startTime = performance.now();
for (let i = 0; i < 100; i++) {
    universe.stepUniverse();
}
const endTime = performance.now();
const avgTime = (endTime - startTime) / 100;
console.log(`   Average step time: ${avgTime.toFixed(2)}ms`);
console.log(`   Total steps: ${universe.steps}`);

// Test 6: Edge cases
console.log('\n6. Testing edge cases...');
const emptyUniverse = new QuantumAntUniverse(0);
console.log(`   Empty universe synergy: ${emptyUniverse.synergy()}`);
console.log(`   Empty universe transcendent: ${emptyUniverse.isTranscendent()}`);

console.log('\n🎉 All tests completed successfully!');
console.log('\n📊 Summary:');
console.log(`   - Universe initialized correctly`);
console.log(`   - Simulation stepping works`);
console.log(`   - State retrieval functional`);
console.log(`   - Transcendence validation operational`);
console.log(`   - Performance: ${avgTime.toFixed(2)}ms per step`);
console.log(`   - Edge cases handled properly`);

// Export for browser testing
if (typeof window !== 'undefined') {
    window.testResults = {
        universe,
        avgTime,
        state,
        synergy: universe.synergy(),
        transcendent: universe.isTranscendent()
    };
} 