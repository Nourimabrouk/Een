#!/usr/bin/env python3
"""Unity Lab: executable contexts where '1+1=1' holds by design.

Run: python unity_lab.py
Outputs: prints examples and saves a convergence plot to unity_lab_convergence.png
"""
import math
import random
import matplotlib.pyplot as plt

PHI = (1 + 5**0.5) / 2

def boolean_or(a: int, b: int) -> int:
    # Interpreting 0/1 as False/True, OR is idempotent: 1 OR 1 = 1
    return 1 if (a or b) else 0

def set_union(A, B):
    # Idempotent aggregation: {1} ∪ {1} = {1}
    return A.union(B)

def max_plus(a: float, b: float) -> float:
    # Tropical (max) addition: idempotent, max(1,1)=1
    return max(a, b)

def phi_contract_to_one(x: float, strength: float = 1/PHI) -> float:
    # Contract x toward 1 by a phi-based factor
    return 1.0 + (x - 1.0) * (1.0 - strength)

def demo_prints():
    print('Boolean OR: 1 ⊕ 1 =', boolean_or(1,1))
    print('Set union: |{1} ⊕ {1}| =', len(set_union({1},{1})))
    print('Max-plus: 1 ⊕ 1 =', max_plus(1.0, 1.0))
    print('Phi-contract examples:')
    for x in [0.2, 0.5, 1.0, 2.0]:
        y = phi_contract_to_one(x)
        print(f'  x={x:.2f} -> {y:.5f}')

def demo_convergence_plot():
    xs = []
    ys = []
    x = 2.5
    for n in range(40):
        xs.append(n)
        ys.append(x)
        x = phi_contract_to_one(x)
    plt.figure()
    plt.plot(xs, ys)
    plt.title('Convergence to Unity under φ-contraction')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.savefig('unity_lab_convergence.png', dpi=150, bbox_inches='tight')

if __name__ == '__main__':
    demo_prints()
    demo_convergence_plot()
    print('Saved plot: unity_lab_convergence.png')
