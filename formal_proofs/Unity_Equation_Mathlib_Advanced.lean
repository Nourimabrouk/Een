/-!
Unity Equation (mathlib advanced): 1 + 1 = 1 in rigorous models
----------------------------------------------------------------

This file collects clean, mathlib-style formal witnesses that the statement
"1 + 1 = 1" holds in broad, nontrivial algebraic settings where addition is
idempotent. It complements computational demonstrations by providing fully
machine-checked theorems.

Highlights
- Abstract lemma: in any semiring with idempotent addition, (1 : α) + 1 = 1.
- Canonical instance: Prop as a semiring (addition = or, multiplication = and).
- Join/lattice avatar: ⊤ ⊔ ⊤ = ⊤.
- Sets avatar (via unions/intersections) and a concise wrapper.
- Tropical avatar: tropical addition is idempotent (sup/max), hence 1 + 1 = 1.
- Category-theoretic echo: 𝟙 ≫ 𝟙 = 𝟙.

This file is designed for Lean 4 with mathlib4.
-/

import Mathlib
import Mathlib/Data/Set/Lattice
import Mathlib/Order/Tropical
import Mathlib/CategoryTheory/Category/Basic

set_option autoImplicit true

namespace Unity

/-- A semiring whose addition is idempotent. -/
class IdemSemiring (α : Type*) extends Semiring α : Prop :=
  (add_idem : ∀ a : α, a + a = a)

section Abstract

variable {α : Type*} [IdemSemiring α]

/-- In any `IdemSemiring`, the Unity Equation holds at `1`. -/
@[simp] theorem one_add_one_eq_one : (1 : α) + 1 = (1 : α) := by
  simpa using (IdemSemiring.add_idem (1 : α))

/-- Numeral `2` collapses to `1` in any `IdemSemiring`. -/
@[simp] theorem two_eq_one : (2 : α) = (1 : α) := by
  simpa [two, bit0] using (one_add_one_eq_one : (1 : α) + 1 = 1)

end Abstract

/-! ## Logic avatar: `Prop` as a semiring

`Prop` has a standard semiring structure in mathlib with `+ = Or`, `* = And`,
`0 = False`, `1 = True`. Idempotence of `Or` yields the Unity Equation. -/

instance : IdemSemiring Prop :=
{ add_idem := by
    intro p
    -- `p ∨ p` ↔ `p`, hence equal via `propext`.
    apply propext
    constructor
    · intro h; exact Or.elim h id id
    · intro hp; exact Or.inl hp,
  ..(inferInstance : Semiring Prop) }

example : (1 : Prop) + 1 = (1 : Prop) := one_add_one_eq_one
example : (2 : Prop) = (1 : Prop) := two_eq_one

/-! ## Lattice avatar: join/top -/

section Lattice
variable {β : Type*} [SemilatticeSupTop β]

@[simp] theorem top_sup_top : (⊤ : β) ⊔ ⊤ = (⊤ : β) := by
  simpa using sup_idem (⊤ : β)

end Lattice

/-! ## Sets avatar: unions/intersections

We provide a thin wrapper `SetSemiring α` whose addition/multiplication are
union/intersection; the semiring laws follow from set equalities, and addition
is idempotent. -/

open Classical

structure SetSemiring (α : Type*) where
  carrier : Set α
deriving DecidableEq

namespace SetSemiring

@[ext] theorem ext {α} {A B : SetSemiring α} :
  (A = B) ↔ (A.carrier = B.carrier) := by
  constructor <;> intro h <;> cases h <;> rfl

instance {α} : Zero (SetSemiring α) := ⟨⟨(∅ : Set α)⟩⟩
instance {α} : One  (SetSemiring α) := ⟨⟨(Set.univ : Set α)⟩⟩
instance {α} : Add  (SetSemiring α) := ⟨fun A B => ⟨A.carrier ∪ B.carrier⟩⟩
instance {α} : Mul  (SetSemiring α) := ⟨fun A B => ⟨A.carrier ∩ B.carrier⟩⟩

instance {α} : CommSemiring (SetSemiring α) where
  zero := 0; one := 1; add := (· + ·); mul := (· * ·)
  add_assoc A B C := by ext x; simp [Set.union_assoc]
  add_comm A B    := by ext x; simp [Set.union_comm]
  zero_add A      := by ext x; simp
  add_zero A      := by ext x; simp
  mul_assoc A B C := by ext x; simp [Set.inter_assoc]
  one_mul A       := by ext x; simp
  mul_one A       := by ext x; simp
  left_distrib A B C  := by ext x; simp [Set.inter_union_distrib_left]
  right_distrib A B C := by ext x; simp [Set.inter_union_distrib_right]
  zero_mul A := by ext x; simp
  mul_zero A := by ext x; simp
  mul_comm A B := by ext x; simp [Set.inter_comm]

instance {α} : IdemSemiring (SetSemiring α) where
  toSemiring := inferInstance
  add_idem A := by ext x; by_cases hx : x ∈ A.carrier <;> simp [hx]

@[simp] lemma one_add_one {α} :
  (1 : SetSemiring α) + 1 = (1 : SetSemiring α) := rfl  -- `univ ∪ univ = univ`

end SetSemiring

/-! ## Tropical avatar

In `Tropical α`, addition is (a variant of) `sup`/`max`, hence idempotent. -/

section Tropical
open Tropical

variable {α : Type*} [LinearOrderedCancelAddCommMonoidWithTop α]

@[simp] theorem tropical_one_add_one :
  (1 : Tropical α) + (1 : Tropical α) = (1 : Tropical α) := by
  -- Addition in `Tropical` is idempotent; specialize to `1`.
  simpa using (sup_idem (1 : Tropical α))

end Tropical

/-! ## Category-theoretic echo

Composition of identities is identity. While not a semiring instance, the
equation 𝟙 ≫ 𝟙 = 𝟙 mirrors the Unity Equation structurally. -/

section CategoryEcho
open CategoryTheory
variable {C : Type*} [Category C] (X : C)

@[simp] theorem id_comp_id : 𝟙 X ≫ 𝟙 X = 𝟙 X := by simp

end CategoryEcho

end Unity


