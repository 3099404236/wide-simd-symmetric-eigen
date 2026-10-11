# 40-Core Wide-SIMD Compute Routine: Matrix Symmetric Eigen (`matrix_symmetric_eigen`)

## 1. Routine Overview
This compute routine performs the spectral decomposition of real symmetric matrices:
$$A_{\text{eff}} = Q \operatorname{diag}(\lambda) Q^T$$
where $Q \in \mathbb{R}^{n \times n}$ is an orthogonal eigenvector matrix ($Q^T Q = I$), and $\lambda = [\lambda_1, \lambda_2, \dots, \lambda_n]^T$ is the vector of real eigenvalues sorted in non-decreasing order ($\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$).

The effective symmetric matrix $A_{\text{eff}}$ is constructed from the specified triangle:
- `UPLO = 'L'`: $A_{\text{eff}} = \operatorname{tril}(A) + \operatorname{tril}(A, -1)^T$
- `UPLO = 'U'`: $A_{\text{eff}} = \operatorname{triu}(A) + \operatorname{triu}(A, 1)^T$

Unselected triangle entries in input $A$ may contain arbitrary values and are ignored.

---

## 2. Input and Output Tensors

| Tensor | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `A` | `[*, n, n]` | `float32` | Real symmetric matrix (`*` denotes 0 or more batch dimensions). |
| `eigenvalues` | `[*, n]` | `float32` | Real eigenvalues sorted in non-decreasing order ($\lambda_1 \le \dots \le \lambda_n$). |
| `eigenvectors`| `[*, n, n]` | `float32` | Orthogonal eigenvector matrix $Q$, with eigenvectors stored column by column ($Q[:, j]$). |

---

## 3. Dimension Constraints & Boundary Guarantees

- **Matrix Dimension ($n$)**: $n \in [1, 512]$.
- **Batch Dimension**: Batch size $\in [1, 64]$.
- **Eigenvalue Separation**: Input test data guarantees that all eigenvalues of $A_{\text{eff}}$ are distinct:
  $$\min_{i \ne j} |\lambda_i - \lambda_j| \ge 10^{-4} \cdot \|A_{\text{eff}}\|_2$$
  (No clustered/multiple eigenvalues near zero).

---

## 4. Precision & Verification Criteria (CRITICAL)

### A. Eigenvalues
- Evaluated element-wise with reference:
  $$\text{atol} \le 10^{-4}, \quad \text{rtol} \le 10^{-4}$$

### B. Eigenvectors: Column Sign Freedom
> [!IMPORTANT]
> **Eigenvectors have column-wide sign flip freedom ($Q[:, j] \leftrightarrow -Q[:, j]$).**
> The platform test suite **DOES NOT** perform element-by-element sign comparison against LAPACK `syevd`!
> Instead, eigenvectors are strictly validated via:
> 1. **Orthogonality Check**:
>    $$\|Q^T Q - I\|_F \le 10^{-4}$$
> 2. **Reconstruction Residual Check**:
>    $$\|A_{\text{eff}} - Q \operatorname{diag}(\lambda) Q^T\|_F \le 10^{-4} \cdot \|A_{\text{eff}}\|_F$$

> [!WARNING]
> **DO NOT** attempt to emulate LAPACK's internal eigenvector sign conventions.
> **DO NOT** execute expensive scalar triple-float iterations on $n > 25$ matrices, which risks hardware execution timeout.
> Keep numerical algorithms clean, fast, and bounded in iterations.

---

## 5. Project Layout

```text
wide-simd-symmetric-eigen/
├── README.md                              # This specification
├── specs/
│   └── symmetric_eigen_spec.md            # Detailed mathematical formulation & notes
├── include/
│   └── simd_machine.hpp                   # SIMD machine API definition (192KB SPAD)
├── kernel/
│   ├── matrix_symmetric_eigen.cpp         # Clean kernel implementation
│   ├── matrix_symmetric_eigen_layout.h    # Workload layout configuration struct
│   └── layout_key_matrix_symmetric_eigen.h# Template argument selector
├── host/
│   └── matrix_symmetric_eigen.cpp         # Host layout & workspace calculation
└── test/
    └── verify_local.py                    # Local validation script (orthogonality & reconstruction)
```

---

## 6. Implementation Strategies
- **Two-Sided Jacobi Method** (ideal for $n \le 64$):
  - Iteratively eliminates off-diagonal elements using $2 \times 2$ Givens rotations.
  - Automatically preserves orthogonality of $Q$ and yields exact reconstruction.
- **Tridiagonalization + QR / Divide-and-Conquer** (ideal for general $n \le 512$):
  - Householder tridiagonal reduction: $A = V T V^T$.
  - Implicitly shifted symmetric QR iteration or QL iteration on tridiagonal $T$.
  - Accumulate $Q = V Z$, where $Z$ diagonalizes $T$.
