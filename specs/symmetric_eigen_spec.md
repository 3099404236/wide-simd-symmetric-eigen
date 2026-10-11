# Mathematical Specification: Symmetric Eigenvalue Decomposition

## Problem Statement
Given a real symmetric matrix $A \in \mathbb{R}^{n \times n}$ and a triangle indicator `UPLO` $\in \{\text{'L'}, \text{'U'}\}$:
Construct $A_{\text{eff}}$:
- `UPLO == 'L'`: $A_{\text{eff}, ij} = A_{ij}$ if $i \ge j$, else $A_{ji}$
- `UPLO == 'U'`: $A_{\text{eff}, ij} = A_{ij}$ if $i \le j$, else $A_{ji}$

Compute orthogonal $Q \in \mathbb{R}^{n \times n}$ ($Q^T Q = I$) and real eigenvalues $\lambda_1 \le \lambda_2 \le \dots \le \lambda_n$ such that:
$$A_{\text{eff}} Q = Q \operatorname{diag}(\lambda)$$

## Validation Protocol
1. **Eigenvalue Metric**:
   $$E_\lambda = \max_{1 \le i \le n} \frac{|\lambda_i - \lambda_i^{\text{ref}}|}{\max(1.0, |\lambda_i^{\text{ref}}|)} \le 10^{-4}$$
2. **Orthogonality Metric**:
   $$E_{\text{orth}} = \|Q^T Q - I\|_F \le 10^{-4}$$
3. **Reconstruction Metric**:
   $$E_{\text{rec}} = \frac{\|A_{\text{eff}} - Q \operatorname{diag}(\lambda) Q^T\|_F}{\|A_{\text{eff}}\|_F} \le 10^{-4}$$

Note: Ground truth reference uses float64 `torch.linalg.eigh` or `scipy.linalg.eigh`.
