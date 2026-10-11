#!/usr/bin/env python3
"""Local reference validation helper for matrix_symmetric_eigen."""
import numpy as np

def generate_case(n=32, batch_size=1, uplo='L', seed=42):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(batch_size):
        # Generate random symmetric matrix
        M = rng.standard_normal((n, n)).astype(np.float32)
        if uplo == 'L':
            A_eff = np.tril(M) + np.tril(M, -1).T
        else:
            A_eff = np.triu(M) + np.triu(M, 1).T
        
        # Ground truth using float64
        evals, evecs = np.linalg.eigh(A_eff.astype(np.float64))
        cases.append((M, A_eff, evals.astype(np.float32), evecs.astype(np.float32)))
    return cases

def verify_solution(A_eff, evals_pred, evecs_pred, evals_ref):
    """Verifies against official evaluation criteria."""
    n = A_eff.shape[0]
    # 1. Eigenvalues relative error
    eval_err = np.max(np.abs(evals_pred - evals_ref) / np.maximum(1.0, np.abs(evals_ref)))
    
    # 2. Orthogonality
    orth_err = np.linalg.norm(evecs_pred.T @ evecs_pred - np.eye(n))
    
    # 3. Reconstruction error
    recon = evecs_pred @ np.diag(evals_pred) @ evecs_pred.T
    norm_A = np.linalg.norm(A_eff)
    rec_err = np.linalg.norm(A_eff - recon) / max(1e-12, norm_A)
    
    passed = (eval_err <= 1e-4) and (orth_err <= 1e-4) and (rec_err <= 1e-4)
    return passed, eval_err, orth_err, rec_err

if __name__ == "__main__":
    cases = generate_case(n=32, batch_size=2, uplo='L')
    print(f"Generated {len(cases)} test cases.")
    for i, (M, A_eff, evals, evecs) in enumerate(cases):
        passed, ev_err, orth_err, rec_err = verify_solution(A_eff, evals, evecs, evals)
        print(f"Case {i+1}: passed={passed}, eval_err={ev_err:.2e}, orth_err={orth_err:.2e}, rec_err={rec_err:.2e}")
