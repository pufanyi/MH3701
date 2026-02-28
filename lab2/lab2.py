import numpy as np
from scipy.optimize import linprog

# Exposure matrix A[i, j]:
# ad channel i contributes A[i, j] (in 1,000s) to market group j per $1,000 spent.
A = np.array(
    [
        [0, 15, 40, 60, 40, 0, 7],   # TVL
        [0, 40, 5, 10, 1, 0, 0],     # TVP
        [6, 0, 0, 0, 0, 13, 11],     # BLB
        [1, 0, 0, 0, 0, 1, 13],      # NEW
        [0, 5, 5, 7, 6, 12, 0],      # RAD
    ],
    dtype=float,
)

MIN_EXPOSURE = np.array([100, 15, 40, 15, 120, 50, 45], dtype=float)
SATURATION = np.array([150, 110, 90, 45, 130, 140, 130], dtype=float)


def solve_for_budget(budget: float):
    """
    Maximize capped total exposure for a fixed budget.

    Variables:
    - x_i >= 0: spend on each channel i (in $1,000s)
    - z_j >= 0: credited exposure for group j, capped by saturation
    """
    n_channels, n_groups = A.shape
    n_vars = n_channels + n_groups

    # Max sum(z_j)  <=>  Min -sum(z_j)
    c = np.concatenate([np.zeros(n_channels), -np.ones(n_groups)])

    A_ub = []
    b_ub = []

    # z_j <= saturation_j
    for j in range(n_groups):
        row = np.zeros(n_vars)
        row[n_channels + j] = 1.0
        A_ub.append(row)
        b_ub.append(SATURATION[j])

    # z_j <= achieved exposure_j = sum_i A[i, j] * x_i
    for j in range(n_groups):
        row = np.zeros(n_vars)
        row[:n_channels] = -A[:, j]
        row[n_channels + j] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)

    # achieved exposure_j >= minimum_j
    for j in range(n_groups):
        row = np.zeros(n_vars)
        row[:n_channels] = -A[:, j]
        A_ub.append(row)
        b_ub.append(-MIN_EXPOSURE[j])

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_channels] = 1.0
    b_eq = np.array([budget], dtype=float)

    bounds = [(0.0, None)] * n_vars
    return linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )


def frontier(b_min: float = 20.0, b_max: float = 23.0, step: float = 0.1):
    budgets = np.arange(b_min, b_max + 1e-9, step)
    values = []
    for b in budgets:
        result = solve_for_budget(float(b))
        if not result.success:
            raise RuntimeError(f"Infeasible at budget={b:.4f}: {result.message}")
        values.append(-result.fun)
    return budgets, np.array(values)


def find_corner_budget(b_min: float = 20.0, b_max: float = 23.0):
    """
    Find the first budget in [b_min, b_max] where any non-saturated z_j reaches SATURATION[j].
    That event is a corner point on the efficient frontier.
    """
    left_result = solve_for_budget(b_min)
    right_result = solve_for_budget(b_max)
    if not left_result.success or not right_result.success:
        raise RuntimeError("Model infeasible in requested range.")

    n_groups = A.shape[1]
    left_z = left_result.x[A.shape[0] :]
    right_z = right_result.x[A.shape[0] :]

    candidates = []
    eps = 1e-8
    for j in range(n_groups):
        if left_z[j] < SATURATION[j] - eps and right_z[j] >= SATURATION[j] - eps:
            lo, hi = b_min, b_max
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                mid_result = solve_for_budget(mid)
                if not mid_result.success:
                    raise RuntimeError(f"Solve failed during bisection at budget={mid:.6f}.")
                z_mid = mid_result.x[A.shape[0] + j]
                if z_mid < SATURATION[j]:
                    lo = mid
                else:
                    hi = mid
            candidates.append(0.5 * (lo + hi))

    if not candidates:
        raise RuntimeError("No corner found in the requested budget interval.")
    return min(candidates)


def main():
    budgets, values = frontier(20.0, 23.0, 0.1)
    corner = find_corner_budget(20.0, 23.0)

    print("=" * 60)
    print("Lab 2: Efficient Frontier (Budget 20 to 23, in $1,000s)")
    print("=" * 60)
    for b, v in zip(budgets, values):
        print(f"Budget = {b:>4.1f},  Max capped exposure = {v:.6f}")

    print("\nCorner budget in [20, 23]:")
    print(f"Exact (solver precision): {corner:.6f}")
    print(f"To 1 decimal place: {corner:.1f}")


if __name__ == "__main__":
    main()
