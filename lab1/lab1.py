import numpy as np
from scipy.optimize import linprog

# ============================================================
# Multi-period planning problem
# ============================================================
# Cash needed (in $1,000s) for years 0-14
cash = [10, 11, 13, 15, 16, 18, 21, 24, 26, 28, 31, 34, 35, 38, 39]

# Savings account annual interest rate
r = 0.04

# Securities (values in $1,000s per unit of face value $1,000)
#              Cost    Coupon  Maturity(yr)  Principal
# Security 1:  0.965   0.060   13            1.000
# Security 2:  0.985   0.050   3             1.000
# Security 3:  0.975   0.085   10            1.000

n_years = 15  # years 0 through 14

# Decision variables (18 total):
# [W, x1, x2, x3, s0, s1, s2, ..., s13]
#  0   1   2   3   4   5   6  ...   17
n_vars = 4 + n_years  # W + 3 securities + 14 savings variables

# Objective: minimize W
c = np.zeros(n_vars)
c[0] = 1  # minimize W

# Equality constraints: Aeq @ x = beq
Aeq = np.zeros((n_years, n_vars))
beq = np.array(cash, dtype=float)

# Year 0: W - 0.965*x1 - 0.985*x2 - 0.975*x3 - s0 = 10
Aeq[0, 0] = 1        # W
Aeq[0, 1] = -0.965   # x1 cost
Aeq[0, 2] = -0.985   # x2 cost
Aeq[0, 3] = -0.975   # x3 cost
Aeq[0, 4] = -1       # s0

# Years 1-14
for t in range(1, n_years):
    s_idx = 4 + t  # index of s_t in variable vector

    # Savings from previous year: 1.04 * s_{t-1}
    Aeq[t, 4 + t - 1] = 1.04

    # Current savings: -s_t
    if s_idx < n_vars:
        Aeq[t, s_idx] = -1

    # Security 1: coupon years 1-13, principal at year 13
    if 1 <= t <= 13:
        Aeq[t, 1] = 0.060
    if t == 13:
        Aeq[t, 1] += 1.000  # principal repayment

    # Security 2: coupon years 1-3, principal at year 3
    if 1 <= t <= 3:
        Aeq[t, 2] = 0.050
    if t == 3:
        Aeq[t, 2] += 1.000  # principal repayment

    # Security 3: coupon years 1-10, principal at year 10
    if 1 <= t <= 10:
        Aeq[t, 3] = 0.085
    if t == 10:
        Aeq[t, 3] += 1.000  # principal repayment

# All variables >= 0
bounds = [(0, None)] * n_vars

# Solve
result = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')

print("=" * 50)
print("Multi-period Planning Problem Solution")
print("=" * 50)
print(f"Status: {result.message}")
print(f"\nMinimum lump sum needed: {result.fun:.4f} ($1,000s)")
print(f"                       = ${result.fun * 1000:,.2f}")
print(f"\nSecurity 1 (units): {result.x[1]:.4f}")
print(f"Security 2 (units): {result.x[2]:.4f}")
print(f"Security 3 (units): {result.x[3]:.4f}")
print(f"\nSavings at end of each year:")
for t in range(n_years):
    print(f"  Year {t:2d}: s_{t} = {result.x[4 + t]:.4f} ($1,000s)")
