import numpy as np
from scipy.optimize import linprog

# ============================================================
# Lab Test 3: PERT Network & Critical Path via LP
# ============================================================
#
# Activities table
# ---------------------------------------------------------------
# Activity  Time  Predecessors
# A1        10    -
# A2        13    -
# A3         8    -
# A4        16    A2
# A5        23    A2
# A6        18    A1, A4
# A7         9    A2
# A8        19    A2
# A9        15    A3, A5
# A10       14    A3, A5
# A11       14    A8, A9, A12
# A12       16    A10
# A13        4    A6, A7, A11
# A14        4    A8, A9, A12
# A15        8    A10
# A16       25    A10
# A17       23    A10
# A18        9    A16
# A19        9    A16
# A20       12    A13, A14, A15, A18
# A21        2    A16
# A22        3    A17, A19
# ---------------------------------------------------------------
#
# PERT Network (Activity-on-Arrow)
# ---------------------------------------------------------------
# Nodes (Events):
#   Node 1 : Project Start
#   Node 2 : End of A2
#   Node 3 : End of A1 & A4  (merge → start A6)
#   Node 4 : End of A3 & A5  (merge → start A9, A10)
#   Node 5 : End of A10      (start A12, A15, A16, A17)
#   Node 6 : End of A8, A9 & A12  (merge → start A11, A14)
#   Node 7 : End of A6, A7 & A11  (merge → start A13)
#   Node 8 : End of A16      (start A18, A19, A21)
#   Node 9 : End of A13, A14, A15 & A18  (merge → start A20)
#   Node 10: End of A17 & A19 (merge → start A22)
#   Node 11: Project End
#
# Arrows (Activities):
#   A1:  1 → 3   (d=10)     A12: 5 → 6   (d=16)
#   A2:  1 → 2   (d=13)     A13: 7 → 9   (d= 4)
#   A3:  1 → 4   (d= 8)     A14: 6 → 9   (d= 4)
#   A4:  2 → 3   (d=16)     A15: 5 → 9   (d= 8)
#   A5:  2 → 4   (d=23)     A16: 5 → 8   (d=25)
#   A6:  3 → 7   (d=18)     A17: 5 → 10  (d=23)
#   A7:  2 → 7   (d= 9)     A18: 8 → 9   (d= 9)
#   A8:  2 → 6   (d=19)     A19: 8 → 10  (d= 9)
#   A9:  4 → 6   (d=15)     A20: 9 → 11  (d=12)
#   A10: 4 → 5   (d=14)     A21: 8 → 11  (d= 2)
#   A11: 6 → 7   (d=14)     A22: 10→ 11  (d= 3)
#
# No dummy activities are required.
# ---------------------------------------------------------------

# Activity data: (name, from_node, to_node, duration)
activities = [
    ("A1",   1,  3,  10),
    ("A2",   1,  2,  13),
    ("A3",   1,  4,   8),
    ("A4",   2,  3,  16),
    ("A5",   2,  4,  23),
    ("A6",   3,  7,  18),
    ("A7",   2,  7,   9),
    ("A8",   2,  6,  19),
    ("A9",   4,  6,  15),
    ("A10",  4,  5,  14),
    ("A11",  6,  7,  14),
    ("A12",  5,  6,  16),
    ("A13",  7,  9,   4),
    ("A14",  6,  9,   4),
    ("A15",  5,  9,   8),
    ("A16",  5,  8,  25),
    ("A17",  5, 10,  23),
    ("A18",  8,  9,   9),
    ("A19",  8, 10,   9),
    ("A20",  9, 11,  12),
    ("A21",  8, 11,   2),
    ("A22", 10, 11,   3),
]

n_nodes = 11
n_activities = len(activities)

# ============================================================
# LP Model for Finding the Critical Path
# ============================================================
#
# Decision variables:
#   E_j  = earliest event time at node j,  j = 1, 2, ..., 11
#
# Objective:
#   Minimize  E_11   (project completion time)
#
# Subject to:
#   E_1 = 0                              (project starts at time 0)
#   E_j >= E_i + d_{ij}   for each activity (i → j) with duration d_{ij}
#   E_j >= 0              for all j
#
# The full set of constraints:
#   E_3  >= E_1  + 10     (A1)
#   E_2  >= E_1  + 13     (A2)
#   E_4  >= E_1  +  8     (A3)
#   E_3  >= E_2  + 16     (A4)
#   E_4  >= E_2  + 23     (A5)
#   E_7  >= E_3  + 18     (A6)
#   E_7  >= E_2  +  9     (A7)
#   E_6  >= E_2  + 19     (A8)
#   E_6  >= E_4  + 15     (A9)
#   E_5  >= E_4  + 14     (A10)
#   E_7  >= E_6  + 14     (A11)
#   E_6  >= E_5  + 16     (A12)
#   E_9  >= E_7  +  4     (A13)
#   E_9  >= E_6  +  4     (A14)
#   E_9  >= E_5  +  8     (A15)
#   E_8  >= E_5  + 25     (A16)
#   E_10 >= E_5  + 23     (A17)
#   E_9  >= E_8  +  9     (A18)
#   E_10 >= E_8  +  9     (A19)
#   E_11 >= E_9  + 12     (A20)
#   E_11 >= E_8  +  2     (A21)
#   E_10 >= E_10 +  3     ... wait, A22: 10 → 11
#   E_11 >= E_10 +  3     (A22)
#   E_1  =  0
#   E_j  >= 0   for all j
# ============================================================

# Variables: E_1 .. E_11  (indices 0..10 in the array)
n_vars = n_nodes

# Objective: minimize E_11 (index 10)
c = np.zeros(n_vars)
c[10] = 1.0  # minimize E_11

# Inequality constraints: E_i + d_ij <= E_j  =>  E_i - E_j <= -d_ij
A_ub = []
b_ub = []

for name, i, j, d in activities:
    row = np.zeros(n_vars)
    row[i - 1] = 1.0   # E_i
    row[j - 1] = -1.0  # -E_j
    A_ub.append(row)
    b_ub.append(-d)     # <= -d_ij

A_ub = np.array(A_ub)
b_ub = np.array(b_ub, dtype=float)

# Equality constraint: E_1 = 0
A_eq = np.zeros((1, n_vars))
A_eq[0, 0] = 1.0
b_eq = np.array([0.0])

# All E_j >= 0
bounds = [(0, None)] * n_vars

# Solve the LP
result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# ============================================================
# Output results
# ============================================================
print("=" * 60)
print("Lab Test 3: PERT Critical Path via LP")
print("=" * 60)
print(f"Status: {result.message}\n")

E = result.x
print("Event (earliest) times:")
for j in range(n_nodes):
    print(f"  E_{j + 1:>2d} = {E[j]:>6.1f}")

print(f"\nMinimum project completion time: {E[10]:.1f}")

# Identify critical activities (binding constraints: E_j - E_i = d_ij)
print("\nCritical activities (E_j - E_i = d_ij):")
critical = []
for k, (name, i, j, d) in enumerate(activities):
    slack = E[j - 1] - E[i - 1] - d
    if abs(slack) < 1e-6:
        critical.append(name)
        print(f"  {name:>3s}: node {i:>2d} -> node {j:>2d},  d = {d:>2d},  "
              f"E_{j} - E_{i} = {E[j - 1]:.1f} - {E[i - 1]:.1f} = {d:.1f}")

# Trace critical path(s) from node 1 to node 11
print("\nCritical path(s) from node 1 to node 11:")
critical_set = {(i, j) for name, i, j, d in activities
                if abs(E[j - 1] - E[i - 1] - d) < 1e-6}


def find_paths(node, target, path):
    """DFS to enumerate all critical paths."""
    if node == target:
        yield list(path)
        return
    for name, i, j, d in activities:
        if i == node and (i, j) in critical_set:
            path.append(name)
            yield from find_paths(j, target, path)
            path.pop()


for idx, p in enumerate(find_paths(1, 11, []), 1):
    durations = []
    for name in p:
        for n, i, j, d in activities:
            if n == name:
                durations.append(d)
                break
    path_str = " -> ".join(p)
    dur_str = " + ".join(str(d) for d in durations)
    print(f"  Path {idx}: {path_str}")
    print(f"          {dur_str} = {sum(durations)}")
