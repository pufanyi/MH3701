using JuMP, HiGHS

# ============================================================
# Lab 4: Two-Person Zero-Sum Game — Minimax Strategies via LP
# ============================================================
#
# Payoff matrix (from Blue to Gold):
#
#              Blue's Move
#              a     b     c     d     e     f
# Gold  a  [  8   -13    14    20     1    16 ]
#       b  [-10   -16    14    15    -9    -9 ]
#       c  [ -1   -13   -15    15    12    -4 ]
#       d  [-13    -4     1     1    17   -10 ]
#
# Blue (column player) PAYS Gold (row player).
#   - Gold wants to MAXIMIZE the payoff.
#   - Blue wants to MINIMIZE the payoff.
# ============================================================

A = [
      8  -13   14   20    1   16;
    -10  -16   14   15   -9   -9;
     -1  -13  -15   15   12   -4;
    -13   -4    1    1   17  -10
]

n_gold, n_blue = size(A)  # 4 rows (Gold), 6 columns (Blue)

blue_labels = ["a", "b", "c", "d", "e", "f"]
gold_labels = ["a", "b", "c", "d"]

# ============================================================
# Blue's LP (minimizer — column player)
# ============================================================
#
# Minimize  w
# s.t.  ∑_j A[i,j] x[j] ≤ w   ∀ i (Gold's moves)
#       ∑_j x[j] = 1
#       x[j] ≥ 0,  w free
# ============================================================

blue_model = Model(HiGHS.Optimizer)
set_silent(blue_model)

@variable(blue_model, x[1:n_blue] >= 0)
@variable(blue_model, w)
@objective(blue_model, Min, w)
@constraint(blue_model, [i = 1:n_gold], sum(A[i, j] * x[j] for j in 1:n_blue) <= w)
@constraint(blue_model, sum(x) == 1)

optimize!(blue_model)

# ============================================================
# Gold's LP (maximizer — row player)
# ============================================================
#
# Maximize  v
# s.t.  ∑_i A[i,j] y[i] ≥ v   ∀ j (Blue's moves)
#       ∑_i y[i] = 1
#       y[i] ≥ 0,  v free
# ============================================================

gold_model = Model(HiGHS.Optimizer)
set_silent(gold_model)

@variable(gold_model, y[1:n_gold] >= 0)
@variable(gold_model, v)
@objective(gold_model, Max, v)
@constraint(gold_model, [j = 1:n_blue], sum(A[i, j] * y[i] for i in 1:n_gold) >= v)
@constraint(gold_model, sum(y) == 1)

optimize!(gold_model)

# ============================================================
# Output
# ============================================================
println("=" ^ 60)
println("Lab 4: Two-Person Zero-Sum Game — Minimax Strategies")
println("=" ^ 60)

println("\nBlue's optimal strategy (minimizer):")
for j in 1:n_blue
    println("  BM_$(blue_labels[j]) = $(value(x[j]))")
end
println("  Value of game (Blue's LP): w = $(value(w))")

println("\nGold's optimal strategy (maximizer):")
for i in 1:n_gold
    println("  GM_$(gold_labels[i]) = $(value(y[i]))")
end
println("  Value of game (Gold's LP): v = $(value(v))")

println("\nValue of the game: $(value(w))")
