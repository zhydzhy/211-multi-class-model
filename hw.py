from pulp import LpProblem, LpVariable, lpSum, LpMinimize

# Define power consumption levels
P = {1: 0.5, 2: 1.0, 3: 2.0}  # Power in Watts

# Define execution times at the lowest frequency (500MHz)
C = {
    1: {1: 10, 2: 10 * 500 / 800, 3: 10 * 500 / 1000},
    2: {1: 5, 2: 5 * 500 / 800, 3: 5 * 500 / 1000},
    3: {1: 7, 2: 7 * 500 / 800, 3: 7 * 500 / 1000}
}

# Deadlines
D = {1: 20, 2: 10, 3: 12}

# Create ILP problem
prob = LpProblem("Task_Scheduling", LpMinimize)

# Define variables: x_{i,j} = 1 if task i runs at mode j
x = {(i, j): LpVariable(f"x_{i}_{j}", cat='Binary') for i in range(1, 4) for j in range(1, 4)}

# Objective: Minimize total energy consumption
prob += lpSum(P[j] * C[i][j] * x[i, j] for i in range(1, 4) for j in range(1, 4))

# Each task must be assigned exactly one frequency mode
for i in range(1, 4):
    prob += lpSum(x[i, j] for j in range(1, 4)) == 1

# Ensure deadlines are met with cumulative execution time constraints
prob += lpSum(C[1][j] * x[1, j] for j in range(1, 4)) <= D[1]
prob += lpSum(C[1][j] * x[1, j] for j in range(1, 4)) + lpSum(C[2][j] * x[2, j] for j in range(1, 4)) <= D[2]
prob += (lpSum(C[1][j] * x[1, j] for j in range(1, 4)) +
         lpSum(C[2][j] * x[2, j] for j in range(1, 4)) +
         lpSum(C[3][j] * x[3, j] for j in range(1, 4))) <= D[3]

# Solve the problem
prob.solve()

# Print results
for i in range(1, 4):
    for j in range(1, 4):
        if x[i, j].varValue == 1:
            print(f"Task {i} runs at mode {j} with (v{j}, f{j})")

print("Total Energy Consumption:", prob.objective.value(), "mJ")
