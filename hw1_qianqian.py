import numpy as np

# Initialization of parameters
# action
A = [0, 1, 2] # 0 : do nothing; 1: maintenance; 2: improvement
X = list(range(1,11)) # states from 1 (worst) to 10 (best)
T = list(range(1,101)) # the planning horizon from year 1 to year 100

## Transition matrix
P = np.zeros((len(A), len(X), len(X))) # transition matrix with dimension 2*10*10
P[0] = [
    [0.69, 0.31, 0,    0,    0,    0,    0,    0,    0,    0],
    [0,    0.77, 0.23, 0,    0,    0,    0,    0,    0,    0],
    [0,    0,    0.92, 0.08, 0,    0,    0,    0,    0,    0],
    [0,    0,    0,    0.91, 0.09, 0,    0,    0,    0,    0],
    [0,    0,    0,    0,    0.9,  0.1,  0,    0,    0,    0],
    [0,    0,    0,    0,    0,    0.79, 0.21, 0,    0,    0],
    [0,    0,    0,    0,    0,    0,    0.5,  0.5,  0,    0],
    [0,    0,    0,    0,    0,    0,    0,    1,    0,    0],
    [0,    0,    0,    0,    0,    0,    0,    0,    1,    0],
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    1]] # when action =0 from x_t to x_t+1

P[1] = [
    [0.69, 0.31, 0,    0,    0,    0,    0,    0,    0,    0],
    [0.69, 0.31, 0,    0,    0,    0,    0,    0,    0,    0],
    [0,    0.77, 0.23, 0,    0,    0,    0,    0,    0,    0],
    [0,    0,    0.92, 0.08, 0,    0,    0,    0,    0,    0],
    [0,    0,    0,    0.91, 0.09, 0,    0,    0,    0,    0],
    [0,    0,    0,    0,    0.9,  0.1,  0,    0,    0,    0],
    [0,    0,    0,    0,    0,    0.79, 0.21, 0,    0,    0],
    [0,    0,    0,    0,    0,    0,    0.5,  0.5,  0,    0],
    [0,    0,    0,    0,    0,    0,    0,    1,    0,    0],
    [0,    0,    0,    0,    0,    0,    0,    0,    1,    0]] # when action = 1 from x_t to x_t+1

n = 10
# P[2] = np.eye(n) # when action = 2 from x_t to x_t+1
P[2] = [
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0],
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0]] # when action = 2 from x_t to x_t+1

## Savalge value
SV = {}
for i in range(1, 11):
    SV[i] = 0   # SV[100]=100, and SV[1]=1 (1 is the worst)
    if i <= 3:
        SV[i] = 100000

print(SV)

## cost of each action at each state
cost = {}
values = [0.5, 3, 8.5, 16.5, 43.5, 53.5, 55.5, 57, 58, 58.5]
for i in range(1, 11):
    cost[0, i] = 0  # do-nothing cost
    cost[1, i] = values[i - 1]  # rehabilitation cost
    cost[2, i] = 60 # reconstruction cost
print(f"the dictionary for the cost is: {cost}")

# value function
V_fn = {}
for t in T:
  for x in X:
    V_fn[t, x] = t

for x in X:
  V_fn[len(T), x] = - SV[x]

# optimal action
a_opt = {}

## discount factor
alpha = 1

# implement MDP
for t in reversed(T[:-1]):
# for t in range(98, -1, -1): # moving backward from year 9 to year 1
  for x_i in range(len(X)): # loop through all the states
    V_0 = cost[0, X[x_i]] + alpha * sum(P[0, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    V_1 = cost[1, X[x_i]] + alpha * sum(P[1, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    V_2 = cost[2, X[x_i]] + alpha * sum(P[2, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    V = [V_0, V_1, V_2]
    V_fn[t, X[x_i]] = min(V)
    a_opt[t+1, X[x_i]] = np.argmin(V)
    # if np.argmin(V) != 0:
    print(f"The optimal action at time {t + 1} at state {X[x_i]} is {a_opt[t + 1, X[x_i]]}")

# for t in reversed(T[:-1]):  # Iterate backwards through time
#     for x in X:
#         V_values = []
#         for a in A:
#             expected_value = sum(P[a, x - 1, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
#             V_a = cost[a, x] + alpha * expected_value
#             V_values.append(V_a)
#
#         # Store the optimal value and action for state x at time t
#         V_fn[t, x] = min(V_values)
#         a_opt[t, x] = A[np.argmin(V_values)]
#         if np.argmin(V_fn) != 0:
#             print(f"The optimal action at time {t + 1} at state {X[x_i]} is {a_opt[t + 1, X[x_i]]}")
#
# # Output the optimal policy and value function
# print("Optimal actions by state and time:")
# print(a_opt)
#
print("\nValue function:")
print(V_fn)