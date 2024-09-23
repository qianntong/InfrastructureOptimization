import numpy as np
import matplotlib.pyplot as plt

# Initialization of parameters
# Action
A = [0, 1, 2] # 0 : do nothing; 1: maintenance; 2: improvement
X = list(range(1,11)) # states from 1 (worst) to 10 (best)
T = list(range(1,101)) # the planning horizon from year 1 to year 100

# Transition matrix
def matrix_transform(MA):
    AM = np.zeros((10, 10))
    for i in range(0,len(AM)):
        for j in range(0,len(AM)):
            AM[9-i][9-j] = MA[i][j]
    return AM

P = np.zeros((len(A), len(X), len(X)))  # initialize three transition matrix with dimension 2*10*10

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
    [0,    0,    0,    0,    0,    0,    0,    0,    0,    1]]  # when action =0 from x_t to x_t+1

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
    [0,    0,    0,    0,    0,    0,    0,    0,    1,    0]]  # when action = 1 from x_t to x_t+1

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
    [1,    0,    0,    0,    0,    0,    0,    0,    0,    0]]  # when action = 2 from x_t to x_t+1

# Adjust the index to make sure 10 is the best stage
P[0] = matrix_transform(P[0])
P[1] = matrix_transform(P[1])
P[2] = matrix_transform(P[2])


# Salvage value
SV = {}
for i in range(1, 11):
    SV[i] = 0   # SV[1] = 10000, SV[10] = 0 (1 is the worst)
    if i <= 3:
        SV[i] = -10000

## cost of each action at each state
cost = {}
# values = [0.5, 3, 8.5, 16.5, 43.5, 53.5, 55.5, 57, 58, 58.5]
values = [58.5, 58, 57, 55.5, 53.55, 43.5, 16.5, 8.5, 3, 0.5]
for i in range(1, 11):
    cost[0, i] = 0  # do-nothing cost
    cost[1, i] = values[i - 1]  # rehabilitation cost
    cost[2, i] = 60 # reconstruction cost
print(f"the dictionary for the cost is: {cost}")

# value function
V_fn = {}
for t in T:
  for x in X:
    V_fn[t, x] = 1000

for x in X:
  V_fn[len(T), x] = - SV[x]

# optimal action
a_opt = {}

## discount factor
alpha = 0.95

# implement MDP

time = []
state = []
optimal_action = []

for t in range(99, -1, -1):  # moving backward from year 100 to year 1
  for x_i in range(len(X)): # loop through all the states
    V_0 = cost[0, X[x_i]] + alpha * sum(P[0, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    V_1 = cost[1, X[x_i]] + alpha * sum(P[1, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    V_2 = cost[2, X[x_i]] + alpha * sum(P[2, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    if x_i < 3:
        V_0 = cost[0, X[x_i]] + 1000000 + alpha * sum(P[0, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
        V_1 = cost[1, X[x_i]] + 1000000 + alpha * sum(P[1, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
        V_2 = cost[2, X[x_i]] + 1000000 + alpha * sum(P[2, x_i, j] * V_fn[t + 1, X[j]] for j in range(len(X)))
    # print(X[x_i], X[1] , P[2, x_i, 1])
    V = [V_0, V_1, V_2]
    # print(f"V is {V}")
    V_fn[t, X[x_i]] = min(V)
    a_opt[t+1, X[x_i]] = np.argmin(V)
    # restore data
    time.append(t)
    state.append(x_i)
    optimal_action.append(a_opt[t+1, X[x_i]])
    print(f"The optimal action at time {t + 1} at state {X[x_i]} is {a_opt[t + 1, X[x_i]]}")


print("\nValue function:")
print(V_fn)

# visualization
time = np.array(time)
state = np.array(state)
optimal_action = np.array(optimal_action)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = {0: 'r', 1: 'g', 2: 'b'}

for action in [0, 1, 2]:
    indices = optimal_action == action
    # ax.plot(time[indices], state[indices], optimal_action[indices], color=colors[action], label=f'Action {action}')
    ax.scatter(time[indices], state[indices], optimal_action[indices], color=colors[action], label=f'Action {action}', s=50)

ax.set_title(f"Decision Process Trace of Optimal Action (discount factor = {alpha})")
ax.set_xlabel('Time (T)')
ax.set_ylabel('State (X)')
ax.set_zlabel('Optimal Action (A)')

ax.set_zlim(0, 2)
ax.set_zticks([0, 1, 2])

ax.legend()
plt.show()