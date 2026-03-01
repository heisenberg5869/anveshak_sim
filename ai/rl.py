import numpy as np
max_f = 10
max_n = 5
prev_actions = [0, 1]

R_PROGRESS = 10
R_SAFE_TIME = -5
R_FAST_TIME = -2
R_FAILURE = -50
R_SUCCESS_BONUS = 20

V = np.zeros((max_n + 1, max_f + 1, 2, 2), dtype=float)

for n in range(1, max_n + 1):
    for f in range(max_f):
        for p in prev_actions:
            # --- ACTION: SAFE ---
            v_safe = 0
            for df, prob in [(1, 0.8), (2, 0.2)]:
                f_next = f + df
                if f_next >= max_f:
                    v_safe += prob * R_FAILURE
                else:
                    reward = R_PROGRESS + R_SAFE_TIME
                    if n == 1:
                        reward += R_SUCCESS_BONUS
                    # CORRECTED: Use np.max over both actions (axis=-1 gets max over actions)
                    v_safe += prob * (reward + np.max(V[n-1][f_next][0]))

            # --- ACTION: FAST ---
            v_fast = 0
            increments = [(3, 0.7), (4, 0.3)] if p == 0 else [(5, 0.6), (7, 0.4)]
            for df, p_base in increments:
                f_base = f + df
                if f_base >= 8:
                    tear_outcomes = [(f_base, 0.8), (f_base + 4, 0.2)]
                else:
                    tear_outcomes = [(f_base, 1.0)]
                
                for f_final, p_tear in tear_outcomes:
                    total_prob = p_base * p_tear
                    if f_final >= max_f:
                        v_fast += total_prob * R_FAILURE
                    else:
                        reward = R_PROGRESS + R_FAST_TIME
                        if n == 1:
                            reward += R_SUCCESS_BONUS
                        # CORRECTED: Use np.max over both actions
                        v_fast += total_prob * (reward + np.max(V[n-1][f_final][1]))

            V[n][f][p][0] = v_safe
            V[n][f][p][1] = v_fast

            
policy = np.zeros((6, 11, 2), dtype=int)
for n in range(1, 6):
    for f in range(10):
        for p in [0, 1]:
            if V[n][f][p][0] >= V[n][f][p][1]:
                policy[n][f][p] = 0
            else:
                policy[n][f][p] = 1

print(policy)

def simulate_mission():
    f = 0
    p = 0
    total_reward = 0
    for n in range(5, 0, -1):
        # 1. Choose action based on optimal policy
        action = policy[n][f][p]
        
        if action == 0:  # SAFE
            total_reward += (10 - 5)
            f += np.random.choice([1, 2], p=[0.8, 0.2])
            p = 0
        else:  # FAST
            total_reward += (10 - 2)
            inc = np.random.choice([3, 4], p=[0.7, 0.3]) if p == 0 else \
                  np.random.choice([5, 7], p=[0.6, 0.4])
            f += inc
            # Fragility Check
            if f >= 8 and np.random.random() < 0.2:
                f += 4
            p = 1
        
        if f >= 10:
            return False, total_reward - 50, f  # Failure
    
    return True, total_reward + 20, f  # Success

# Run 1000 simulations
results = [simulate_mission() for _ in range(1000)]
successes = [r for r in results if r[0]]
failures = [r for r in results if not r[0]]

prob_failure = len(failures) / 1000
avg_reward = np.mean([r[1] for r in results])
avg_fatigue = np.mean([r[2] for r in results])

print(f"Probability of Failure: {prob_failure}")
print(f"Average Reward: {avg_reward:.2f}")
print(f"Average Fatigue: {avg_fatigue:.2f}")