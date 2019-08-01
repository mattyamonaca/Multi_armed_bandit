import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#ベルヌーイマシン
class BernoulliArm():
    def __init__(self,p):
        self.p = p
    
    def draw(self):
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0

#ε-greedy法
class EpsilonGreedy():
    def __init__(self,epsilon,counts,values):
        self.epsilon = epsilon #ランダムなarmを引く確率
        self.counts = counts #armを引く回数
        self.values = values #armを引いた結果得られた報酬の平均
        
    def initialize(self,n_arms):
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms) 

    def select_arm(self):
        #εより大きい乱数を引いた時にランダムでarmを選択
        if random.random() > self.epsilon: 
            return np.argmax(self.values) #最も良いarmを選択
        else:
            return random.randrange(len(self.values)) #ランダムでarmを選択
            
    def update(self,chosen_arm,reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm] #今回のarmを選択した回数
        value = self.values[chosen_arm]
        new_value = ((n-1) / float(n)) * value + (1/float(n)) * reward #Vt+1 = ((n-1)/n)Vt + R/n で与えられる報酬式
        self.values[chosen_arm] = new_value
        
def test_algorithm(algo,arms,num_sims,horizon):
    chosen_arms = np.zeros(num_sims * horizon)
    rewards = np.zeros(num_sims * horizon)
    cumulative_rewards = np.zeros(num_sims * horizon) #累積報酬
    sim_nums = np.zeros(num_sims * horizon) 
    times = np.zeros(num_sims * horizon)
    
    for sim in range(num_sims):
        if sim%200 == 0:
            print("SIM : {}".format(sim))
        sim = sim + 1
        algo.initialize(len(arms))
        
        for t in range(horizon):
            t = t+1
            index = (sim-1)*horizon + t-1
            sim_nums[index] = sim
            times[index] = t
            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm
            
            reward = arms[chosen_arm].draw()
            rewards[index] = reward
            
            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward #前回までの報酬と今回の即時報酬を加算
            algo.update(chosen_arm,reward)
            
    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]

theta = np.array([0.1, 0.1, 0.1, 0.1, 0.9])
n_arms = len(theta) 
random.shuffle(theta)

arms = map(lambda x: BernoulliArm(x), theta)  
arms = list(arms)

for epsilon in [0, 0.1, 0.2, 0.3]:
    print("epsilon = {}".format(epsilon))
    algo = EpsilonGreedy(epsilon, [], []) 
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, num_sims=2000, horizon=200) 
    df = pd.DataFrame({"times": results[1], "rewards": results[3]})
    grouped = df["rewards"].groupby(df["times"]) 
    plt.plot(grouped.mean(), label="epsilon="+str(epsilon)) 
plt.savefig("figure.png")
    
        
