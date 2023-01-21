import numpy as np
import math
import numpy.random
import scipy
import scipy.stats

def truncate(a):
    dist = math.sqrt(a[0]**2 + a[1]**2)
    if dist > 0.5:
        a_truncate = [x / dist * .5 for x in a]
    else:
        a_truncate = a
    return a_truncate

def torus_board_full(s, border):
    s[0] = (s[0] + border) % (2*border) - border
    s[1] = (s[1] + border) % (2*border) - border
    return s

def transition_likelihood_full(s, a, sPrime, truncate):
    a_truncate = truncate(a)
    x_pdf = scipy.stats.norm.pdf(sPrime[0], s[0] + a_truncate[0], 1)
    y_pdf = scipy.stats.norm.pdf(sPrime[1], s[1] + a_truncate[1], 1)
    return x_pdf*y_pdf

def transition_full(s, a, sPrime, torus_board): #(s, a, sPrime)?
    #I = np.ones(len(s))
    sPrime = numpy.random.normal(s[0]+truncate(a)[0], 0.25, 1)
    sPrime = np.concatenate((sPrime, numpy.random.normal(s[1]+truncate(a)[1], 0.25, 1)))
    return torus_board(sPrime)

def transition_reward_full(s, a, sPrime, r, torus_board): #?(s, a, sPrime)
    sPrime = torus_board(sPrime)
    a = truncate(a)
    dist = math.sqrt(a[0]**2 + a[1]**2)
    return r(sPrime) - 0.1*dist

def observation(sPrime, o, sigmaO): #?(sPrime, a, o)
    obs = numpy.random.normal(sPrime[0], sigmaO, 1)
    obs = np.concatenate((obs, numpy.random.normal(sPrime[1], sigmaO, 1)))
    return obs

def observation_likelihood(sPrime, o, sigmaO):
    x_pdf = scipy.stats.norm.pdf(sPrime[0], o[0], sigmaO)
    y_pdf = scipy.stats.norm.pdf(sPrime[1], o[1], sigmaO)
    return x_pdf*y_pdf

def main():
    border = 10
    r = lambda sPrime: 1
    torus_board = lambda s: torus_board_full(s, border)
    transition = lambda s, a, sPrime: transition_full(s, a, sPrime, torus_board)
    transition_reward = lambda s, a, sPrime: transition_reward_full(s, a, sPrime, r, torus_board)
    transition_likelihood = lambda s, a, sPrime: transition_likelihood_full(s, a, sPrime, truncate)


    s = [0, 0]
    a = [.6, 0]
    sigma0 = [0, 1.5, 3]
    sPrime = [.4, 0]


    sPrime = transition(s, a, sPrime)

    print(sPrime)

    sPrime_pdf = transition_likelihood(s, a, sPrime)

    print(sPrime_pdf)

    reward = transition_reward(s, a, sPrime)
    print(reward)

    obs = observation(sPrime, 1, sigma0[1])
    print(obs)
    obs_pdf = observation_likelihood(sPrime, obs, sigma0[1])
    print(obs_pdf)

if __name__ == '__main__':
    main()

#truncate
check = []
for i in range(1000):
    x = [np.random.uniform(1, 1000000000000), np.random.uniform(1, 1000000000000)]
    x = truncate(x)
    if math.sqrt(x[0]**2 + x[1]**2) <= 0.5:
        check.append(True)
print(all(check))

torus_board_full([-10, 15], 10)
torus_board_full([15, 0], 10)

#transition_full
check = []
s = [0, 0]
a = [1, 0]
for i in range(10000):
    np.random.seed(1000)
    sprime = transition_full(s, a, [100000, 0], torus_board)
    s = sprime
    check.append(s)
#why is sPrime predetermined if it's not used in the function? Changing the value of sprime while setting the seed yields no difference

import matplotlib.pyplot as plt
plt.plot(*zip(*check[:500]))

x = []
y = []
s = [0, 0]
a = [1, 1]
border = 10
torus_board = lambda s: torus_board_full(s, border)
for i in range(100000):
    sprime = transition_full(s, a, [0, 0], torus_board)
    x.append(sprime[0])
    y.append(sprime[1])
plt.hist(x, bins = 50)
plt.hist(y, bins = 50)

print(np.mean(x), np.mean(y))
print(np.std(x), np.std(y))

#observation
x = []
y = []
s = [50, 90]
a = [1, 0]
sigma0 = [0, 1.5, 3]
for i in range(10000):
    obs = observation(s, sigma0[1], 1)
    x.append(obs[0])
    y.append(obs[1])
#o not used in the function

plt.hist(x, bins = 50)
plt.hist(y, bins = 50)

print(np.mean(x), np.mean(y))
print(np.std(x), np.std(y))

d = []
s = [0, 0]
a = [1, 1]
r = lambda sPrime: 1
transition_reward_full(s, a, [50, 90], r, torus_board)

d = []
s = [0, 0]
a = [1, 1]
for i in range(len(c)):
    d.append(transition_likelihood_full(s, a, c[i], truncate))
plt.hist(d, bins = 50)

d = []
sprime = [1, 1]

for i in range(len(c)):
    d.append(observation_likelihood(sprime, c[i], 1.5))
plt.hist(d, bins = 50)

