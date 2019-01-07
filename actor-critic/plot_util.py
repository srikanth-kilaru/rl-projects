import numpy as np
import matplotlib.pyplot as plt


lines = [line.rstrip('\n') for line in open('pg_log.txt')]
lines = lines[1:]
pg_mean_returns = []
iterations = []
i = 1

for line in lines:
    vals = [float(n) for n in line.split()]
    iterations.append(i)
    pg_mean_returns.append(vals[2])
    i += 1

lines = [line.rstrip('\n') for line in open('ac_log.txt')]
lines = lines[1:]
ac_mean_returns = []

for line in lines:
    vals = [float(n) for n in line.split()]
    ac_mean_returns.append(vals[2])

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.set_title('Average returns')
plt.plot(iterations, pg_mean_returns, label='Policy Gradient')
plt.plot(iterations, ac_mean_returns, label='Actor-Critic')
plt.legend(loc='lower right')
plt.show()
