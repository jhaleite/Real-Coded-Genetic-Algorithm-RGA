import External_Functions
from RGA_Bib import RGA
import matplotlib.pyplot as plt
import numpy as np

obj = RGA(40, 4, 1000, 5, -5, 1, 15, 20, 0.01)
obj.RGA_Optimizer(External_Functions.Rosembrock_Function, 'min')


print(obj.X_j)
print(obj.fx)

plt.figure(1)
plt.rcParams.update({'font.size': 12})
plt.plot(obj.Memory_t, obj.Memory_fx,label=('Best Fitness RGA'))
#plt.plot(obj1.Memory_t, obj1.Memory_Gb,label=('Best Fitness PSO'))

#plt.plot(obj2.Memory_t, obj2.Memory_Gb,label=('Best Fitness'))
#plt.plot(obj3.Memory_t, obj3.Memory_Gb,label=('Best Fitness'))
#plt.xticks(daytime)
plt.xlabel(' Generation ')
plt.ylabel(' Best Fitness')
plt.legend()
plt.grid(True)
#plt.savefig('Power_Flow_results.png')
plt.show()

