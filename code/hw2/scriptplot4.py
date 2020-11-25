import matplotlib.pyplot as plt
import math
import numpy as np

# Problem 4
data = []
with open("4.accuracy") as f:
    for line in f:
        line2 = line.replace("\n", "").split(" ")
        line3 = [float(elt) for elt in line2]
        data += [line3]

n=3133
for d in [1,2,3,4]:
    c = []
    error = []
    stdup = []
    stddown = []
    for line in data:
        if line[1] == d:
            error += [1 - line[0]/100]
            c += [np.log2(line[2])]
            stddown += [error[-1] - math.sqrt((error[-1] * (1 - error[-1]))/n)]
            stdup += [error[-1] + math.sqrt((error[-1] * (1 - error[-1]))/n)]
    plt.plot(c, error, label="Error")
    plt.plot(c, stddown, label="Error - 1 SD")
    plt.plot(c, stdup, label="Error + 1 SD")
    plt.grid(True)
    plt.ylabel('Error')
    plt.xlabel('log2 C')
    plt.suptitle('10-Fold Cross-Validation (CV) Error with +/- 1 Standard Deviation (SD) for')
    plt.title('Polynomial Kernel of Degree ' + str(d))
    plt.legend()
    plt.savefig("plots/4."+str(d)+".png")
    plt.close()
