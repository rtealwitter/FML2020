import matplotlib.pyplot as plt
import math
import numpy as np

# Problem 6
cv = []
with open("6accuracy.cross") as f:
    for line in f:
        line2 = line.replace("\n", "").split(" ")
        line3 = [float(elt) for elt in line2]
        cv += [1-line3[0]/100]

test = []
with open("6accuracy.test") as f:
    for line in f:
        line2 = line.replace("\n", "").split(" ")
        line3 = [float(elt) for elt in line2]
        test += [1-line3[0]/100]

degrees = [i + 1 for i in range(len(test))]

plt.plot(degrees, cv, label="10-Fold CV Error")
plt.plot(degrees, test, label="Test Error")
plt.grid(True)
plt.ylabel('Error')
plt.xlabel('Degree')
plt.title('10-Fold CV and Test Error with C=32 by Degree')
plt.legend()
plt.savefig("plots/6error.png")
plt.close()
