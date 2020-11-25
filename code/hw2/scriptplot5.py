import matplotlib.pyplot as plt
import math
import numpy as np

# Problem 5
cv = []
with open("5accuracy.cross") as f:
    for line in f:
        line2 = line.replace("\n", "").split(" ")
        line3 = [float(elt) for elt in line2]
        cv += [1-line3[0]/100]

test = []
with open("5accuracy.test") as f:
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
plt.title('10-Fold and Test Error with C=128 by Degree')
plt.legend()
plt.savefig("plots/5error.png")
plt.close()

totals = []
margins = []
with open("5.counts") as f:
    for line in f:
        line2 = line.replace("\n", "").split(" ")
        line3 = [int(elt) for elt in line2]
        totals += [line3[0]]
        margins += [line3[1]]

plt.plot(degrees, totals, label="Total Support Vectors")
plt.plot(degrees, margins, label="Marginal Support Vectors")
plt.grid(True)
plt.ylabel('Number of Support Vectors')
plt.xlabel('Degree')
plt.title('Number of Total and Marginal Support Vectors by Degree')
plt.legend()
plt.savefig("plots/5supports.png")
plt.close()

