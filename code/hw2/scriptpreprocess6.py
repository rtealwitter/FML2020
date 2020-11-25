from scriptpreprocess3 import importdata
import numpy as np

def tolist(xs):
    newxs = []
    for x in xs:
        newx = []
        for i in range(len(x)):
            newx += [x[i+1]]
        newxs += [np.array(newx)]
    return newxs

def tozs(xs):
    zs = []
    for i in range(len(xs)):
        xi = xs[i]
        zi = {}
        for j in range(len(xs)):
            zi[j+1] = ys[j] * np.dot(xi, xs[j])
        zs += [zi]
    return zs

def tosign(ys):
    for i in range(len(ys)):
        if ys[i] == 0:
            ys[i] = -1
    return ys

def exportdata(xs, ys, filename):
    with open(filename, "w") as f:
        for i in range(len(xs)):
            x, y = xs[i], ys[i]
            if y == -1: stry = "-1 "
            else: stry = "+1 "
            strx = str(x)[1:-1].replace(",", "").replace(": ", ":")
            f.write(stry + strx + "\n")

xs,ys = importdata("abalone.data")
xs = tolist(xs)
ys = tosign(ys)
print("generating zs...")
zs = tozs(xs)
print("writing zs...")
exportdata(zs[:3133], ys[:3133], "6.train")
exportdata(zs[3133:], ys[3133:], "6.test")

