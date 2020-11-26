import numpy as np
import cProfile

def importdata(filename):
    xs, ys = [], []
    with open(filename) as f:
        for line in f:
            row = line.replace("\n", "").split(",")
            y = int(int(row[-1])<=9) # classification
            # handle numeric variables
            x = [0,0,0]
            for i in range(1,len(row)-1):
                x += [float(row[i])]
            # handle categorical variable
            if row[0] == "M": x[0] = 1
            elif row[0] == "F": x[1] = 1
            elif row[0] == "I": x[2] = 1
            xs += [x]
            ys += [[-1,1][y]]
    return np.matrix(xs), np.matrix(ys).transpose()

def threshold(xi, cutoff):
    if xi <= cutoff: return -1
    elif xi > cutoff: return 1

def point(Sx, Sy, D, feature, index):
    xi = Sx[index,feature].item(0)
    yi = Sy[index,].item(0)
    mass = D[index,].item(0)
    return xi, yi, mass

def stumperror(Sx, Sy, D, feature, cutoff, pred):
    m, error = len(Sx), 0
    for index in range(m):
        xi, yi, mass = point(Sx, Sy, D, feature, index)
        prediction = pred*threshold(xi, cutoff)
        if prediction * yi == -1:
            error += mass
    return error

def handleclose(a, b, atol=.05):
    if not np.allclose(a, b, atol):
        print(a)
        print(b)
    assert np.allclose(a, b, atol)

def stumpsubroutine(Sx, Sy, D, requirement, feature):
    minerror, mincutoff, minpred = 1, np.Inf, np.Inf
    error = stumperror(Sx, Sy, D, feature, -np.Inf, 1)
    error1 = np.dot((Sy[:,] != 1).transpose(), D).item(0)
    handleclose(error, error1)
    for index in np.argsort(Sx[:, feature], axis=0):
        xi, yi, mass = point(Sx, Sy, D, feature, index)
        if -1 * yi == 1: error -= mass
        elif yi == 1: error += mass
        else: raise ValueError("Unrecognized classification.")
        if error < minerror:
            minerror, mincutoff, minpred = error, xi, 1
            #if minerror < requirement:
            #    minerror1 = stumperror(Sx, Sy, D, feature, mincutoff, minpred)
            #    if minerror1 < requirement:
            #        return minerror1, mincutoff, minpred
        if (1-error) < minerror:
            minerror, mincutoff, minpred = 1-error, xi, -1
            #if minerror < requirement:
            #    minerror1 = stumperror(Sx, Sy, D, feature, mincutoff, minpred)
            #    if minerror1 < requirement:
            #        return minerror1, mincutoff, minpred
    minerror1 = stumperror(Sx, Sy, D, feature, mincutoff, minpred)
    if minerror1 < requirement:
        return minerror1, mincutoff, minpred
    raise ValueError("No stump found.")

def stump(Sx, Sy, D, requirement, feature):
    minerror, mincutoff, minpred = stumpsubroutine(Sx, Sy, D, requirement, feature)
    ht = []
    for index in range(Sx.shape[0]):
        ht += [[-1*minpred,minpred][int(Sx[index,feature] > mincutoff)]]
    ht = np.matrix(ht).transpose()
    assert minerror < requirement
    weightedprediction = np.multiply(D, np.multiply(ht, Sy)<0)
    minerror1 = np.sum(weightedprediction).item(0)
    handleclose(minerror1, minerror, atol=0)
    return ht, minerror, mincutoff, minpred

def adaboost(Sx, Sy, T, requirement, feature):
    m = len(Sx)
    D = {0:np.zeros((m,1))}
    alphas, cutoffs, preds = [], [], []
    for i in range(m):
        D[0][i,] = 1/m
    for t in range(T):
        ht, errort, cutofft, predt = stump(Sx, Sy, D[t], requirement, feature)
        alphat = .5 * np.log((1-errort)/errort)
        Zt = 2 * np.sqrt(errort*(1-errort))
        D[t+1] = np.zeros((m,1))
        for index in range(m):
            yi, hti, masst = Sy[index,].item(0), ht[index,].item(0), D[t][index,].item(0)
            D[t+1][index,] = masst * np.e**(-1*alphat * hti*yi)/Zt
        handleclose(np.sum(D[t+1]), 1, atol=.05)
        alphas += [alphat]
        cutoffs += [cutofft]
        preds += [predt]
    return alphas, cutoffs, preds

def buildf(alphas, cutoffs, preds, feature):
    def f(Sx, index):
        result = 0
        xi = Sx[index,feature].item(0)
        for j in range(len(alphas)):
            result += [-1*preds[j], preds[j]][xi>cutoffs[j]]
        return [-1,1][result > 0]
    return f

def buildlogisticZ(Sx, Sy, f):
    Z, m = 0, Sx.shape[0]
    for index in range(m):
        yi = Sy[index,].item(0)
        exp = np.e**(-1*yi*f(Sx,index))
        Z += exp / (np.log(2)*(1+exp))
    return Z

def logisticloss(Sx, Sy, T, requirement, feature):
    m = len(Sx)
    D = {0:np.zeros((m,1))}
    h = np.zeros((m,1))
    alphas, cutoffs, preds = [], [], []
    for i in range(m):
        D[0][i,] = 1/m
    for t in range(T):
        ht, errort, cutofft, predt = stump(Sx, Sy, D[t], requirement, feature)
        alphat = .5 * np.log((1-errort)/errort)
        for index in range(m):
            h[index,] = h[index,].item(0) + alphat*ht[index,].item(0)
        #f = buildfunction(alpha, h)
        #Z = buildlogisticZ(Sx, Sy, f)
        D[t+1] = np.zeros((m,1))
        for index in range(m):
            yi, hi = Sy[index,].item(0), h[index,].item(0)
            exp = np.e**(-1*yi*hi)
            D[t+1][index,] = exp / (np.log(2) * (1 + exp))#/Z
        D[t+1] = D[t+1]/np.sum(D[t+1]) # faster than calculating Z
        handleclose(np.sum(D[t+1]), 1)
        alphas += [alphat]
        cutoffs += [cutofft]
        preds += [predt]
    return alphas, cutoffs, preds

def evalerror(f, Sx, Sy):
    error = 0
    m = Sx.shape[0]
    for index in range(m):
        yi = Sy[index,].item(0)
        fi = f(Sx, index)
        if fi*yi == -1:
            error += 1/m
    return error

def constructgroups(m, k):
    groups = {}
    for i in range(k):
        groups[i] = []
    indices = list(range(m))
    shuffledindices = np.random.permutation(indices)
    for index in indices:
        groups[index % k] += [shuffledindices[index].item(0)]
    return groups

def crossvalidate(xs, ys, k, algorithm, T, feature):
    m, trainerr, testerr = len(xs), 0, 0
    groups = constructgroups(m,k)
    for i in range(k):
        testidx = groups[i]
        trainidx = []
        for j in range(k):
            if j != i: trainidx += groups[j]
        alphas, cutoffs, preds = algorithm(Sx=xs[trainidx], Sy=ys[trainidx], T=T, requirement=.5, feature=feature)
        f = buildf(alphas, cutoffs, preds, feature)
        trainerr += evalerror(f, xs[trainidx], ys[trainidx])
        testerr += evalerror(f, xs[testidx], ys[testidx])
    return trainerr/k, testerr/k

def crossvalidateT(xs, ys, k, Ts, feature=3):
    for T in Ts:
        atrainavg, atestavg = crossvalidate(xs=xs, ys=ys, k=k, algorithm=adaboost, T=T, feature=feature)
        astring = " ".join([str(i) for i in [T, atrainavg, atestavg, "\n"]])        
        with open("erroradaboost.txt", "a") as f: f.write(astring)
        print(astring)
        ltrainavg, ltestavg = crossvalidate(xs=xs, ys=ys, k=k, algorithm=logisticloss, T=T, feature=feature)
        lstring = " ".join([str(i) for i in [T, ltrainavg, ltestavg, "\n"]])
        print(lstring)
        with open("errorlogistic.txt", "a") as f: f.write(lstring)


np.random.seed(1)

Ts = [10]#0000]
xs, ys = importdata("abalone.data")
cProfile.run("crossvalidateT(xs, ys, 10, Ts)")


# Toy example
#Sx = np.matrix([.23, .51, .01, .42, .63, .15, .91, .37]).transpose()
#Sy = np.matrix([-1, 1, 1, 1, -1, -1, 1, -1]).transpose()
#D = np.matrix([.11,.09,.109,.091,.105,.095, .19, .21]).transpose()
#stump(Sx, Sy,D)


