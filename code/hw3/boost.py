import numpy as np

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
    #assert np.allclose(a, b, atol)

def stumpsubroutine(Sx, Sy, D, requirement):
    minerror0, mincutoff0, minfeature0 = 1, 0, 0
    for feature in list(range(3, Sx.shape[1])):
        error = stumperror(Sx, Sy, D, feature, -np.Inf, 1)
        error1 = np.dot((Sy[:,] != 1).transpose(), D).item(0)
        handleclose(error, error1)
        for index in np.argsort(Sx[:, feature], axis=0):
            xi, yi, mass = point(Sx, Sy, D, feature, index)
            if -1 * yi == 1:
                error -= mass
            elif yi == 1:
                error += mass
            else: raise ValueError("Unrecognized classification.")
            if error < minerror0:
                minerror0, mincutoff0, minfeature0, minpred0 = error, xi, feature, 1
                #if minerror0 < requirement:
                #    error = stumperror(Sx, Sy, D, minfeature0, mincutoff0, minpred0)
                #    if error < requirement:
                #        return error, mincutoff0, minfeature0, minpred0
            if (1-error) < minerror0:
                minerror0, mincutoff0, minfeature0, minpred0 = 1-error, xi, feature, -1
                if minerror0 < requirement:
                    error = 1-stumperror(Sx, Sy, D, minfeature0, mincutoff0, minpred0)
                    if 1-error < requirement:
                        return 1-error, mincutoff0, minfeature0, minpred0
        if stumperror(Sx, Sy, D, minfeature0, mincutoff0, minpred0) < requirement:
            return minerror0, mincutoff0, minfeature0, minpred0
    return minerror0, mincutoff0, minfeature0, minpred0

def stump(Sx, Sy, D, requirement):
    minerror, mincutoff, minfeature, minpred = stumpsubroutine(Sx, Sy, D, requirement) 
    minerror1 = stumperror(Sx, Sy, D, minfeature, mincutoff, minpred)
    handleclose(minerror, minerror1, atol=.1)
    def ht(Sx, i):
        xi = Sx[i,minfeature]
        return minpred*threshold(xi, mincutoff)
    assert minerror1 < .5
    return ht, minerror1

def adaboost(Sx, Sy, T, requirement):
    m = len(Sx)
    D = {0:np.zeros((m,1))}
    alpha, h = [], []
    for i in range(m):
        D[0][i,] = 1/m
    for t in range(T):
        ht, errort = stump(Sx, Sy, D[t], requirement)
        alphat = .5 * np.log((1-errort)/errort)
        Zt = 2 * np.sqrt(errort*(1-errort))
        D[t+1] = np.zeros((m,1))
        for i in range(m):
            yi = Sy[i,].item(0)
            D[t+1][i,] = D[t][i,] * np.e**(-1*alphat * ht(Sx, i)*yi)/Zt
        handleclose(np.sum(D[t+1]), 1, atol=.05)
        alpha += [alphat]
        h += [ht]
    return buildfunction(alpha, h)

def buildfunction(alpha, h):
    def f(Sx, i):
        result = 0
        for j in range(len(alpha)):
            result += h[j](Sx, i) * alpha[j]
        return threshold(result, 0)
    return f

def buildlogisticZ(Sx, Sy, f):
    Z, m = 0, Sx.shape[0]
    for index in range(m):
        yi = Sy[index,].item(0)
        exp = np.e**(-1*yi*f(Sx,index))
        Z += exp / (np.log(2)*(1+exp))
    return Z

def logisticloss(Sx, Sy, T, requirement):
    m = len(Sx)
    D = {0:np.zeros((m,1))}
    alpha, h = [], []
    for i in range(m):
        D[0][i,] = 1/m
    for t in range(T):
        ht, errort = stump(Sx, Sy, D[t], requirement)
        alphat = .5 * np.log((1-errort)/errort)
        alpha += [alphat]
        h += [ht]        
        f = buildfunction(alpha, h)
        Z = buildlogisticZ(Sx, Sy, f)
        D[t+1] = np.zeros((m,1))
        for index in range(m):
            yi = Sy[index,].item(0)
            exp = np.e**(-1*yi*f(Sx, index))
            D[t+1][index,] = exp / (np.log(2) * (1 + exp)*Z)
        handleclose(np.sum(D[t+1]), 1)
    return f

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

def crossvalidate(xs, ys, k, algorithm, T):
    m, trainerr, testerr = len(xs), 0, 0
    groups = constructgroups(m,k)
    for i in range(k):
        testidx = groups[i]
        trainidx = []
        for j in range(k):
            if j != i: trainidx += groups[j]
        f = algorithm(Sx=xs[trainidx], Sy=ys[trainidx], T=T, requirement=.5)
        trainerr += evalerror(f, xs[trainidx], ys[trainidx])
        testerr += evalerror(f, xs[testidx], ys[testidx])
    return trainerr/k, testerr/k

def crossvalidateT(xs, ys, k, Ts):
    for T in Ts:
        ltrainavg, ltestavg = crossvalidate(xs=xs, ys=ys, k=k, algorithm=logisticloss, T=T)
        lstring = " ".join([str(i) for i in [T, ltrainavg, ltestavg, "\n"]])
        print(lstring)
        with open("errorlogistic.txt", "a") as f: f.write(lstring)
        atrainavg, atestavg = crossvalidate(xs=xs, ys=ys, k=k, algorithm=adaboost, T=T)
        astring = " ".join([str(i) for i in [T, atrainavg, atestavg, "\n"]])        
        with open("erroradaboost.txt", "a") as f: f.write(astring)
        print(astring)

np.random.seed(1)

Ts = [10, 100, 1000, 10000, 100000]
xs, ys = importdata("abalone.data")
crossvalidateT(xs, ys, 10, Ts)

# Toy example
#Sx = np.matrix([.23, .51, .01, .42, .63, .15, .91, .37]).transpose()
#Sy = np.matrix([-1, 1, 1, 1, -1, -1, 1, -1]).transpose()
#D = np.matrix([.11,.09,.109,.091,.105,.095, .19, .21]).transpose()
#stump(Sx, Sy,D)


