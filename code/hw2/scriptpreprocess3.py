import random

def importdata(filename):
    xs, ys = [], []
    with open(filename) as f:
        for line in f:
            row = line.replace("\n", "").split(",")
            y = int(int(row[-1])<=9) # classification
            # handle numberic variables
            x = {1:0, 2:0, 3:0}
            for i in range(1,len(row)-1):
                x[i+3] = float(row[i])
            # handle categorical variable
            if row[0] == "M": x[1] = 1
            elif row[0] == "F": x[2] = 1
            elif row[0] == "I": x[3] = 1
            else: raise ValueError("Unknown type")
            # save x and y
            xs += [x]
            ys += [y]
    return xs, ys

def exportdata(xs, ys, filename):
    ykey = ["-1 ", "+1 "]
    with open(filename, "w") as f:
        for i in range(len(xs)):
            x, y = xs[i], ys[i]
            strx = str(x)[1:-1].replace(",", "").replace(": ", ":")
            f.write(ykey[y] + strx + "\n")

if __name__ == '__main__':
    xs, ys = importdata("abalone.data")
    exportdata(xs[:3133], ys[:3133], "3.train")
    exportdata(xs[3133:], ys[3133:], "3.test")
