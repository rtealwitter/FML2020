import sys

def readcount(from_file):
    on_margin, off_margin = 0, 0
    with open(from_file, "r") as f:
        lines = f.readlines()
        total = lines[6][9:].replace("\n", "")
        for line in lines[11:]:
            if line[:4] == "128 " or line[:5] == "-128 ":
                on_margin += 1
            else: off_margin += 1
    return total, on_margin

def writecount(total, on_margin, to_file):
    with open(to_file, "a") as f:
        f.write(total + " " + str(on_margin) + "\n")

total, on_margin = readcount(sys.argv[1])
writecount(total, on_margin, sys.argv[2])
