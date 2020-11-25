import re
import sys
from_file = sys.argv[1]
to_file = sys.argv[2]
d = sys.argv[3]
C = sys.argv[4]

with open(from_file, "r") as f:
    line = f.readlines()[-1]
    accuracy = re.search('= (.+?)%', line).group(1)

with open(to_file, "a") as f:
    f.write(accuracy + " " + d + " " + C + "\n")
