import sys
from GetDataFromList import GetXy
with open(sys.argv[1]) as fIn:
    lLines = fIn.read().splitlines()
lFeatMask = [tuple(sLine.split()) for sLine in lLines]
a,b,c = GetXy(lFeatMask[:300],int(sys.argv[2]),9,78,True)
print(c)

