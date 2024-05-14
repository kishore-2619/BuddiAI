# Program to generate a random number between 0 and 1
num=["abcd", "efgh", "ijkl", "mnop", "qrst", "uvwxyz"]
output=[1,2,4,6,11,3]
def drawSample(pmf:dict[str,int])->list[str]:
    num=list(pmf.keys())
    output=list(pmf.values())
summ = sum(output)
outputprob = []
cmf = []
for _ in output:
    outputprob.append(_/summ)
c=0
for i in outputprob:
    c+=i
    cmf.append(c)
print(output, outputprob, cmf)
# importing the random module
samples = 6
import random
for ko in range(samples):
    n=random.random()
    print(n)
    for i in cmf:
        if i>=n:
            print(num[cmf.index(i)])
            break
pmf={"ABCD":1,"EFGH":2,"IJKL":4,"MNOP":6,"QRST":11,"UVWXYZ":3}
num=list(pmf.keys())
output=list(pmf.values())
