import numpy as np
import matplotlib.pyplot as plt
txt="The plot that shows the Lagrange's polynomial of the given set of input and output"

def lagrange(X_input, Y_output, xi, n):
    res=0.0
    for i in range(n):
        t=Y_output[i]
        for j in range(n):
            if(j!=i):
                t=t*(xi-X_input[j])/(X_input[i]-X_input[j])
        res+=t
    return res

X_input=[-3,-2,-1,0,1,2,3]
Y_output=[7,2,0,0,0,2,7]
n=len(X_input)
Y_output_cap=[lagrange(X_input,Y_output,X_input[i],n) for i in range(n)]
print(Y_output_cap)

plt.title("Lagrange's Polynomial Curve ")
plt.scatter(X_input,Y_output, marker="*", c="red",label="Original data points")
plt.plot(X_input,Y_output_cap, color="green", label="Lagrange's polynomial ")
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=6)
plt.legend()
plt.show()
