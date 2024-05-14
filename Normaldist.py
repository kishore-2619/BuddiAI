import numpy as np
import matplotlib.pyplot as plt


def normal(x, m, s):
    y = 1/(s*np.sqrt(2*np.pi))*np.exp(-((x-m)**2)/(2*s**2))
    return y

m = 1
s1, s2, s3 = 2, 4, 6
X1 = np.arange(m - 5 * s1, m + 5 * s1, 0.01)
X2 = np.arange(m - 5 * s2, m + 5 * s2, 0.01)
X3 = np.arange(m - 5 * s3, m + 5 * s3, 0.01)
Y1 = normal(X1, m, s1)
Y2 = normal(X2, m, s2)
Y3 = normal(X3, m, s3)

plt.figure(figsize=(10, 5))
plt.plot(X1, Y1, label=f"μ{m} σ {s1}")
plt.plot(X2, Y2, label=f"μ{m} σ {s2}")
plt.plot(X3, Y3, label=f"μ{m} σ {s3}")
plt.legend()
txt1 = "The Normal Distribution Curve with Fixed mean and different Standard Deviation"
plt.title("Normal Distribution Curve")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.figtext(
    0.5,
    0.01,
    txt1,
    wrap=True,
    horizontalalignment="center",
    fontsize=8,
    bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
)
plt.grid(True)
plt.show()

s = 1
m1, m2, m3 = 2, 4, 6
X1 = np.arange(m1 - 5 * s, m1 + 5 * s, 0.01)
X2 = np.arange(m2 - 3 * s, m2 + 3 * s, 0.01)
X3 = np.arange(m3 - 3 * s, m3 + 3 * s, 0.01)
Y1 = normal(X1, m1, s)
Y2 = normal(X2, m2, s)
Y3 = normal(X3, m3, s)

plt.figure(figsize=(10, 5))
plt.plot(X1, Y1, label=f"μ {m1} σ {s}")
plt.plot(X2, Y2, label=f"μ {m2} σ {s}")
plt.plot(X3, Y3, label=f"μ {m3} σ {s}")
plt.legend()
txt2 = "The Normal Distribution Curve with different mean and fixed Standard Deviation"
plt.title("Normal Distribution Curve")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.figtext(
    0.5,
    0.01,
    txt2,
    wrap=True,
    horizontalalignment="center",
    fontsize=8,
    bbox={"facecolor": "grey", "alpha": 0.3, "pad": 5},
)
plt.grid(True)
plt.show()
