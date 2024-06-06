import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define constants
ro = 0.05
hf = np.pi / 90
M = 180
coef = (M / 2) * ro ** 2

# Define the grid function
def grid(rm):
    res = rm
    num = 0
    hr = ro
    Hr = [0, ro]
    hro = [ro]
    n = 1
    while res > 0:
        num += 1
        hr = coef / (0.5 + Hr[num])
        Hr.append(Hr[num] + hr)
        hro.append(hr)
        res -= hr
        n += 1
    Gr = np.zeros((n, M, 2), dtype=np.float64)
    C0 = np.zeros((n, M), dtype=np.float64)
    C1 = np.zeros((n, M), dtype=np.float64)
    for i in range(n):
        for j in range(M):
            Gr[i, j] = np.array([Hr[i], j * hf], dtype=np.float64)
            C0[i, j] = np.exp(-(Gr[i, j, 0] - 2))
    Hf = Hr[:-1]
    return Gr, C0, C1, Hf, hro

# Initialize grid and concentrations
st, C0, C1, HR, hr = grid(5)

# Define integrals
def Int1(C, gr):
    Sum = C[0, 0] * np.pi * gr[1][0][0] ** 2
    for i in range(1, len(C) - 1):
        for j in range(M):
            Sum += 0.5 * C[i, j] * hf * (gr[i + 1, j, 0] ** 2 - gr[i, j, 0] ** 2)
    return Sum

# Normalize C0
test1 = Int1(C0, st)
C0 = 5 * C0 / test1
test1 = Int1(C0, st)

# Define delta function
def delta_f(x, y):
    if (x == 0) and (y == 0):
        return 0
    elif (x == 0):
        return np.pi / 2 if y > 0 else 3 * np.pi / 2
    elif (x > 0):
        return np.arctan(y / x) if y >= 0 else 2 * np.pi + np.arctan(y / x)
    else:
        return np.pi + np.arctan(y / x)

# Define function to subtract grids
def gr_pl_sub(gr1, gr2):
    dx = gr1[0] * np.cos(gr1[1]) - gr2[0] * np.cos(gr2[1])
    dy = gr1[0] * np.sin(gr1[1]) - gr2[0] * np.sin(gr2[1])
    dr = np.sqrt(dx ** 2 + dy ** 2)
    df = delta_f(dx, dy)
    HH = abs(HR - dr * np.ones(len(HR), dtype=np.float64))
    m = np.where(HH == min(HH))[0][0]
    l = round(df / hf)
    return m, 0 if l == M else l

# Define the second integral
def Int2(C, gr, r, fi):
    Sum = 0
    for i in range(len(C) - 1):
        for j in range(M):
            m, l = gr_pl_sub(gr[r, fi], gr[i, j])
            if i == 0:
                Sum += C[m, l] * np.pi * gr[1][0][0] ** 2
            else:
                Sum += 0.5 * C[m, l] * hf * (gr[i + 1, j, 0] ** 2 - gr[i, j, 0] ** 2)
    return Sum

# Define the third integral
def Int3(C, gr, r, fi):
    Sum = 0
    for i in range(len(C) - 1):
        for j in range(M):
            m, l = gr_pl_sub(gr[r, fi], gr[i, j])
            if i == 0:
                Sum += C[m, l] * C[0, 0] * np.pi * gr[1][0][0] ** 2
            else:
                Sum += 0.5 * C[i, j] * C[m, l] * hf * (gr[i + 1, j, 0] ** 2 - gr[i, j, 0] ** 2)
    return Sum

# Define parameters
n = 2.0
a = 0.05
b = 0.1
d = 0.02
s = 0.01
rm = st[len(st) - 1, 0, 0]

# Calculate Sdash and Bdash
Sdash = gamma(n / 2 + 1) * s / ((rm ** n) * np.pi ** (n / 2))
Bdash = gamma(n / 2 + 1) * b / ((rm ** n) * np.pi ** (n / 2))

# Initialize steps and errors
steps = 10
EPS_step1 = np.zeros(steps)
EPS_step2 = np.zeros(steps)

# Perform Neumann series for same kernels
q0 = C0.copy()
C1_same_kernels = np.zeros_like(C0)
for k in range(steps):
    N = Sdash * Int1(q0, st) / (b - d)
    eps = 0
    C1 = np.zeros_like(C0)
    for i in range(len(st)):
        N2 = Int2(q0, st, i, 0)
        N3 = Int3(q0, st, i, 0)
        C1[i, 0] = (Bdash * (N + N2) + 0.5 * (a * s * N ** 3 - a * Sdash * N3 / N)) / (
                    (1 - a / 2) * b + 0.5 * a * d + Sdash + 0.5 * a * Sdash * N2 / N)
        eps += np.sqrt(abs(C1[i, 0] ** 2 - q0[i, 0] ** 2))
        C1[i, :] = C1[i, 0]
    q0 = C1
    EPS_step1[k] = eps
    C1_same_kernels = C1

# Perform Neumann series for different kernels
q0 = C0.copy()
C1_diff_kernels = np.zeros_like(C0)
for k in range(steps):
    N = Sdash * Int1(q0, st) / (b - d)
    eps = 0
    C1 = np.zeros_like(C0)
    for i in range(len(st)):
        N2 = Int2(q0, st, i, 0)
        N3 = Int3(q0, st, i, 0)
        C1[i, 0] = (Bdash * (N + N2) + 0.5 * (a * s * N ** 3 - a * Sdash * N3 / N)) / (
                    (1 - a / 2) * b + 0.5 * a * d + Sdash + 0.5 * a * Sdash * N2 / N)
        eps += np.sqrt(abs(C1[i, 0] ** 2 - q0[i, 0] ** 2))
        C1[i, :] = C1[i, 0]
    q0 = C1
    EPS_step2[k] = eps
    C1_diff_kernels = C1

# Plotting
x_values = np.arange(len(C1_same_kernels))
err_values = np.arange(len(EPS_step1))

plt.figure(figsize=(12, 6))
plt.plot(x_values, C1_same_kernels[:, 0], label="C(x)")
plt.plot(err_values, EPS_step1, label="Error", linestyle='--')
plt.xlabel('x')
plt.ylabel('C(x)')
plt.title('Dependence of C on x')
plt.legend()
plt.grid(True)
plt.show()