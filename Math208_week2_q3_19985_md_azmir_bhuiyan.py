import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data
data = {'X': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'Y': [30, 25, 95, 115, 265, 325, 570, 700, 1085, 1300]}
df = pd.DataFrame(data)

# Linear regression
X = df['X'].values.reshape(-1,1)
Y = df['Y'].values.reshape(-1,1)
A = np.hstack([X, np.ones_like(X)])
b, a = np.linalg.lstsq(A, Y, rcond=None)[0]
r = np.corrcoef(df['X'], df['Y'])[0, 1]

print(f"b1 (slope): {b[0]}")
print(f"b0 (intercept): {a[0]}")
print(f"Coefficient of linear correlation (r): {r}")

# Plotting the data
plt.figure(figsize=(10, 6))
plt.scatter(df['X'], df['Y'], color='red', label='Data points')

# Plotting the linear fit
plt.plot(df['X'], a + b * df['X'], label=f'Linear Fit; y = {a[0]:.2f} + {b[0]:.2f}x, r = {r:.2f}')

# Try polynomial fit
p = np.polyfit(df['X'], df['Y'], 2) # change 2 to higher order for higher polynomial fit
poly = np.poly1d(p)
new_x = np.linspace(df['X'].min(), df['X'].max(), 100)
new_y = poly(new_x)

plt.plot(new_x, new_y, label='Polynomial Fit', color='green')

plt.title('Data points and Fitting Models')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
