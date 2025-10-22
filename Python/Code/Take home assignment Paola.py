#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 13:03:40 2025

@author: PaolaPavon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# QUESTION 1 -----------------------------------------------------------------
# a) Reading and loading the data --------------------------------------------
# Creating the data frame
df = pd.read_csv("../Data/Gold.csv", dtype = {0:str}, index_col=0)

df.index = pd.to_datetime(df.index, dayfirst=True)

print(df.index.dtype) # datetime64[ns]

# Number of rows and columns
rr, cc = df.shape # (1258, 1)

# Price Lag column
df['Price Lag'] = df['Price'].shift(1)
"""
Date
2020-01-02            NaN
2020-01-03    1524.500000
2020-01-06    1549.199951
2020-01-07    1566.199951
"""

# Price Change column
df['Price Change'] = df['Price'] - df['Price Lag']
"""
Date
2020-01-02          NaN
2020-01-03    24.699951
2020-01-06    17.000000
2020-01-07     5.600098
"""

# Gain column
df['Gain'] = 0 * rr

for i in range(rr):
    if df.iloc[i, 2] > 0:
        df.iloc[i, 3] = df.iloc[i, 2]
"""
Date
2020-01-02     0.000000
2020-01-03    24.699951
2020-01-06    17.000000
"""
        
# Loss column
df['Loss'] = 0 * rr

for i in range(rr):
    if df.iloc[i, 2] < 0:
        df.iloc[i, 4] = -df.iloc[i, 2]   
"""
Date
2020-01-02     0.000000
2020-01-03     0.000000
2020-01-06     0.000000
"""

# b) Calculating average gains and losses ------------------------------------
# Average Gain
df['Average Gain'] = df['Gain'].rolling(window = 14).sum() / 14
"""
2024-12-27    11.299997
2024-12-30     9.421439
2024-12-31     8.735700
"""

# Average Loss
df['Average Loss'] = df['Loss'].rolling(window = 14).sum() / 14
"""
2024-12-27    12.828578
2024-12-30    13.621425
2024-12-31    13.621425
"""

# c) Calculating RS and RSI --------------------------------------------------
df['RS'] = df['Average Gain'] / df['Average Loss']
"""
2024-12-27    0.880846
2024-12-30    0.691663
2024-12-31    0.641321
"""

df['RSI'] = 100 - (100 / (1 + df['RS']))
"""
2024-12-27    46.832424
2024-12-30    40.886580
2024-12-31    39.073450
"""

# d) 200 day Moving Average --------------------------------------------------
df['MA200'] = df['Price'].rolling(window = 200).sum() / 200

df = df.dropna()

# First 3 rows
print(df.iloc[0:3,:])
"""
                  Price    Price Lag  ...        RSI        MA200
Date                                  ...                        
2020-10-15  1903.199951  1901.300049  ...  61.301524  1747.541999
2020-10-16  1900.800049  1903.199951  ...  57.423114  1749.423499
2020-10-19  1906.400024  1900.800049  ...  53.503176  1751.209500
"""

# Last 3 rows
print(df.iloc[-3:,:])
"""
                  Price    Price Lag  ...        RSI        MA200
Date                                  ...                        
2024-12-27  2617.199951  2638.800049  ...  46.832424  2473.048995
2024-12-30  2606.100098  2617.199951  ...  40.886580  2475.264496
2024-12-31  2629.199951  2606.100098  ...  39.073450  2477.623995
"""

# QUESTION 2 -----------------------------------------------------------------
df = pd.read_csv("../Data/industryPortfolios_1970.csv", dtype = {0:str}, 
                      index_col=0)

nr, nc = df.shape # (624, 49)

# 1) Computing mean-variance frontier ----------------------------------------
# This is the data we have
mu_r     = df.mean(axis = 0) 
sigmas_r = df.std(axis = 0)
vcov_r   = df.cov()

# We want to obtain A, B, C and D
def MV(ones, vcov_r, mu_r):
    temp = np.dot(ones, np.linalg.inv(vcov_r))
    A = np.dot(temp, ones) 
    
    temp = np.dot(mu_r, np.linalg.inv(vcov_r))
    B = np.dot(temp, ones)
    
    temp = np.dot(mu_r, np.linalg.inv(vcov_r))
    C = np.dot(temp, mu_r)

    D = A * C - B ** 2
    
    return A, B, C, D

ones = np.ones(nc) 
A, B, C, D = MV(ones, vcov_r, mu_r)
"""
(np.float64(0.10389296131529883),
 np.float64(0.09629206848951471),
 np.float64(0.1808293908721528),
 np.float64(0.00951473845656023))
"""

# Create our target expected return vector
min_mu = 0.25 # It's already in %
max_mu = 2
n_mu = 100
gmu = np.linspace(min_mu, max_mu, n_mu)

# Now we compute the corresponding variance
sigma_mvf = ((A * gmu ** 2 - 2 * gmu * B + C) / D) ** 0.5
# ([3.82458914, 3.79072726, 3.75746829, 3.72482839, 3.69282397,...

# Let's plot it!!!
plt.figure(figsize=(10,8))
plt.plot(sigma_mvf, gmu, 'o')
plt.xlabel('Volatility')
plt.ylabel('Mean')
plt.title('Mean-Variance Frontier')
plt.plot(sigmas_r, mu_r, 'ro') # red circles correspont to every IndPort
plt.show()

# 2) Computing the optimal portfolio weights ---------------------------------
expected_mu = 1

aux1 = np.dot(np.linalg.inv(vcov_r),ones) # ([ 0.01126377,  0.0067746 , ...
aux2 = np.dot(np.linalg.inv(vcov_r),mu_r) # ([ 0.00370184,  0.00865536, ...
g1 = (C - expected_mu * B) / D # 8.884881362592916
g2 = (expected_mu * A - B) / D # 0.7988546254304498

weights = np.outer(g1, aux1) + np.outer(g2, aux2)
# [ 0.10303448,  0.06710593, -0.02117617,  0.03159728,  0.00564961, ...

# 3) No short possitions -----------------------------------------------------
# First we remove those assets with a negative weight
bool_vector = weights[0] > 0

df2 = df.iloc[:,bool_vector]
"""
     Agric   Food  Beer  Smoke  Books  ...  Trans  Whlsl  Rtail  RlEst    Fin
Date                                      ...                                   
197001   0.83  -2.81 -1.35  -6.99 -11.39  ...  -7.62  -7.68  -5.73 -10.94 -10.82
197002   9.48   5.96  6.87   0.28   0.68  ...  10.48   1.81   5.79   0.24   9.18
197003 -13.28  -0.61 -0.60   1.42  -2.89  ...  -3.34  -5.28  -0.98  -0.95  -0.59
"""

# We repeat everything again...
nr, nc = df2.shape # (624, 29) we dropped 20 columns

mu_r2     = df2.mean(axis = 0)
sigmas_r2 = df2.std(axis = 0)
vcov_r2   = df2.cov()

ones2 = np.ones(nc)
A, B, C, D = MV(ones2, vcov_r2, mu_r2)
"""
(np.float64(0.0889546923570895),
 np.float64(0.08300724182889813),
 np.float64(0.11908336228747562),
 np.float64(0.0037028216610890535))
"""

min_mu = 0.25
max_mu = 2
n_mu = 100
gmu = np.linspace(min_mu, max_mu, n_mu)
sigma_mvf = ((A * gmu ** 2 - 2 * gmu * B + C) / D) ** 0.5
# ([4.73845838, 4.6776375 , 4.61764123, 4.55850213, 4.50025398, ...

plt.figure(figsize=(10,8))
plt.plot(sigma_mvf, gmu, 'o')
plt.xlabel('Volatility')
plt.ylabel('Mean')
plt.title('Mean-Variance Frontier')
plt.plot(sigmas_r2, mu_r2, 'ro') # red circles correspont to every IndPort
plt.show()

# And NOW we compute the new weights for these 29 industries:
aux1 = np.dot(np.linalg.inv(vcov_r2),ones2) # ([ 0.008659  ,  0.00703777, ...
aux2 = np.dot(np.linalg.inv(vcov_r2),mu_r2) # ([ 0.0032549 ,  0.01004055, ...
g1 = (C - expected_mu * B) / D # 9.742872803646446
g2 = (expected_mu * A - B) / D # 1.606194160169772

weights2 = np.outer(g1, aux1) + np.outer(g2, aux2)
# [[ 8.95915500e-02,  8.46951604e-02, -2.36369517e-04,1.97299015e-02, ...

"""
Of course this still gives us negative positions, but now on a portfolio
of 29 insdustries instead of one of 49. If we really wanted for the portfolio
to not have negative positions, then we would need to include that as a 
constrain, not just eliminating the industries.

print(weights2)
[[ 8.95915500e-02  8.46951604e-02 -2.36369517e-04  1.97299015e-02
  -7.63915423e-02  1.55829425e-01  6.66483940e-04  4.76575452e-02
   4.48095163e-02  5.01958346e-02  2.90660692e-03  2.25295793e-02
  -1.04707448e-01  1.10006503e-02  6.61319962e-02  4.46790434e-02
   2.67952884e-03  5.97409530e-02  3.87093022e-01  2.46219943e-01
   7.99614336e-02 -5.56090660e-02  5.22530014e-03  2.79434926e-02
  -5.34099114e-03 -1.17638378e-01  1.05868373e-01 -1.01714465e-01
  -9.35170800e-02]]
"""

# QUESTION 3 -----------------------------------------------------------------
"""
# 1) -------------------------------------------------------------------------
Removing the asset with the lowest expected return is definitely not a good 
suggestion, for an asset may not have a great return but it might contribute
to diversifying the portfolio and, therefore, decreasing total volatility.

The idea of removing assets by their Sharpe Ratio makes a bit more sense,
since this at least accounts for the asset's volatility. However, this approach
WOULD NOT account for how the asset relates to the rest of the portfolio 
(correlation).

A better idea would be to do something like what we did in one class, I think
it was lecture 6, where we removed the asset that contributed less to the 
mean-variance frontier. That is, the asset that, once removed, gave the 
smallest reduction in the maximum Sharpe ratio. This approach focuses on the 
portfolio AS A WHOLE instead of on individual assets.

# 2) -------------------------------------------------------------------------
We also saw this in class. The thing is, we always want our portfolio weights
to sum up to 100%. This means, if weights are allowed to be between -100% and 
+200%, we can FULLY allocate our portfolio in either value or growth by 
shorting and leveraging in the other. While being able to short initially 
allows us to reduce volatility, if we do these in extreme we take unbalanced 
positions — for example, going 100% short in growth and 100% long in value.

These highly leveraged portfolios allow us to lower the target expected return,
but by doing so we're sacrificing diversification. As a result, volatility
reaches its minimum and begins to increase again, creating the U-shape of 
the mean-variance frontier.
"""

# QUESTION 4 -----------------------------------------------------------------
data = pd.read_csv("../Data/portfolios.csv", dtype = {0:str}, index_col=0)
BM = pd.read_csv("../Data/bookToMarket.csv", dtype = {0:str}, index_col=0)
ME = pd.read_csv("../Data/marketEquity.csv", dtype = {0:str}, index_col=0)
assets = data.iloc[:,0:100].copy()

# 1) Sorting the assets ------------------------------------------------------
rr, cc = assets.shape # (696, 100)
assets_sorted = pd.DataFrame(np.nan, index = assets.index, 
                             columns = ['N/A'] * cc)
me_sorted = pd.DataFrame(np.nan, index = assets.index, 
                         columns = ['N/A'] * cc)

for i in range(rr):
    inds = BM.iloc[i, :].argsort() # Indeces based on BM
    assets_sorted.iloc[i, :] = assets.iloc[i, inds] # Sorting returns based 
                                                    # on BM
    me_sorted.iloc[i,:] = ME.iloc[i,inds] # Sorted ME based on BM
    
n = 4 # number of portfolios
n1 = round(cc/2)  # 50 (first split 100 portfolios into 2x50)
n2 = round(cc/(2*2)) # 25 (each 50 assets split into another 2 portfolios, 4*25)
portfolios = pd.DataFrame(np.nan, 
                          index = assets.index, 
                          columns = range(1,n+1)) # 696 x 4 i.e. 4 portfolios

for i in range(rr):
    col = 0
    for j in range(0,cc,n1): # [0, 50]
        pSlice = assets_sorted.iloc[i, j:j+n1].copy()
        ind = me_sorted.iloc[i, j:j+n1].argsort()
        pSlice_sorted = pSlice.iloc[ind]
        for k in range(0,n1-1,n2): # [0, 25]
            portfolios.iloc[i, col] = pSlice_sorted[k:k+n2].mean()
            col += 1      

print(portfolios)
"""
               1         2         3         4
Date                                          
196401  2.251584  1.478916  3.143624  2.099152
196402  1.903408  2.250828  1.235388  2.227740
196403  2.084608  2.057504  3.329688  3.853676
"""
            
# 2) Creating value and size factors -----------------------------------------
val_top = portfolios.iloc[:,-2:] # Portfolios 3 and 4
val_bottom = portfolios.iloc[:,0:2] # Portfolios 1 and 2

r_value = (val_top.sum(axis = 1) - val_bottom.sum(axis = 1))/ 2
r_value.mean() # 0.07787162068965514

size_top = portfolios.iloc[:,[1,3]] # Portfolios 2 and 4
size_bottom = portfolios.iloc[:,[0,2]] # Portfolios 1 and 3

r_size = (size_top.sum(axis=1) - size_bottom.sum(axis=1))/ 2
r_size.mean() # -0.1507970057471264

# 3) Alphas, betas and p-values ----------------------------------------------
res = pd.DataFrame(np.nan, index = assets.columns, 
                   columns = ['alpha', 'betaValue', 'betaSize', 'p-val Value', 
                              'p-val Size'])

rf = data['Rf']

x = pd.DataFrame(r_value, columns = ['Value'], index = r_value.index)
x['Size'] = r_size
X = sm.add_constant(x) 

for j in range(cc): 
    y = assets.iloc[:,j] - rf
    model= sm.OLS(y, X, missing='drop')
    results = model.fit()
    res.iloc[j, 0:3] = results.params
    res.iloc[j, 3:] = results.pvalues[1:3]
    
print(res)
"""
           alpha  betaValue  betaSize   p-val Value    p-val Size
Asset1   -0.027240   0.254115 -2.334225  1.851026e-01  1.048537e-66
Asset2    0.400546   0.063992 -1.925721  6.397364e-01  1.621898e-83
Asset3    0.501555   0.161178 -1.977462  2.325090e-01  1.271121e-88
"""
    
# 4) Monte Carlo -------------------------------------------------------------
mu_value = r_value.mean()
mu_size = r_size.mean()
cov = pd.concat([r_value, r_size], axis = 1).cov()

weights = np.array([0.7, 0.3])
mu = np.dot(weights, [mu_value, mu_size]) # 0.00927103275862068
sigma = np.sqrt(np.dot(weights,np.dot(cov, weights))) # 0.8944460314943015

np.random.seed(1234)
nsim = 10000

simulations = np.random.normal(mu, sigma, nsim)
sum(simulations < -5) # 0

simulations_quart = np.random.normal(mu * 3, sigma * np.sqrt(3), nsim)
sum(simulations_quart < 0) # 4964

