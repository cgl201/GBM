# -*- coding: utf-8 -*-
# Time: 4/18/2019 12:41 PM
# Author: Guanlin Chen

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm

# ==== Problem2 ====
# ====import data====

pd.set_option('expand_frame_repr', False)  # do not wrap
pd.set_option('display.max_rows', 1000)
df = pd.read_csv(r'C:\Users\Jason\Desktop\Python Project\Financial engineer\Data\stock_prices.csv')

# ==== hyper-parameters ====
dt = 1 / 252
T = 1 / 12

# # ===== Q1 =======
df['log_return_S1'] = np.log(df['S1'] / df['S1'].shift(1))
df['log_return_S2'] = np.log(df['S2'] / df['S2'].shift(1))

sigma_S1 = df['log_return_S1'].std(ddof=1) / np.sqrt(dt)
sigma_S2 = df['log_return_S2'].std(ddof=1) / np.sqrt(dt)

cov = df[['log_return_S1', 'log_return_S2']].cov().iat[0, 1]
rho = cov / (sigma_S1 * sigma_S2 * dt)

# ===== Q2-1 =======
W1 = np.sqrt(T) * np.random.standard_normal(1000)
W2 = np.sqrt(T) * np.random.standard_normal(1000)
W2 = rho * W1 + np.sqrt(1 - rho ** 2) * W2

S1_init = df.iloc[-1]['S1']  # select the most recent observation as S0
S2_init = df.iloc[-1]['S2']

S1_final = S1_init * np.exp((-0.5 * sigma_S1 ** 2) * T + sigma_S1 * W1)
S2_final = S2_init * np.exp((-0.5 * sigma_S2 ** 2) * T + sigma_S2 * W2)

df2 = pd.DataFrame({'S1_final': S1_final, 'S2_final': S2_final})

threshold1 = S1_init * 0.95
threshold2 = S2_init * 0.95

condition1 = df2['S1_final'] <= threshold1
condition2 = df2['S2_final'] <= threshold2

prob = df2[condition1 & condition2].count()[0] / df2.count()[0]  # use frequency as probability

# ===== Q2-2 =======

portfolio_init = 1 * S1_init + 10 * S2_init
df2['portfolio'] = df2['S1_final'] + 10 * df2['S2_final']
df2['loss'] = df2['portfolio'] - portfolio_init
VaR = -df2['loss'].quantile(0.05)  # VaR is positive

# ==== Q3 =====
"""
Yes, for the analytical solution of Q2-1, we have the joint distribution, so Prob = F(S1<log(S1*0.95/S), S2<log(S2*0.95<S2)).

Likewise, we have the covariance matrix of log return and the weight of portfolio, so we can calculate analytical VaR.
"""
mu = -0.5 * T * np.array([sigma_S1 ** 2, sigma_S2 ** 2])
cov_matrix = T * np.array([sigma_S1 ** 2, rho * sigma_S1 * sigma_S2, rho * sigma_S1 * sigma_S2, sigma_S2 ** 2]).reshape(2, 2)
Prob_analytical = multivariate_normal.cdf([np.log(0.95), np.log(0.95)], mu, cov_matrix)  # log(0.95 * S/ S) = log(0.95)

S1_frac = S1_init / portfolio_init
S2_frac = 10 * S2_init / portfolio_init

frac_vec = np.array([S1_frac, S2_frac])

x = np.dot(cov_matrix, frac_vec.T)
x = np.dot(frac_vec, x)
log_ret_VaR = np.sqrt(x) * norm.ppf(0.05)

VaR_analytical = -(np.exp(log_ret_VaR) * portfolio_init - portfolio_init)  # restore from return_VaR to price_VaR

# ==== summarize solutions =====
solution_list = [sigma_S1, sigma_S2, rho, prob, VaR, Prob_analytical, VaR_analytical]
arrays = [np.array(['Q1', 'Q1', 'Q1', 'Q2', 'Q2', 'Q3', 'Q3']),
          np.array(['Sigma1', 'Sigma2', 'Rho', 'Prob', 'VaR', 'Prob_analytical', 'VaR_analytical'])]

Solutions = pd.Series(solution_list, index=arrays)

print(Solutions)








