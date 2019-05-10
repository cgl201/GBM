# Parameters from Problem 2
S1_0 = 100
S2_0 = 105
sig1 = 0.15
sig2 = 0.2
rho = -0.5  # new correlation
TT = 2
r = 0.04

nsim = 2000

# correlated Brownian motions
W1_T = sqrt(TT)*rnorm(nsim)
W2_T = rho*W1_T + sqrt(1-rho^2)*sqrt(TT)*rnorm(nsim)

# corresponding terminal asset prices
S1_T = S1_0 * exp((r-0.5*sig1^2)*TT + sig1*W1_T)
S2_T = S2_0 * exp((r-0.5*sig2^2)*TT + sig2*W2_T)

# 1. stock price scatter plot
plot(S1_T, S2_T, pch=18, cex=0.5)

# 2. copula scatter plot
FW1 = pnorm(W1_T, sd=sqrt(TT))
FW2 = pnorm(W2_T, sd=sqrt(TT))
plot(FW1, FW2, pch=18, cex=0.5)

# 3. new copula
# for FW1, swap the smallest 10% with the largest 10%
idx_FW1_smallest = order(FW1)[1:200]
idx_FW1_largest = order(FW1, decreasing=T)[1:200]
temp = FW1[idx_FW1_smallest]
FW1[idx_FW1_smallest] = FW1[idx_FW1_largest]
FW1[idx_FW1_largest] = temp

# scatter plot of new copula
plot(FW1, FW2, pch=18, cex=0.5)

# 4. new Brownian motions
W1r_T = qnorm(FW1, sd=sqrt(TT))
W2r_T = qnorm(FW2, sd=sqrt(TT))
plot(W1r_T, W2r_T, pch=18, cex=0.5)

# 5. new terminal asset prices
S1r_T = S1_0*exp((r-0.5*sig1^2)*TT + sig1*W1r_T)
S2r_T = S2_0*exp((r-0.5*sig2^2)*TT + sig2*W2r_T)
plot(S1r_T, S2r_T, pch=18, cex=0.5, main="Adjusted")

# 6. exchange option price
exp(-r*TT)*mean(pmax(S1_T-S2_T,0))   # before
exp(-r*TT)*mean(pmax(S1r_T-S2r_T,0)) # after