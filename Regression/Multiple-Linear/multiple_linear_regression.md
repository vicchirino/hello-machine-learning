### Simple Linear Regression

`y = b0 + b1*x1 + b2*x2 + ... + bn*nx`

y: Dependent variable (DV)
xn: Independent variables (IVs)
b0: Coefficient
bn: Constant

No need to run feature scaling

## Dummy Variables
- Numericals variables
- Categorically variables --> Dummy variables

Create new variables per each categorically variable but never includes all dummy variables on the model.
Ommit one dummy variable. 

## Statistical Significance

- Drop a coin:
  
H0: this is a fair coin
H1: this is not a fair coin

tails 0.5     | `P-Value`
tails 0.25    |
tails 0.12    |
tails 0.06    |
---------------------- α = 0.05
tails 0.03    |
tails 0.015   |
              ˇ

## Building A Model

- Only keep the important variables.
- Methods:
  - 1. All-in
  - 2. Backward Elimination *  
  - 3. Forward Selection *
  - 4. Bidirectional Elimination *
  - 5. Score Comparasion

* Stepwise Regression

# All-in

- Prior knowledge; OR
- You have to; OR
- Preparing for Backward Elimination

# Backward Elimination

1. Select a significate level to stay in the model (e.g. SL = 0.05).
2. Fit the full model with all possible predictors.
3. Consider the predictor with the *highest* P-Value. If P > SL, go to Step 4, othwersie go to FIN.
4. Remove the predicor.
5. Fit the model without this variable *.
6. Return to Step 3.

# Forward Selection

1. Select a significate level to enter the model (e.g. SL = 0.05).
2. Fit all sample regression models *y ~ xn* Select the one with the lowest P-Value.
3. Keep this variable and fit all possible models with one extra predicator added to the one(s) you already have.
4. Consider the prefictor with the *lowest* P-Value. If P < SL, go to Step 3, otherwise go to FIN.

# Bidirectional Elimination

1. Select a significance level to enter and to stay in the model (e.g. SLENTER = 0.05, SLSTAY = 0.05).
2. Perform the next step of Forward Selection (new variables must have P < SLENTER to enter).
3. Perform ALL steps of Backwoard Elimination (old variables muyst have P < SLSTAY to stay).
4. No new variables can enter and no old variables can exit.

# All Possible Models

1. Select a criterion of goodness of fit (e.g. Akaike criterion).
2. Constrict All possible Regression Models `(2^n) - 1` total combinations.
3. Select the one with the best criterion.