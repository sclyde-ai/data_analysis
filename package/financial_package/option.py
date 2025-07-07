import numpy as np

class binomial_model:
    def __init__(self, term: int, up: float, down:float, interest_rate:float, S_0: float, A_0: float, error: int=12):
        if S_0 < 0 or up < 0 or down < 0:
            raise ValueError("no way to have neagtive stock price!")
        if up <= down:
            raise ValueError("up factor must be greater than down factor")
        if term < 0:
            raise ValueError("you cant travel into the past even if you regret something...")
        if interest_rate < 0:
            raise ValueError("theere is not negative interest rate except Japan")
        self.S_0 = S_0
        self.A_0 = A_0
        self.u = up
        self.d = down
        self.r = interest_rate
        self.term = term
        self.S = self.stock()
        self.A = self.safety_asset()
        self.error = error
        self.p, self.q = self.risk_neutral_prbability()

    def stock(self):
        S = np.empty((self.term+1, self.term+1))
        S.fill(np.nan)
        S[0, 0] = self.S_0
        for i in range(self.term):
            S[0, i+1] = S[0, i] * self.u
        for i in range(self.term+1):
            for j in range(self.term-i):
                S[j+1, i] = S[j, i] * self.d
        return S
    
    def safety_asset(self):
        A = [round(self.A_0 * ((1+self.r)**i)) for i in range(self.term+1)]
        return A
    
    def risk_neutral_prbability(self):
        p = round(((1+self.r)-self.d)/(self.u-self.d), self.error)
        q = round((self.u -(1+self.r))/(self.u-self.d), self.error)
        return p, q

    def call_european(self, strike_price):
        K_matrix = np.fliplr(np.eye(self.term+1))*strike_price
        V = self.S - K_matrix
        V = np.where(V < 0, 0, V)
        for i in range(1, self.term+1):
            for j in range(self.term+1-i):
                # print(self.stock[j, self.term-i-j])
                # print(V[j, self.term-i-j+1], V[j+1, self.term-i-j])
                V[j, self.term-i-j] = self.p*V[j, self.term-i-j+1] + self.q*V[j+1, self.term-i-j]
        return V