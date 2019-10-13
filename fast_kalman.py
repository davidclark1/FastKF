"""
David Clark
2019
"""

import numpy as np
from sklearn.decomposition import FactorAnalysis

def sym(X):
    return .5 * (X + X.T)

def M_step_batched(y_batches, x_hat_batches, P_batches, P_adj_batches):
    num_batches = len(y_batches)
    y_all = np.concatenate(y_batches)
    x_hat_all = np.concatenate(x_hat_batches)
    x_hat_firsts = np.array([x_hat[0] for x_hat in x_hat_batches])
    P_firsts = np.array([P[0] for P in P_batches])
    P_all = np.concatenate(P_batches)
    P_adj_sum_nofirsts = np.concatenate([P_adj[1:] for P_adj in P_adj_batches]).sum(axis=0)
    P_sum_nolasts = np.concatenate([P[:-1] for P in P_batches]).sum(axis=0)
    P_sum_nofirsts = np.concatenate([P[1:] for P in P_batches]).sum(axis=0)
    T = len(y_all)

    C = (y_all.T.dot(x_hat_all)).dot(np.linalg.inv(P_all.sum(axis=0)))
    R = (1. / T) * (y_all.T.dot(y_all) - C.dot(x_hat_all.T).dot(y_all))
    A = P_adj_sum_nofirsts.dot(np.linalg.inv(P_sum_nolasts))
    Q = (1. / (T - num_batches)) * (P_sum_nofirsts - A.dot(P_adj_sum_nofirsts.T))
    pi_1 = x_hat_firsts.mean(axis=0)
    V_1 = P_firsts.mean(axis=0) - np.outer(pi_1, pi_1)

    Q, R = sym(Q), sym(R)
    return A, Q, C, R, pi_1, V_1
    
def E_step(y, A, Q, C, R, pi_1, V_1, ss_eps=1e-8):
    T = len(y)
    d, n = C.shape #d = observation dim, n = latent dim
    
    #=====Forward pass=====
    #initialize storage variables
    x_m_fwd = np.zeros((T, n))
    x_fwd = np.zeros((T, n))
    V_m_fwd = np.zeros((T, n, n))
    V_fwd = np.zeros((T, n, n))
    K = np.zeros((T, n, d))
    #run forward pass (t=0,...,T-1)
    in_ss_fwd = False
    ss_fwd_idx = -1
    for t in range(T):
        if t == 0:
            x_m_fwd[t] = pi_1
            V_m_fwd[t] = V_1
        else:
            x_m_fwd[t] = A.dot(x_fwd[t-1])
            V_m_fwd[t] = V_m_fwd[t-1] if in_ss_fwd else sym(A.dot(V_fwd[t-1]).dot(A.T) + Q)
        K[t] = K[t-1] if in_ss_fwd else V_m_fwd[t].dot(C.T).dot(np.linalg.inv(C.dot(V_m_fwd[t]).dot(C.T) + R))            
        x_fwd[t] = x_m_fwd[t] + K[t].dot(y[t] - C.dot(x_m_fwd[t]))
        V_fwd[t] = V_fwd[t-1] if in_ss_fwd else sym(V_m_fwd[t] - K[t].dot(C).dot(V_m_fwd[t]))
        if not in_ss_fwd and t >= 5 and t % 100 == 0:
            d1 = np.max(np.abs(K[t] - K[t-1]))
            d2 = np.max(np.abs(V_fwd[t] - V_fwd[t-1]))
            d3 = np.max(np.abs(V_m_fwd[t] - V_m_fwd[t-1]))
            if np.max((d1, d2, d3)) < ss_eps:
                in_ss_fwd = True
                ss_fwd_idx = t
    
    #=====Backward pass=====
    #initialize storage variables
    J = np.zeros((T, n, n))
    x_back = np.zeros((T, n))
    V_back = np.zeros((T, n, n))
    V_adj_back = np.zeros((T, n, n))
    P = np.zeros((T, n, n))
    P_adj = np.zeros((T, n, n))
    #run backward pass (t=T-1,...,0)
    in_ss_bw = False
    for t in range(T-1, -1, -1):
        if t == T-1:
            V_back[t] = V_fwd[t]
            x_back[t] = x_fwd[t]
        else:
            J[t] = J[t+1] if (in_ss_fwd and in_ss_bw) else V_fwd[t].dot(A.T).dot(np.linalg.inv(V_m_fwd[t+1]))
            V_back[t] = V_back[t+1] if (in_ss_fwd and in_ss_bw) else sym(V_fwd[t] + J[t].dot(V_back[t+1] - V_m_fwd[t+1]).dot(J[t].T))
            V_adj_back[t+1] = V_adj_back[t+2] if (in_ss_fwd and in_ss_bw) else V_back[t+1].dot(J[t].T)
            x_back[t] = x_fwd[t] + J[t].dot(x_back[t+1] - A.dot(x_fwd[t]))
            P_adj[t+1] = V_adj_back[t+1] + np.outer(x_back[t+1], x_back[t])
        P[t] = sym(V_back[t] + np.outer(x_back[t], x_back[t]))
        if not in_ss_bw and t <= T-5 and t % 100 == 0:
            d1 = np.max(np.abs(V_back[t+1] - V_back[t]))
            d2 = np.max(np.abs(V_adj_back[t+2] - V_back[t+1]))
            d3 = np.max(np.abs(J[t] - J[t]))
            if np.max((d1, d2, d3)) < ss_eps:
                in_ss_bw = True
        in_ss_fw = t > ss_fwd_idx

    return x_back, P, P_adj

def E_step_batched(y_batches, A, Q, C, R, pi_1, V_1, ss_eps=1e-8):
    num_batches = len(y_batches)
    x_back_batches, P_batches, P_adj_batches = [], [], []
    for i in range(num_batches):
        x_back, P, P_adj = E_step(y_batches[i], A, Q, C, R, pi_1, V_1, ss_eps=ss_eps)
        x_back_batches.append(x_back)
        P_batches.append(P)
        P_adj_batches.append(P_adj)
    return x_back_batches, P_batches, P_adj_batches

def log_likelihood(x, y, A, Q, C, R, pi_1, V_1):
    T = len(y)
    d, n = C.shape #d = observation dim, n = latent dim
    
    y_dev = y - x.dot(C.T)
    R_inv = np.linalg.inv(R)
    y_part = -.5 * np.sum([y_dev[t].dot(R_inv).dot(y_dev[t]) for t in range(T)])
    
    x_dev = x[1:] - x[:-1].dot(A.T)
    Q_inv = np.linalg.inv(Q)
    x_part = -.5 * np.sum([x_dev[t].dot(Q_inv).dot(x_dev[t]) for t in range(T-1)])
    
    x1_dev = x[0] - pi_1
    x1_part = -.5 * x1_dev.dot(np.linalg.inv(V_1)).dot(x1_dev)
                          
    det_R_part   = -.5 * T * np.linalg.slogdet(R)[1]
    det_Q_part   = -.5 * (T - 1) * np.linalg.slogdet(Q)[1]
    det_V_1_part = -.5 * np.linalg.slogdet(V_1)[1]
    full_det_part = det_R_part + det_Q_part + det_V_1_part
    const_part = -.5 * T*(n + d) * np.log(2*np.pi)
    
    ll = y_part + x_part + x1_part + full_det_part + const_part
    return ll

def log_likelihood_batched(x_batches, y_batches, A, Q, C, R, pi_1, V_1):
    total_ll = np.sum([log_likelihood(x_batches[i], y_batches[i], A, Q, C, R, pi_1, V_1) for i in range(len(y_batches))])
    return total_ll

def EM_batched(y_batches, latent_dim, n_iter, ss_eps=1e-8, print_interval=None):
    #initilzie with Factor Analysis
    n = latent_dim
    num_batches = len(y_batches)
    fa = FactorAnalysis(n_components=n)
    x_hat_batches, P_batches, P_adj_batches = [], [], []
    for i in range(num_batches): 
        np.random.seed(42)
        x_hat = fa.fit_transform(y_batches[i])
        T = len(y_batches[i])
        P = np.repeat(np.eye(n)[np.newaxis, :, :], T, axis=0)
        P_adj = np.concatenate((np.zeros((1, n, n)), np.array([np.outer(x_hat[t], x_hat[t-1]) for t in range(1, T)])), axis=0)
        x_hat_batches.append(x_hat)
        P_batches.append(P)
        P_adj_batches.append(P_adj)
    
    #run EM
    ll_vals = np.zeros(n_iter)
    for i in range(n_iter):
        if print_interval is not None and i % print_interval == 0:
            print("iter", i)
        A, Q, C, R, pi_1, V_1 = M_step_batched(y_batches, x_hat_batches, P_batches, P_adj_batches)
        x_hat_batches, P_batches, P_adj_batches = E_step_batched(y_batches, A, Q, C, R, pi_1, V_1, ss_eps=ss_eps)
        ll_vals[i] = log_likelihood_batched(x_hat_batches, y_batches, A, Q, C, R, pi_1, V_1)
    if print_interval is not None:
        print("iter", n_iter)

    return x_hat_batches, P_batches, P_adj_batches, A, Q, C, R, pi_1, V_1, ll_vals

class KalmanFilter(object):

    def fit(self, y_batches, latent_dim, n_iter=20, ss_eps=1e-8, print_interval=None):
        #run EM
        x_hat_batches, P_batches, P_adj_batches, A, Q, C, R, pi_1, V_1, ll_vals = EM_batched(y_batches, latent_dim, n_iter, ss_eps=ss_eps, print_interval=print_interval)

        #store EM results (including E-vals for the training data)
        self.train_x_hat = x_hat_batches
        self.train_P = P_batches
        self.train_P_adj = P_adj_batches
        self.A = A
        self.Q = Q 
        self.C = C
        self.R = R 
        self.pi_1 = pi_1 
        self.V_1 = V_1 
        self.ll_vals = ll_vals
        self.ss_eps = ss_eps

    def transform(self, y):
        #run the Kalman smoother
        x_hat, P, P_adj = E_step(y, self.A, self.Q, self.C, self.R, self.pi_1, self.V_1, ss_eps=self.ss_eps)
        return x_hat





















