"""
David Clark
2019
"""


def sym(M):
    return .5*(M + M.T)

def M_step(y, x_hat, P, P_adj):
    T = len(x_hat)
    C = (y.T.dot(x_hat)).dot(np.linalg.inv(P.sum(axis=0)))
    R = (1. / T) * (y.T.dot(y) - C.dot(x_hat.T).dot(y))
    P_adj_sum = P_adj[1:].sum(axis=0)
    A = P_adj_sum.dot(np.linalg.inv(P[:-1].sum(axis=0)))
    Q = (1. / (T - 1)) * (P[1:].sum(axis=0) - A.dot(P_adj_sum.T))
    pi_1 = x_hat[0]
    V_1 = P[0] - np.outer(x_hat[0], x_hat[0])    
    return A, Q, C, R, pi_1, V_1
    
def E_step(y, A, Q, C, R, pi_1, V_1):
    T = len(y)
    d, n = C.shape #n -- latent dim, d -- full dim
    
    #=====Forward pass=====
    #initialize storage variables
    print("Allocating...")
    x_m_fwd = np.zeros((T, n))
    x_fwd = np.zeros((T, n))
    V_m_fwd = np.zeros((T, n, n))
    V_fwd = np.zeros((T, n, n))
    K = np.zeros((T, n, d))
    print("Done.")
    #run forward pass (t=0,...,T-1)
    in_ss_fwd = False
    ss_fwd_idx = -1
    eps = 1e-12
    for t in range(T):
        if t == 0:
            x_m_fwd[t] = pi_1
            V_m_fwd[t] = sym(V_1)
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
            if np.max((d1, d2, d3)) < eps:
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
            V_back[t] = sym(V_fwd[t])
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
            if np.max((d1, d2, d3)) < eps:
                in_ss_bw = True
        in_ss_fw = t > ss_fwd_idx
    return x_back, P, P_adj

def loglikelihood(x, y, A, Q, C, R, pi_1, V_1):
    T = len(y)
    d, n = C.shape #n -- latent dim, d -- full dim
    
    y_dev = y - x.dot(C.T)
    R_inv = np.linalg.inv(R)
    y_part = -.5 * np.sum([y_dev[t].dot(R_inv).dot(y_dev[t]) for t in range(T)])
    
    x_dev = x[1:] - x[:-1].dot(A.T)
    Q_inv = np.linalg.inv(Q)
    x_part = -.5 * np.sum([x_dev[t].dot(Q_inv).dot(x_dev[t]) for t in range(T-1)])
    
    #print("val", np.max(np.abs(Q)))
    #print("sym", np.max(np.abs(Q - Q.T)))
    
    x1_dev = x[0] - pi_1
    x1_part = -.5 * x1_dev.dot(np.linalg.inv(V_1)).dot(x1_dev)
                          
    det_R_part   = -.5 * T * np.linalg.slogdet(R)[1]
    det_Q_part   = -.5 * (T - 1) * np.linalg.slogdet(Q)[1]
    det_V_1_part = -.5 * np.linalg.slogdet(V_1)[1]
    full_det_part = det_R_part + det_Q_part + det_V_1_part
    const_part = -.5 * T*(n + d) * np.log(2*np.pi) #just for fun :)
    
    ll = y_part + x_part + x1_part + full_det_part + const_part
    return ll
    

def EM(y, n, n_iter=10):
    #initilzie with Factor Analysis
    fa = FactorAnalysis(n_components=n)
    x_hat = fa.fit_transform(y)
    T = len(y)
    P = np.repeat(np.eye(n)[np.newaxis, :, :], T, axis=0)
    #P_adj = np.array([np.outer(x_hat[t], x_hat[t-1]) for t in range(1, T)])
    #P_adj = np.concatenate((np.zeros((1, n, n)), P_adj), axis=0)
    P_adj = np.zeros((T, n, n))
    
    #run EM
    ll_vals = np.zeros(n_iter)
    for i in range(n_iter):
        if i % 10 == 0:
            print(i, ll_vals[i-1])
        A, Q, C, R, pi_1, V_1 = M_step(y, x_hat, P, P_adj)
        x_hat, P, P_adj = E_step(y, A, Q, C, R, pi_1, V_1)
        ll_vals[i] = loglikelihood(x_hat, y, A, Q, C, R, pi_1, V_1)
        
    return ll_vals, x_hat
    


np.random.seed(42)
y = M1["M1"][:50000, :]
y = y - y.mean(axis=0) #/y.std(axis=0)
ll_vals, x = EM(y, 2, n_iter=300)
plt.plot(ll_vals)
print(ll_vals[-1])
        
        