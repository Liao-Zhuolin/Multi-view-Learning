import numpy as np
#from process import bs_compute_H

def coordinate_descent_beta(beta, H, lambdaa):
    beta_new = beta
    for iter in range(0, 20):
        for ii in range(0, len(beta)):
            for jj in range(ii+1, len(beta)):
                beta_new[ii] = (beta[ii] + beta[jj])/2 + 0.5*(H[jj] - H[ii] )/lambdaa
                beta_new[jj] = beta[ii] + beta[jj] - beta_new[ii]
                if beta_new[ii] < 0:
                    beta_new[ii] = 0
                    beta_new[jj] = beta[ii] + beta[jj]
                if beta_new[jj] < 0:
                    beta_new[jj] = 0
                    beta_new[ii] = beta[ii] + beta[jj]
                beta = beta_new
    return beta_new

def CGD(W, I, para_mu, para_max_iter_diffusion, para_max_iter_alternating, para_thres, para_beta):
    alpha = para_beta/(para_mu + sum(para_beta))
    D = []
    S = []
    A_tmp = [] #6*210*210
    for w in W:
        d = 1.0/np.sqrt(np.sum(w, axis=1))
        d = d[:, np.newaxis]*d
        D.append(d)
        S.append(w*d)
        A_tmp.append(w*0.0)
    #X, Y, V = find(W)
    A = I
    for ii in range(0, para_max_iter_alternating):
    # update A by diffusion
        tmp = np.zeros((para_max_iter_diffusion, 1), dtype=float)
        for iter in range(0, para_max_iter_diffusion):
        #         tic
            for v in range(0, len(W)):
                A_tmp[v] = alpha[v]*np.dot(np.dot(S[v], A), S[v].T)
        #         toc
            A_new = sum(A_tmp, 0) + (1-sum(alpha))*I
            A = A_new

        # update beta
        #H = np.zeros((len(W), 1), dtype=float)
        #for v in range(0, len(W)):
        #    H[v] = bs_compute_H(A, D[v], X[v], Y[v], V[v])
        #H = [11.4598, 11.4774, 11.3936, 11.3356, 11.3755, 11.5692]
        H = [0, 1, 0, 1, 1, 1]
        para_lambda = 19
        para_beta = coordinate_descent_beta(para_beta, H, para_lambda)
        alpha = para_beta/(para_mu + sum(para_beta))
    #A = single(A)
    out_beta = para_beta
    return A, out_beta

def find(W):
    X = []
    Y = []
    V = []
    for z in W:
        x = []
        y = []
        v = []
        for j in range(0, len(z[0])):
            for i in range(0, len(z)):
                if z[i][j] != 0:
                    x.append(i)
                    y.append(j)
                    v.append(z[i][j])
        X.append(x)
        Y.append(y)
        V.append(v)
    return X, Y, V
