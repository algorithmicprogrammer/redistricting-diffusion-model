import numpy as np

def compute_vote_ratio(data):
    d = data.get("T16PRESD", 0.0)
    r = data.get("T16PRESR", 0.0)
    if d + r == 0:
        return 0.0
    return (d - r) / (d + r)



def build_kernel_weights(G,
                         sigma_pop=1000,
                         sigma_pol=0.2,
                         sigma_demo=0.2,
                         alpha=1.0,
                         beta=1.0,
                         gamma=1.0,
                         delta=1.0):

    for u, v, data in G.edges(data=True):

        shared = data.get("shared_perim", 1.0)

        du = G.nodes[u]
        dv = G.nodes[v]

        # --- geographic kernel
        w_geo = shared

        # --- population similarity
        pop_u = du.get("TOT_POP", 0)
        pop_v = dv.get("TOT_POP", 0)

        w_pop = shared * np.exp(-((pop_u-pop_v)**2)/(sigma_pop**2))

        # --- political similarity
        r_u = compute_vote_ratio(du)
        r_v = compute_vote_ratio(dv)

        w_pol = shared * np.exp(-((r_u-r_v)**2)/(sigma_pol**2))

        # --- demographic similarity
        demo_u = du.get("BLACK_POP",0) / max(1,du.get("TOT_POP",1))
        demo_v = dv.get("BLACK_POP",0) / max(1,dv.get("TOT_POP",1))

        w_demo = shared * np.exp(-((demo_u-demo_v)**2)/(sigma_demo**2))

        w = alpha*w_geo + beta*w_pop + gamma*w_pol + delta*w_demo

        data["kernel_weight"] = w


import networkx as nx
import scipy.sparse as sp

def build_laplacian(G):

    nodelist = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodelist)}

    rows, cols, vals = [], [], []
    diag = np.zeros(len(nodelist))

    for u,v,data in G.edges(data=True):

        w = data["kernel_weight"]

        i = idx[u]
        j = idx[v]

        rows.extend([i,j])
        cols.extend([j,i])
        vals.extend([-w,-w])

        diag[i] += w
        diag[j] += w

    L = sp.coo_matrix((vals,(rows,cols)),shape=(len(nodelist),len(nodelist)))

    D = sp.diags(diag)

    return D + L


from scipy.sparse.linalg import spsolve


def solve_qp(L, u0, lam=1.0):

    n = L.shape[0]

    A = L + lam * sp.eye(n)
    b = lam * u0

    u = spsolve(A, b)

    return u
