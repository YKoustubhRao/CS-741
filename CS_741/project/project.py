import os
import random
import numpy as np

NUM_VEC = 100
DIM = 50
SEC_C = 5
SEC_EPS = 2
K_NEAR = 5
EPHEMERAL = 0.001
UP = 100

def gen_sec_db(dim, c): # generates s & w
    return UP*np.random.rand(dim+1), UP*np.random.rand(c)

def rand_db(n, dim): # generates DB, (p1, p2, ..., pn)
    ret = []

    for _ in range(n):
        vec = UP*np.random.rand(dim)
        ret.append(vec)

    return ret

def max_norm_db(db):
    ret = 0
    n = len(db)

    for i in range(n):
        en = np.linalg.norm(db[i])
        ret = max(ret, en)

    return ret

def db_tuple(n, e): # generates (z1, z2, ..., zn)
    ret = []

    for _ in range(n):
        vec = EPHEMERAL*np.random.rand(e)
        ret.append(vec)

    return ret

def get_M_base(dim ,c, e): # generates M_base
    return UP*np.random.rand(dim+1+c+e, dim+1+c+e)

def encrypt_db(db, s, w, M_base, z):
    n = len(db)
    tot_l = M_base.shape[0]
    dim = db[0].shape[0]
    c = w.shape[0]
    e = tot_l-1-dim-c
    ret = []

    for i in range(n):
        p = db[i]
        p1 = np.zeros(tot_l)
        p1[0:dim] = s[0:dim] - 2*p
        p1[dim] = s[dim] + np.sum(p**2)
        p1[dim+1:dim+1+c] = w
        p1[dim+1+c:] = z[i]
        p2 = np.matmul(p1, np.linalg.inv(M_base))
        ret.append(p2)

    return ret

def rand_query(dim): # produces random query vector
    return UP*np.random.rand(dim)

def rand_diag(d):
    N = np.zeros((d, d))
    diag = UP*np.random.rand(d)

    for i in range(d):
        N[i][i] = diag[i]

    return N

def qu_encrypt1(q, beta, N):
    return beta*(np.matmul(q, N))

def do_encrypt(q, max_norm, M_base, tot, c, beta2):
    q_max = np.max(q)
    dim = q.shape[0]
    M_t = np.random.uniform(low=q_max, high=UP*q_max, size=(tot,tot))
    E = np.random.uniform(low=q_max, high=UP*q_max, size=(tot,tot))
    diag = np.random.uniform(low=max_norm, high=UP*max_norm, size=(tot,))

    for i in range(tot):
        M_t[i][i] = diag[i]

    M_sec = np.matmul(M_t, M_base)
    q1 = np.zeros(tot)
    x = np.zeros(c)

    for i in range(c):
        p = random.random() # EPHEMERAL
        if p > 0.67:
            x[i] = 0
        elif p < 0.33:
            x[i] = 0


    q1[0:dim] = q
    q1[dim] = 1
    q1[dim+1:dim+1+c] = x

    Q = np.zeros((tot, tot))
    for i in range(tot):
        Q[i][i] = q1[i]

    q2 = beta2*(np.matmul(M_sec, Q) + E)
    return q2, M_sec

def do_encrypt_db(db, M):
    n = len(db)
    ret = []

    for i in range(n):
        ret.append(np.matmul(db[i], np.linalg.inv(M))) 

    return db

def qu_encrypt2(q, N, tot):
    N1 = np.zeros((tot, tot))
    dim = N.shape[0]

    for i in range(tot):
        N1[i][i] = 1
        if i < dim:
            N1[i][i] = N[i][i]

    q1 = np.matmul(q, np.linalg.inv(N1))
    q2 = np.sum(q1, axis=1)

    return q2

def og_near(q, db, k):
    n = len(db)
    len_dict = {}

    for i in range(n):
        dist = np.linalg.norm(q - db[i])
        len_dict[dist] = i

    shor_dict = dict(sorted(len_dict.items()))
    ordl = []

    for key in shor_dict.keys():
        ordl.append(shor_dict[key])

    ret = ordl[:k]
    # print(shor_dict)
    return ret

def crypt_near(q, db, k):
    n = len(db)
    len_dict = {}

    for i in range(n):
        dist = np.sum(np.multiply(q, db[i]))
        len_dict[dist] = i

    shor_dict = dict(sorted(len_dict.items()))
    ordl = []

    for key in shor_dict.keys():
        ordl.append(shor_dict[key])

    ret = ordl[:k]
    # print(shor_dict)
    return ret

def project():
    db = rand_db(NUM_VEC, DIM)
    s, w = gen_sec_db(DIM, SEC_C)
    z = db_tuple(NUM_VEC, SEC_EPS)
    M_base = get_M_base(DIM, SEC_C, SEC_EPS)
    db1 = encrypt_db(db, s, w, M_base, z)
    max_norm = max_norm_db(db)

    q = rand_query(DIM)
    beta1 = random.uniform(0, UP)
    beta2 = random.uniform(0, UP)
    N = rand_diag(DIM)
    q1 = qu_encrypt1(q, beta1, N)

    q2, M_t = do_encrypt(q1, max_norm, M_base, DIM+1+SEC_C+SEC_EPS, SEC_C, beta2)
    db2 = do_encrypt_db(db1, M_t)

    q3 = qu_encrypt2(q2, N, DIM+1+SEC_C+SEC_EPS)

    near_list_og = og_near(q, db, K_NEAR)
    near_list_crypt = crypt_near(q3, db2, K_NEAR)
    # print(q1)
    print(near_list_og, near_list_crypt)

project()