import os
import random
import numpy as np

NUM_VEC = 100
DIM = 50
SEC_C = 5
SEC_EPS = 2
K_NEAR = 5
EPHEMERAL = 0.00001

def gen_sec_db(dim, c): # generates s & tau
    return [np.random.rand(dim+1), np.random.rand(c)]

def rand_db(n, dim): # generates DB, (p1, p2, ..., pn)
    ret = []

    for _ in range(n):
        vec = 10*np.random.rand(dim)
        ret.append(vec)

    return ret

def rand_query(dim): # produces random query vector
    return np.random.rand(dim)

def db_tuple(n, e): # generates (v1, v2, ..., vn)
    ret = []

    for _ in range(n):
        vec = EPHEMERAL*np.random.rand(e)
        ret.append(vec)

    return ret

def per_query(c): # generates r(q) & beta(q)
    return [EPHEMERAL*np.random.rand(c), EPHEMERAL*random.uniform(0.0, 1.0)]

def secret_perm(dim, c, e): # generates pi
    arr = np.arange(dim+1+c+e)
    np.random.shuffle(arr)
    return arr

def get_M(dim ,c, e): # generates M
    return np.random.rand(dim+1+c+e, dim+1+c+e)

def mat_shuf(M, pi): # shuffles column of M according to pi
    sz = M.shape[0]
    ret = np.zeros((sz, sz))
    c = 0

    for i in pi:
        ret[:][c] = M[:][i]
        c += 1

    return ret

def encrypt_query(q, M, pi, c):
    tot_l = M.shape[0]
    dim = q.shape[0]
    r, beta = per_query(c)
    q1 = np.zeros(tot_l)
    q1[0:dim] = q
    q1[dim] = 1
    q1[dim+1:dim+1+c] = r
    M1 = mat_shuf(M, pi)
    q2 = beta*np.matmul(M1, q1)
    return q2

def encrypt_db(db, s, tau, M, pi, v):
    n = len(db)
    tot_l = M.shape[0]
    dim = db[0].shape[0]
    c = tau.shape[0]
    e = tot_l-1-dim-c
    ret = []

    for i in range(n):
        p = db[i]
        p1 = np.zeros(tot_l)
        p1[0:dim] = s[0:dim] - 2*p
        p1[dim] = s[dim] + np.sum(p**2)
        p1[dim+1:dim+1+c] = tau
        p1[dim+1+c:] = v[i]
        M1 = mat_shuf(M, pi)
        p2 = np.matmul(p1, np.linalg.inv(M1))
        ret.append(p2)

    return ret

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
    return ret

def project():
    db = rand_db(NUM_VEC, DIM)
    s, tau = gen_sec_db(DIM, SEC_C)
    v = db_tuple(NUM_VEC, SEC_EPS)
    pi = secret_perm(DIM, SEC_C, SEC_EPS)
    M = get_M(DIM, SEC_C, SEC_EPS)
    db1 = encrypt_db(db, s, tau, M, pi, v)
    q = rand_query(DIM)
    near_list_og = og_near(q, db, K_NEAR)
    q1 = encrypt_query(q, M, pi, SEC_C)
    near_list_crypt = crypt_near(q1, db1, K_NEAR)
    print(near_list_og, near_list_crypt)

project()