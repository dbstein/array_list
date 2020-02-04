import numpy as np
import numba
import time
from array_list.array_list import create_selector

# use multiple cores?
ispar = True

################################################################################
# 1D Arrays

# number of arrays in 'list'
NL = 300
# maximum number of elements in each array
NEM = 10000
# choose elements in each array
NE = np.random.choice(np.arange(2, NEM), NL)
# construct list of arrays
L = [np.random.rand(NEH) for NEH in NE]

# number of test points
NT = 1000*1000
sel1 = np.random.choice(NL, NT)
out1 = np.empty(NT)
out2 = np.empty(NT)

# get second selection index
@numba.njit(parallel=ispar)
def fill(sel1, NE):
	sel2 = np.empty_like(sel1)
	for i in numba.prange(sel1.size):
		sel2[i] = np.random.choice(NE[sel1[i]])
	return sel2
sel2 = fill(sel1, NE)

### test pure python selection from list of arrays
def pure_python(L, sel1, sel2, out):
	for i in range(sel1.size):
		out[i] = L[sel1[i]][sel2[i]]

st = time.time()
pure_python(L, sel1, sel2, out1)
print('Time, pure python  (ms): {:0.1f}'.format((time.time()-st)*1000))

### test array_list for selection from list of arrays
S = create_selector(L)

@numba.njit(parallel=ispar)
def using_array_list(S, sel1, sel2, out):
	for i in numba.prange(sel1.size):
		out[i] = S.get(sel1[i])[sel2[i]]

using_array_list(S, sel1, sel2, out2)
st = time.time()
using_array_list(S, sel1, sel2, out2)
print('Time, selector     (ms): {:0.1f}'.format((time.time()-st)*1000))
print(np.allclose(out1, out2))

################################################################################
# 2D Arrays

# second dim for 2D array
NS = 10
# construct list of arrays
L = [np.random.rand(NEH, NS) for NEH in NE]

# reuse sel1/sel2 from before
out1 = np.empty([NT, NS])
out2 = np.empty([NT, NS])

### test pure python selection from list of arrays
def pure_python(L, sel1, sel2, out):
	for i in range(sel1.size):
		out[i] = L[sel1[i]][sel2[i]]

st = time.time()
pure_python(L, sel1, sel2, out1)
print('Time, pure python  (ms): {:0.1f}'.format((time.time()-st)*1000))

### test array_list for selection from list of arrays
S = create_selector(L)

@numba.njit(parallel=ispar)
def using_array_list(S, sel1, sel2, out):
	for i in numba.prange(sel1.size):
		out[i] = S.get(sel1[i])[sel2[i]]

using_array_list(S, sel1, sel2, out2)
st = time.time()
using_array_list(S, sel1, sel2, out2)
print('Time, selector     (ms): {:0.1f}'.format((time.time()-st)*1000))
print(np.allclose(out1, out2))

################################################################################
# 3D Arrays

# second and third dim for 3D array
NS = 2
# construct list of arrays
L = [np.random.rand(NEH, NS, NS) for NEH in NE]

# reuse sel1/sel2 from before
out1 = np.empty([NT, NS, NS])
out2 = np.empty([NT, NS, NS])

### test pure python selection from list of arrays
def pure_python(L, sel1, sel2, out):
	for i in range(sel1.size):
		out[i] = L[sel1[i]][sel2[i]]

st = time.time()
pure_python(L, sel1, sel2, out1)
print('Time, pure python  (ms): {:0.1f}'.format((time.time()-st)*1000))

### test array_list for selection from list of arrays
S = create_selector(L)

@numba.njit(parallel=ispar)
def using_array_list(S, sel1, sel2, out):
	for i in numba.prange(sel1.size):
		out[i] = S.get(sel1[i])[sel2[i]]

using_array_list(S, sel1, sel2, out2)
st = time.time()
using_array_list(S, sel1, sel2, out2)
print('Time, selector     (ms): {:0.1f}'.format((time.time()-st)*1000))
print(np.allclose(out1, out2))

################################################################################
# 1D Arrays of Ints

# construct list of arrays
L = [np.round(np.random.rand(NEH)*10).astype(int) for NEH in NE]

# number of test points
NT = 1000*1000
sel1 = np.random.choice(NL, NT)
out1 = np.empty(NT, dtype=int)
out2 = np.empty(NT, dtype=int)

# get second selection index
@numba.njit(parallel=ispar)
def fill(sel1, NE):
	sel2 = np.empty_like(sel1)
	for i in numba.prange(sel1.size):
		sel2[i] = np.random.choice(NE[sel1[i]])
	return sel2
sel2 = fill(sel1, NE)

### test pure python selection from list of arrays
def pure_python(L, sel1, sel2, out):
	for i in range(sel1.size):
		out[i] = L[sel1[i]][sel2[i]]

st = time.time()
pure_python(L, sel1, sel2, out1)
print('Time, pure python  (ms): {:0.1f}'.format((time.time()-st)*1000))

### test array_list for selection from list of arrays
S = create_selector(L)

@numba.njit(parallel=ispar)
def using_array_list(S, sel1, sel2, out):
	for i in numba.prange(sel1.size):
		out[i] = S.get(sel1[i])[sel2[i]]

using_array_list(S, sel1, sel2, out2)
st = time.time()
using_array_list(S, sel1, sel2, out2)
print('Time, selector     (ms): {:0.1f}'.format((time.time()-st)*1000))
print(np.allclose(out1, out2))







