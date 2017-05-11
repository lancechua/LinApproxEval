import mpmath as mp
import numpy as np
import pathos.multiprocessing as pmp 
import itertools as it
import timeit

def LinApproxEval(err_f, fn, m, b, bounds, fn_kwargs={}, use_mp=True):
	'''
	Evaluates accuracy of a function's linear approximation

	returns absolute error / total curve area over approximation bounds
	absolute error = abs(curve area - linear area)

	Input
		err_f = error evaluation function
			err_f parameters must be: 
				fn_param = lambda x: fn(x , *args)
				m, b, x0, x1 = slope, intercept, bounds of approx segment
		fn = function approximated, must take x as first argument 
		m, list of length n = slopes; sorted
		b, list of length n = intercept; sorted
		bounds, list of length n+1 = linear approximation bounds; sorted
			first element = lower bound, last element = upper bound
		fn_kwargs = additional keyword arguments for fn
		use_mp = use multiprocessing, only use if segments are many (else overhead not worth it) 
	'''

	p = pmp.Pool()

	check = np.diff(bounds)
	if not np.all(check >= 0):
		raise ValueError("bounds parameter should be increasing.")

	fn_param = lambda x: fn(x, **fn_kwargs)

	if use_mp:
		helper = lambda x: err_f(*x) 
		run_er = sum(p.map(helper, zip(it.repeat(fn_param), m, b, bounds[:-1], bounds[1:])))
	else:
		run_er = 0
		for lb_i, ub_i, m_i, b_i in zip(bounds[:-1], bounds[1:], m, b):
			run_er+= err_f(fn_param, m_i, b_i, lb_i, ub_i)

	return run_er / mp.quad(fn_param, [bounds[0], bounds[-1]])


def obj1(fn_param, m, b, x0, x1):
	lin_y0 = mp.mp.mpf(m) * x0 + b
	lin_y1 = mp.mp.mpf(m) * x1 + b
	lin_area = (lin_y0 + lin_y1) / 2 * (x1 - x0)
	fn_area = mp.quad(fn_param, [x0, x1])

	return abs(lin_area - fn_area)

def obj2(fn_param, m, b, x0, x1):
	obj_f = lambda x: (fn_param(x) - (m*x + b))**2
	return mp.quad(obj_f,[x0,x1]) ** 1/2


if __name__ == '__main__':
	my_fn = lambda x: x**2
	my_m = [1,3]
	my_b = [0,-2]
	my_bounds = [0,1,2]
	print(LinApproxEval(obj1, my_fn, my_m, my_b, my_bounds))
	print(LinApproxEval(obj1, my_fn, my_m, my_b, my_bounds, use_mp=False))
	print(LinApproxEval(obj2, my_fn, my_m, my_b, my_bounds))
	print(LinApproxEval(obj2, my_fn, my_m, my_b, my_bounds, use_mp=False))
	


