import mpmath as mp
import numpy as np

def LinApproxEval(fn, m, b, bounds, fn_kwargs={}):
	'''
	Evaluates accuracy of a function's linear approximation

	returns absolute error / total curve area over approximation bounds
	absolute error = abs(curve area - linear area)

	Input
		fn = function approximated, must take x as first argument 
		m, list of length n = slopes; sorted
		b, list of length n = intercept; sorted
		bounds, list of length n+1 = linear approximation bounds; sorted
			first element = lower bound, last element = upper bound
	'''

	check = np.diff(bounds)
	if not np.all(check >= 0):
		raise ValueError("bounds parameter should be increasing.")

	fn_param = lambda x: fn(x, **fn_kwargs)

	run_er = 0
	for lb_i, ub_i, m_i, b_i in zip(bounds[:-1], bounds[1:], m, b):
		lin_y0 = mp.mp.mpf(m_i) * lb_i + b_i
		lin_y1 = mp.mp.mpf(m_i) * ub_i + b_i
		lin_area = (lin_y0 + lin_y1) / 2 * (ub_i - lb_i)

		fn_area = mp.quad(fn_param, [lb_i, ub_i])

		run_er+= abs(lin_area - fn_area)

	return run_er / mp.quad(fn_param, [bounds[0], bounds[-1]])

if __name__ == '__main__':
	my_fn = lambda x: x**2
	my_m = [1,3]
	my_b = [0,-2]
	my_bounds = [0,1,2]
	print(LinApproxEval(my_fn, my_m, my_b, my_bounds))


	



