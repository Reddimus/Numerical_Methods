'''
Single Equation:

Bisection Method examples:

Ex 1:
(e^x) - (x^2) = 0 has a root in the interval [-1, 0]. Use bisection method to find it. 
How many iterations are needed to get results that agree to five significant digits?

Ex 2:
x - x ^1/3 - 2 = 0, interval = [3, 4]


Newtons Raphson method examples:

Ex 3:
e^x - 2x^2 = 0, guess = 2.6

Ex 4:
x - x^(1/3) - 2, guess = 3

Secant method examples:

Ex 5:
x - x^(1/3) - 2, guess = [3.0, 3.2]

Ex 6:
x^2 + e^x - 5, guess = [0.9, 1.1]
'''
import pandas as pd 							# neatly display array/ result into data frame
from scipy.optimize import root_scalar, fsolve 	# for test case

class Math:
	def __init__(self, function: int, interval: list[int] = None, guess: int or list[float] = None):
		self.function = function
		self.interval = interval
		self.guess = guess 

	# Time and space complexit: O(log(b - a)) ~= O(log n), the size of the table list grows logarithmically 
	# with the size of the interval
	def bisection_method(self) -> tuple:
		func, inter = self.function, self.interval
		a, b = inter 	# root in range(left pointer(a) & right pointer(b))
		a_res, b_res = float("-infinity"), float("infinity")	# temporary values
		table = []
		while a < b and (round(a_res, 5) != round(b_res, 5)):
			a_res, b_res = func(a), func(b) 	# insert a & b into function
			error = (b - a) / 2
			mid_pt = (a + b) / 2
			iter_row = [a, mid_pt, b, error]
			rounded_iter = [round(x, 5) for x in iter_row]	# round numbers to 5th decimal place
			table.append(rounded_iter)

			# Replace pointer who's result is furthest from 0 or f(a)f(b) < 0 for next iteration
			if abs(a_res) > abs(b_res):
				a = mid_pt
			else:
				b = mid_pt
		root = round(a, 5)
		#self.testcases(expected = root) 	# Verify with other predefined method(s)
		table_df = self.arr_to_df(arr = table, column_names = ["a", "mid", "b", "error"])
		return root, table_df

	# Time and space complexity: O(log n) or O(n^(1/2)), where n is the number of iterations
	# Time complexity: newton_raphson_method < bisection_method in most cases
	# Depends on several factors such as the initial guess, the number of iterations, 
	# and the accuracy of the function and its derivative.
	def newton_raphson_method(self) -> tuple:
		func, guess = self.function, self.guess
		table = []
		prev_guess = float("infinity")	# temporary valye
		while round(prev_guess, 5) != round(guess, 5):
			func_res = func(guess)
			deriv_res = self.derivative(func, guess)
			delta_x = func_res / deriv_res
			# store iteration data into table
			iter_row = [guess, func_res, deriv_res, delta_x]
			rounded_iter = [round(x, 5) for x in iter_row]	# round numbers to 5th decimal place
			table.append(rounded_iter)

			prev_guess = guess 	# store prev guess for comparison
			guess -= delta_x	# update guess
		root = round(guess, 5)
		#self.testcases(expected = root)
		table_df = self.arr_to_df(arr = table, column_names = ["x (guess)", "f", "f'", "delta x"])
		return root, table_df

	# Time and space complexity: O(log n), where n is the number of iterations required for convergence.
	def secant_method(self) -> tuple:
		func, guess = self.function, self.guess
		guess_0, guess_1 = guess 	# secant method must contain two guesses
		# solve initial data
		res_0, res_1 = func(guess_0), func(guess_1)
		if abs(res_0) < abs(res_1):
			guess_0, guess_1 = guess_1, guess_0
			res_0, res_1 = res_1, res_0
		# iterate to solve
		new = float("infinity")	# temporary value
		table = []
		while round(guess_0, 5) != round(new, 5):
			# Solve for root
			new = guess_1 - ((res_1 * (guess_0 - guess_1)) / (res_0 - res_1))
			res_new = func(new)
			# store iteration data into table
			iter_row = [guess_0, guess_1, new, res_0, res_1, res_new]
			rounded_iter = [round(x, 5) for x in iter_row]	# round numbers to 5th decimal place
			table.append(rounded_iter)
			# rearrange guesses and function values
			guess_0, guess_1 = guess_1, new
			res_0, res_1 = res_1, res_new
		root = round(new, 5)
		#self.testcases(expected = root)
		table_df = self.arr_to_df(arr = table, column_names = ["x0", "x1", "new", "f0", "f1", "fnew"])
		return root, table_df

	def derivative(self, func: int, x: int, h: int = 1e-10) -> int:
		return (func(x + h) - func(x)) / h 	# finite difference method

	def arr_to_df(self, arr: list[list[int]], column_names: list[str]) -> pd.DataFrame: # neatly display table; rename column and row
		table_df = pd.DataFrame(arr, columns = column_names, index = range(1, len(arr)+1))
		return table_df

	def testcases(self, expected: int):
		func, inter, guess = self.function, self.interval, self.guess

		# Test case 1
		if not inter and type(guess) is int:
			inter = [(guess * .75), (guess * 1.25)]
		elif not inter and type(guess) is list:
			inter = [(guess[0] * .75), (guess[1] * 1.25)]
		ans = root_scalar(func, bracket = inter)
		root_ans = round(ans.root, 5)
		assert expected == root_ans, f"Expected: {expected} | Got: {root_ans}"
		# Test case 2
		if not guess and inter:
			a, b = inter
			guess = (a + b) / 2
		elif type(guess) is list and inter:
			guess = (guess[0] + guess[1]) / 2
		root_ans = float(fsolve(func, guess))
		root_ans = round(root_ans, 5)
		assert expected == root_ans, f"Expected: {expected} | Got: {root_ans}"

e = 2.7182818284590452353602874713527

# Ex 1:
print("Ex 1: (e^x) - (x^2) = 0, interval = [-1, 0]")
ex1_root, ex1_table = Math(function = lambda x: e**x - x**2, interval = [-1, 0]).bisection_method()
print(ex1_table)
print("Root =", ex1_root, "\n")

# Ex 2:
print("Ex 2: x - x ^1/3 - 2 = 0, interval = [3, 4]")
ex2_root, ex2_table = Math(function = lambda x: x - (x**(1/3)) - 2, interval = [3, 4]).bisection_method()
print(ex2_table)
print("Root =", ex2_root, "\n")

# Ex 3:
print("Ex 3: e^x - 2x^2 = 0, guess = 2.6")
ex3_root, ex3_table = Math(function = lambda x: e**x - 2*x**2, guess = 2.6).newton_raphson_method()
print(ex3_table)
print("Root =", ex3_root, "\n")

# Ex 4:
print("Ex 4: x - x^(1/3) - 2, guess = 3")
ex4_root, ex4_table = Math(function = lambda x: x - x**(1/3) - 2, guess = 3).newton_raphson_method()
print(ex4_table)
print("Root =", ex4_root, "\n")

# Ex 5:
print("Ex 5: x - x^(1/3) - 2, guess = [3.0, 3.2]")
ex5_root, ex5_table = Math(function = lambda x: x - x**(1/3) - 2, guess = [3.0, 3.2]).secant_method()
print(ex5_table)
print("Root =", ex5_root, "\n")

# Ex 6:
print("Ex 6: x^2 + e^x - 5, guess = [0.9, 1.1]")
ex6_root, ex6_table = Math(function = lambda x: x**2 + e**x - 5, guess = [0.9, 1.1]).secant_method()
print(ex6_table)
print("Root =", ex6_root, "\n")