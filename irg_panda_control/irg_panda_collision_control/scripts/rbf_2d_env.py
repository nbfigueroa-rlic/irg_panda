
import numpy as np
from scipy.interpolate import RectBivariateSpline
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import cv2 as cv

class RBFEnvironment():
	'''
	a 2D environment defined as the sum of several positive RBF functions
	support Gamma function query (returns a value greater than 1 in free space, 1 on obstacle boundary, and smaller than 1 in obstacle)
	and its gradient (pointing "outward" toward free space)
	also supports lidar sensor simulation.
	'''
	def __init__(self, N_points=15, obs_low=-0.7, obs_high=0.7, rbf_gamma=25, threshold=0.9, img_res=150, img_range=1.2, enable_lidar=True):
		self.N_points = N_points
		self.obs_low = obs_low
		self.obs_high = obs_high
		self.rbf_gamma = rbf_gamma
		self.threshold = threshold
		self.img_res = img_res
		self.img_range = img_range
		self.enable_lidar = enable_lidar
		self.center = None
		self.boundary_points = None
		self.reset()
		self.set_boundary_points()

	def reset(self, obs_override=None):
		if obs_override is None:
			self.points = np.random.uniform(low=self.obs_low, high=self.obs_high, size=(self.N_points, 2))
		else:
			assert len(obs_override) == self.N_points, 'incorrect number of obstacle points'
			assert np.all(obs_override < self.obs_high) and np.all(obs_override > self.obs_low), 'obstacle bound exceeded'
			self.points = obs_override
		self.env_img = None
		self.interpolator = None

	def gamma_batch(self, p):
		kernel_val = - rbf_kernel(self.points, p, self.rbf_gamma).sum(axis=0)
		return kernel_val + self.threshold + 1

	def gamma(self, p):
		return self.gamma_batch(np.array([p]))[0]

	def gamma_grad_batch(self, p):
		kernel_vals = rbf_kernel(self.points, p, gamma=self.rbf_gamma)
		coefs = 2 * self.rbf_gamma * np.array([[p_j - p_i for p_j in p] for p_i in self.points]).reshape(self.N_points, p.shape[0], 2)
		coefs = np.transpose(coefs, [2, 0, 1])
		grad_vals = kernel_vals * coefs  # 2 x N_points x p.shape[0]
		grad_vals = grad_vals.sum(axis=1).T  # p.shape[0] x 2
		return grad_vals

	def gamma_grad(self, p):
		return self.gamma_grad_batch(np.array([p]))[0]

	def __call__(self, p):
		return self.gamma(p)
	def grad(self, p):
		return self.gamma_grad(p)
	def draw(self):
		plt.imshow(self.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')
		plt.plot(self.boundary_points.T[0,:], self.boundary_points.T[1,:], 'go')

	def set_boundary_points(self):
		og_img = np.uint8(self.env_img)

		xs = np.linspace(-self.img_range, self.img_range, self.img_res)
		ys = np.linspace(-self.img_range, self.img_range, self.img_res)
		x_grid, y_grid = np.meshgrid(xs, xs)
		xy_grid = np.stack((x_grid, y_grid), axis=2)
		kernel = np.ones((3,3),np.uint8)
		erosion = cv.dilate(og_img, kernel, iterations = 1)
		erosion = cv.dilate(erosion, kernel, iterations = 1)

		boundary_points = np.nonzero(og_img - erosion)
		self.boundary_points = xy_grid[boundary_points]

	def nearest_boundary_pt(self,p):
		dist = np.sum((self.boundary_points - p)**2, axis=1)
		point = self.boundary_points[np.argmin(dist)]
		return point

	@property
	def env_img(self):
		if self.__env_img is None:
			xs = np.linspace(-self.img_range, self.img_range, self.img_res)
			ys = np.linspace(-self.img_range, self.img_range, self.img_res)
			xy_grid = np.stack([a.flatten() for a in np.meshgrid(xs, ys)], axis=1)
			self.__env_img = (self.gamma_batch(xy_grid) > 1).astype('int').reshape(self.img_res, self.img_res)
		return self.__env_img

	@env_img.setter
	def env_img(self, env_img):
		self.__env_img = env_img

	@property
	def interpolator(self):
		if self.__interpolator is None:
			xs = np.linspace(-self.img_range, self.img_range, self.img_res)
			ys = np.linspace(-self.img_range, self.img_range, self.img_res)
			self.__interpolator = RectBivariateSpline(x=xs, y=ys, z=self.env_img)
		return self.__interpolator

	@interpolator.setter
	def interpolator(self, interpolator):
		self.__interpolator = interpolator

	def lidar_sensor(self, is_free_func, p, num_readings=16, max_range=1, dist_incr=0.002, theta=0):
		dists = np.arange(dist_incr, max_range, dist_incr)
		thetas = np.linspace(0, 2 * np.pi, num_readings + 1)[:-1]
		lidar_xs = np.array([dists * np.cos(th) for th in thetas]).flatten()
		lidar_ys = np.array([dists * np.sin(th) for th in thetas]).flatten()
		x, y = p
		xs = lidar_xs + x
		ys = lidar_ys + y
		is_free = is_free_func(xs, ys).reshape(num_readings, -1).astype('int')
		idxs = np.argwhere(is_free==0)
		reading = {th_idx: max_range for th_idx in range(num_readings)}
		for th_idx, incr in idxs:
			if reading[th_idx] > (incr + 1) * dist_incr:
				reading[th_idx] = (incr + 1) * dist_incr
		return np.array([reading[th_idx] for th_idx in range(num_readings)])

	def lidar_fast(self, p, num_readings=16, max_range=1, dist_incr=0.002, theta=0):
		is_free_func = lambda xs, ys: self.interpolator.ev(ys, xs) > 0.1
		return self.lidar_sensor(is_free_func, p, num_readings, max_range, dist_incr, theta)

	def lidar_exact(self, p, num_readings=16, max_range=1, dist_incr=0.002, theta=0):
		is_free_func = lambda xs, ys: self.gamma_batch(np.stack([xs, ys], axis=1)) > 1
		return self.lidar_sensor(is_free_func, p, num_readings, max_range, dist_incr, theta)

	def lidar(self, p, mode='fast', num_readings=16, max_range=1, dist_incr=0.002, theta=0):
		if mode == 'fast':
			return self.lidar_fast(p, num_readings, max_range, dist_incr, theta)
		elif mode == 'exact':
			return self.lidar_exact(p, num_readings, max_range, dist_incr, theta)
		else:
			raise Exception(f'Unrecognized mode: {mode}')

	def get_env_img(self):
		if self.env_img is not None:
			return self.env_img
		xs = np.linspace(-1.2, 1.2, self.img_res)
		ys = np.linspace(-1.2, 1.2, self.img_res)
		xy_grid = np.stack([a.flatten() for a in np.meshgrid(xs, ys)], axis=1)
		self.env_img = (self.gamma_batch(xy_grid) > 1).astype('int').reshape(self.img_res, self.img_res)
		return self.env_img

	def get_img_interpolator(self):
		if self.interpolator is not None:
			return self.interpolator


class PhysicsSimulator():
	def __init__(self, env):
		self.env = env

	def reset(self, p):
		assert self.is_free(*p), 'the reset position must be in free space'
		self.x = p[0]
		self.y = p[1]

	def agent_pos(self):
		return np.array([self.x, self.y])

	def is_collision(self, x, y):
		return self.env.gamma([x, y]) <= 1

	def is_free(self, x, y):
		return not self.is_collision(x, y)

	def find_contact(self, x, y, dx, dy, iteration=5):
		'''find the contact point with obstacle of moving (x, y) in (dx, dy) direction by interval halving'''
		last_good = None
		last_good_u = None
		factor = 0.5
		u = 0.5
		for _ in range(iteration):
			cur_point = [x + dx * u, y + dy * u]
			factor = factor / 2
			if self.is_free(*cur_point):
				last_good = cur_point
				assert last_good_u is None or last_good_u < u
				last_good_u = u
				u = u + factor
			else:
				u = u - factor
		if last_good is None:
			last_good = [x, y]
			last_good_u = 0
		return last_good, last_good_u

	def step(self, dp, return_collide=False):
		assert self.is_free(self.x, self.y)
		dx, dy = dp
		if self.is_free(self.x + dx, self.y + dy):
			self.x += dx
			self.y += dy
			if return_collide:
				return np.array([self.x, self.y]), 0
			else:
				return np.array([self.x, self.y])
		# in collision
		# 1. find closest point without collision by interval halving
		contact, u = self.find_contact(self.x, self.y, dx, dy)
		# 2. projecting the remaining (dx, dy) along the tangent
		normal = self.env.gamma_grad(contact)
		tangent = np.array([normal[1], -normal[0]])
		remaining = ([dx * (1 - u), dy * (1 - u)])
		proj = np.dot(tangent, remaining) / np.linalg.norm(tangent)**2 * tangent
		end = proj + contact
		# 3. if collision, then find again the contact point along the tangent direction
		if self.is_collision(*end):
			end, _ = self.find_contact(contact[0], contact[1], proj[0], proj[1])
		self.x = end[0]
		self.y = end[1]
		if return_collide:
			return np.array([self.x, self.y]), 1
		else:
			return np.array([self.x, self.y])

class RBF2dGym():
	def __init__(self, time_limit=200, dxy_limit=0.03, **kwargs):
		self.env = RBFEnvironment(**kwargs)
		self.time_limit = time_limit
		self.dxy_limit = dxy_limit
		self.target_pos = [1, 1]

	def reset(self, obs_override=None):
		self.env.reset(obs_override=obs_override)
		self.sim = PhysicsSimulator(self.env)
		self.sim.reset([-1, -1])
		self.t = 0
		return self.s()

	def step(self, a):
		assert self.t < self.time_limit, 'time limit already exceeded'
		self.t += 1
		a_clip = np.clip(a, -1, 1) * self.dxy_limit
		self.sim.step(a_clip)
		dist = np.linalg.norm(self.target_pos - self.sim.agent_pos())
		ax, ay = self.sim.agent_pos()
		out_of_bound = not (-1.2 < ax < 1.2 and -1.2 < ay < 1.2)
		done = (self.t == self.time_limit or out_of_bound or dist < 0.03)
		r = - dist
		if out_of_bound:
			r = r - 500
		return self.s(), r, done, None

	def s(self):
		s = self.sim.agent_pos()
		if self.env.enable_lidar:
			s = np.concatenate([s, self.env.lidar(s)])
		return s

	def log_prior(self, value):
		assert np.all(abs(value) <= 0.7)
		return 0

def demo_grad_field():
	while True:
		env = RBFEnvironment()
		plt.imshow(env.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')
		xs = np.linspace(-1, 1, 20)
		ys = np.linspace(-1, 1, 20)
		xys = np.stack([a.flatten() for a in np.meshgrid(xs, ys)], axis=1)
		grads = env.gamma_grad_batch(xys)
		for (x, y), grad in zip(xys, grads):
			plt.arrow(x, y, grad[0] * 0.01, grad[1] * 0.01, head_length=0.05, head_width=0.02)
		plt.axis('off')
		plt.show()

def plot_lidar(p, reading, plotted=None, theta=0):
	num_readings = len(reading)
	angles = np.linspace(0, 2 * np.pi, num_readings + 1)[:-1] + theta
	dxs = np.array([np.cos(th) * r for th, r in zip(angles, reading)])
	dys = np.array([np.sin(th) * r for th, r in zip(angles, reading)])
	if plotted is None:
		plotted = []
		for dx, dy in zip(dxs, dys):
			plotted.append(plt.plot([p[0], p[0] + dx], [p[1], p[1] + dy], 'C1')[0])
	else:
		for pl, dx, dy in zip(plotted, dxs, dys):
			pl.set_data([p[0], p[0] + dx], [p[1], p[1] + dy])
	return plotted

def demo_lidar():
	global mx, my
	mx, my = None, None
	def mouse_move(event):
		global mx, my
		mx, my = event.xdata, event.ydata

	rbf_env = RBFEnvironment()
	sim = PhysicsSimulator(rbf_env)
	x, y = -1, -1
	sim.reset([x, y])

	plt.figure()
	plt.ion()
	plt.connect('motion_notify_event', mouse_move)
	plt.imshow(rbf_env.env_img, extent=[-1.2, 1.2, -1.2, 1.2], origin='lower', cmap='coolwarm')

	# # plot gamma grad direction
	for x in np.linspace(-1.1, 1.1, 25):
		for y in np.linspace(-1.1, 1.1, 25):
			grad = rbf_env.gamma_grad([x, y])
			grad = grad / max(1, np.linalg.norm(grad)) * 0.05
			plt.arrow(x, y, grad[0], grad[1], head_length=0.05, head_width=0.02)

	
	plt.axis([-1, 1, -1, 1])
	plt.gca().set_aspect('equal')
	plt.show()

	

	agent_plot, = plt.plot([-1], [-1], 'C2o')
	reading = rbf_env.lidar(sim.agent_pos())
	plotted = plot_lidar([x, y], reading)

	while True:
		if mx is not None and my is not None:
			d = np.array([mx - x, my - y])
			d = d / np.linalg.norm(d) * 0.03
			sim.step(d)
			x, y = sim.agent_pos()
			agent_plot.set_data([x], [y])
			reading = rbf_env.lidar([x, y])
			plotted = plot_lidar([x, y], reading, plotted)
		plt.pause(0.1)

if __name__ == '__main__':
	demo_lidar()
