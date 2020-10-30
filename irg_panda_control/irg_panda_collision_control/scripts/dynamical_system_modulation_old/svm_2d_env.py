
import numpy as np
from sklearn.svm import SVR, SVC
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.interpolate import RectBivariateSpline
from collections import deque
from sklearn.cluster import DBSCAN
from skimage import measure

import cv2 as cv
from PIL import Image

class SVMEnvironment():
	'''
	a 2D environment based on SVM
	support Gamma function query (returns a value greater than 1 in free space, 1 on obstacle boundary, and -1 in obstacle)
	and its gradient (pointing "outward" toward free space)
	also supports lidar sensor simulation.
	'''
	def __init__(self, svm=None):

		if svm == None:
			nobs = 10
			nfree = 40
			np.random.seed(0)
			obs = np.random.random((nobs, 2)) - 0.5
			print("obstacles", obs)
			np.random.seed(0)
			free = np.random.random((nfree, 2)) * 2 - 1
			print("free", free)

			# define SVM
			svm = SVC(gamma=10, C=100000)
			X = np.vstack((obs, free))
			y = [0] * nobs + [1] * nfree
			svm.fit(X, y)

		self.svm = svm

		def gamma(p):
			p = np.array(p)
			g = (svm.decision_function(p.reshape(1, 2)) + 1)[0]
			return g
		self.gamma = gamma
		def gamma_grad(p, normalize=True):
			p = np.array(p).reshape(1, 2)
			sv = self.svm.support_vectors_
			Ks = rbf_kernel(p, sv, gamma=self.svm.gamma).reshape(-1, 1)
			g_grad = - (2 * self.svm.gamma * (p - sv) * Ks * self.svm.dual_coef_.reshape(-1, 1)).sum(axis=0)
			return g_grad
		self.gamma_grad = gamma_grad
		def gamma_batch(p):
			p = np.array(p)
			assert len(p.shape) == 2 and p.shape[1] == 2
			g = svm.decision_function(p) + 1
			return g

		self.gamma_batch = gamma_batch

		x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 200), np.linspace(-1.0, 1.0, 200))
		x_grid = x_grid.flatten()
		y_grid = y_grid.flatten()
		xy_grid = np.stack((x_grid, y_grid), axis=1)
		gamma_grid = self.gamma_batch(xy_grid)
		pred = (gamma_grid > 1).astype('uint8')
		# use morphology library to extract boundary points
		kernel = np.ones((9),np.uint8)
		erosion = cv.dilate(pred, kernel, iterations = 1).reshape(40000)
		boundary_points = np.nonzero(pred - erosion)
		self.boundary_points = xy_grid[boundary_points]
		print ("boundary_points", self.boundary_points)
		self.contours = measure.find_contours(pred.reshape(200,200), 0.8)
		self.num_obstacles = len(self.contours)

		print ("number of obstacles", self.num_obstacles)
		self.center = None

	def __call__(self, p):
		return self.gamma(p)
	def grad(self, p):
		return self.gamma_grad(p)
	def draw(self):
		# plot gamma value
		x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 200), np.linspace(-1.0, 1.0, 200))
		x_grid = x_grid.flatten()
		y_grid = y_grid.flatten()
		xy_grid = np.stack((x_grid, y_grid), axis=1)
		pred = (self.gamma_batch(xy_grid) > 1).astype('int')
		pred = np.array(pred).reshape(200, 200)# [:, ::-1]
		plt.plot(self.boundary_points.T[0,:], self.boundary_points.T[1,:], 'go')

		plt.imshow(pred, extent=[-1.0, 1.0, -1.0, 1.0], origin='lower', cmap='coolwarm')
	def nearest_boundary_pt(self,p):
		dist = np.sum((self.boundary_points - p)**2, axis=1)
		point = self.boundary_points[np.argmin(dist)]
		return point


	def lidar_sensor(self, p, theta, num_readings=100, dist_incr=0.001, max_range=1):
		angles = np.linspace(0, 2 * np.pi, num_readings + 1)[:-1] + theta
		ranges = np.arange(0, max_range, dist_incr)
		dxss = np.array([[np.cos(th) * r for r in ranges] for th in angles])
		dyss = np.array([[np.sin(th) * r for r in ranges] for th in angles])
		pss = np.stack((p[0] + dxss, p[1] + dyss), axis=-1).reshape(-1, 2)
		is_free = self.svm.predict(pss).reshape(num_readings, len(ranges))
		idxs = np.argwhere(is_free==0)
		reading = {th_idx: max_range for th_idx in range(num_readings)}
		for th_idx, incr in idxs:
			if reading[th_idx] > incr * dist_incr:
				reading[th_idx] = incr * dist_incr
		return np.array([reading[th_idx] for th_idx in range(num_readings)])

	def get_env_image(self, resolution, xlow=-1, xhigh=1, ylow=-1, yhigh=1):
		xs = np.linspace(xlow, xhigh, resolution)
		ys = np.linspace(ylow, yhigh, resolution)
		x_grid, y_grid = np.meshgrid(xs, ys)
		x_grid = x_grid.flatten()
		y_grid = y_grid.flatten()
		xy_grid = np.stack((x_grid, y_grid), axis=1)
		env_img = (self.gamma_batch(xy_grid) > 1).astype('int').reshape(resolution, resolution)
		return env_img

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

class SVM2dXYEnv():
	'''
	a 2D environment defined by SVM obstacles with a Gym-style interface
	the control is X-Y control. i.e. a = [dx, dy] clamped to [-dxy_limit, dxy_limit]
	the state space is [xy of agent, xy of target, xy of relative direction, and lidar readings]
	reward at every step is - distance - 0.1 * abs(control).sum()
	'''
	def __init__(self, svm_nobs=10, svm_nfree=40, lidar_num_readings=16, lidar_dist_incr=0.005, lidar_max_range=0.5, 
				 dxy_limit=0.03, time_limit=100, enable_lidar=True, img_state=False, img_res=100, boundary_penalty=False):
		self.svm_nobs = svm_nobs
		self.svm_nfree = svm_nfree
		self.lidar_num_readings = lidar_num_readings
		self.lidar_dist_incr = lidar_dist_incr
		self.lidar_max_range = lidar_max_range
		self.dxy_limit = dxy_limit
		self.time_limit = time_limit
		self.enable_lidar = enable_lidar
		if enable_lidar:
			self.state_dim = lidar_num_readings + 6
		else:
			self.state_dim = 6
		self.action_dim = 2
		self.img_state = img_state
		self.boundary_penalty = boundary_penalty
		self.img_res = img_res

	def get_xy_grid(self, downfactor=1):
		res = int(self.img_res / downfactor)
		x_grid, y_grid = np.meshgrid(np.linspace(-1.2, 1.2, res), np.linspace(-1.2, 1.2, res))
		x_grid = x_grid.flatten()
		y_grid = y_grid.flatten()
		xy_grid = np.stack((x_grid, y_grid), axis=1)
		return xy_grid

	def get_env_img(self):
		xy_grid = self.get_xy_grid()
		env_img = (self.env.gamma_batch(xy_grid) > 1).astype('int')
		env_img = env_img.reshape(self.img_res, self.img_res)
		return env_img

	def get_agent_img(self):
		x, y = self.sim.agent_pos()
		x_channel = np.ones((self.img_res, self.img_res)) * x
		y_channel = np.ones((self.img_res, self.img_res)) * y
		zs = np.stack((x_channel, y_channel), axis=0)
		# res = int(self.img_res / 4)
		# xy_grid = self.get_xy_grid(downfactor=4)
		# zs = multivariate_normal.pdf(xy_grid, mean=self.sim.agent_pos(), cov=0.05**2) / 60
		# zs = zs.reshape(res, res)
		# img = Image.fromarray(zs)
		# img = img.resize((self.img_res, self.img_res))
		# zs = np.array(img)
		return zs

	def get_target_img(self):
		x, y = self.target_pos
		x_channel = np.ones((self.img_res, self.img_res)) * x
		y_channel = np.ones((self.img_res, self.img_res)) * y
		zs = np.stack((x_channel, y_channel), axis=0)
		# res = int(self.img_res / 4)
		# xy_grid = self.get_xy_grid(downfactor=4)
		# zs = multivariate_normal.pdf(xy_grid, mean=self.target_pos, cov=0.05**2) / 60
		# zs = zs.reshape(res, res)
		# img = Image.fromarray(zs)
		# img = img.resize((self.img_res, self.img_res))
		# zs = np.array(img)
		return zs

	def rand_position(self):
		return np.random.uniform(low=-1, high=1, size=2)

	def reset(self):
		obs = np.random.uniform(low=-0.5, high=0.5, size=(self.svm_nobs, 2))
		free = np.random.random((self.svm_nfree, 2)) * 2 - 1
		X = np.vstack((obs, free))
		y = [0] * self.svm_nobs + [1] * self.svm_nfree
		svm = SVC(gamma=10, C=100000)
		svm.fit(X, y)
		self.env = SVMEnvironment(svm)
		self.sim = PhysicsSimulator(self.env)
		while True:
			agent_pos = self.rand_position()
			if self.sim.is_free(*agent_pos):
				break
		self.sim.reset(agent_pos)
		while True:
			target_pos = self.rand_position()
			if self.sim.is_free(*target_pos):
				break
		self.target_pos = target_pos
		self.t = 0
		if self.img_state:
			self.env_img = self.get_env_img()
		return self.s()

	def step(self, a):
		assert self.t < self.time_limit, 'time limit already exceeded'
		self.t += 1
		a = a * self.dxy_limit
		a_clip = np.clip(a, -self.dxy_limit, self.dxy_limit)
		self.sim.step(a_clip, return_collide=True)
		if self.t == self.time_limit:
			done = True
		else:
			done = False
		r = - np.linalg.norm(self.target_pos - self.sim.agent_pos()) #  - 0.1 * abs(np.array(a)).sum()
		if self.boundary_penalty:
			x, y = self.sim.agent_pos()
			if x < -1.2 or x > 1.2 or y < -1.2 or y > 1.2:
				r = r - 10
		return self.s(), r, done, None

	def s(self):
		state = np.concatenate((self.sim.agent_pos(), self.target_pos, self.target_pos - self.sim.agent_pos()))
		if self.enable_lidar:
			lidar_reading = self.env.lidar_sensor(self.sim.agent_pos(), theta=0, 
				num_readings=self.lidar_num_readings, dist_incr=self.lidar_dist_incr, max_range=self.lidar_max_range)
			state = np.concatenate((state, lidar_reading))
		if not self.img_state:
			return state
		else:
			img = np.concatenate((self.env_img.reshape(1, self.img_res, self.img_res), self.get_agent_img(), self.get_target_img()), axis=0)
			# img = np.stack((self.env_img, self.get_agent_img(), self.get_target_img()), axis=0)
			return img, state

class StackedEnvWrapper():
	def __init__(self, env, num_stack=4):
		self.env = env
		self.num_stack = num_stack
		self.state_dim = self.env.state_dim * num_stack
		self.action_dim = self.env.action_dim
	
	def reset(self):
		step_s = self.env.reset()
		self.states = [step_s] * self.num_stack
		return self.s()

	def step(self, a):
		step_s, r, done, info = self.env.step(a)
		del self.states[0]
		self.states.append(step_s)
		return self.s(), r, done, info

	def s(self):
		return np.concatenate(self.states)

class SVM2dFixedXYLidarEnv():
	def __init__(self, svm_nobs=10, svm_nfree=40, dxy_limit=0.03, time_limit=100, img_res=150, 
				 lidar_n_rays=8, lidar_max_range=1, lidar_res=0.002):
		self.svm_nobs = svm_nobs
		self.svm_nfree = svm_nfree
		self.dxy_limit = dxy_limit
		self.time_limit = time_limit
		self.state_dim = 10
		self.action_dim = 2
		self.img_res = img_res
		self.target_pos = np.array([1, 1])
		self.init_agent_pos = np.array([-1, -1])
		self.lidar_n_rays = lidar_n_rays
		self.lidar_max_range = lidar_max_range
		self.lidar_res = lidar_res
		dists = np.arange(lidar_res, lidar_max_range, lidar_res)
		thetas = np.linspace(0, 2 * np.pi, lidar_n_rays + 1)[:-1]
		self.lidar_xs = np.array([dists * np.cos(th) for th in thetas]).flatten()
		self.lidar_ys = np.array([dists * np.sin(th) for th in thetas]).flatten()

	def lidar(self, pos=None):
		if pos is None:
			x, y = self.sim.agent_pos()
		else:
			x, y = pos
		xs = self.lidar_xs + x
		ys = self.lidar_ys + y
		is_free = (self.interpolator.ev(ys, xs).reshape(self.lidar_n_rays, -1) > 0.5).astype('int')
		idxs = np.argwhere(is_free==0)
		reading = {th_idx: self.lidar_max_range for th_idx in range(self.lidar_n_rays)}
		for th_idx, incr in idxs:
			if reading[th_idx] > (incr + 1) * self.lidar_res:
				reading[th_idx] = (incr + 1) * self.lidar_res
		return np.array([reading[th_idx] for th_idx in range(self.lidar_n_rays)])

	def rand_svm(self):
		while True:
			obs = np.random.uniform(low=-0.5, high=0.5, size=(self.svm_nobs, 2))
			free = np.random.uniform(low=-1, high=1, size=(self.svm_nfree, 2))
			self.obs = obs
			self.free = free
			X = np.vstack((obs, free))
			y = [0] * self.svm_nobs + [1] * self.svm_nfree
			svm = SVC(gamma=10, C=100000)
			svm.fit(X, y)
			if svm.predict([self.init_agent_pos, self.target_pos]).sum() == 2:
				break
		return svm

	def construct_interpolator(self):
		xs = np.linspace(-1.2, 1.2, self.img_res)
		ys = np.linspace(-1.2, 1.2, self.img_res)
		x_grid, y_grid = np.meshgrid(xs, ys)
		x_grid = x_grid.flatten()
		y_grid = y_grid.flatten()
		xy_grid = np.stack((x_grid, y_grid), axis=1)
		env_img = (self.env.gamma_batch(xy_grid) > 1).astype('int')
		env_img = env_img.reshape(self.img_res, self.img_res)
		self.interpolator = RectBivariateSpline(x=xs, y=ys, z=env_img)

	def reset(self):
		svm = self.rand_svm()
		self.env = SVMEnvironment(svm)
		self.sim = PhysicsSimulator(self.env)
		self.sim.reset(self.init_agent_pos)
		self.t = 0
		self.construct_interpolator()
		return self.s()

	def reward(self):
		r = - np.linalg.norm(self.target_pos - self.sim.agent_pos()) #  - 0.1 * abs(np.array(a)).sum()
		return r

	def step(self, a):
		assert self.t < self.time_limit, 'time limit already exceeded'
		self.t += 1
		a = a * self.dxy_limit
		a_clip = np.clip(a, -self.dxy_limit, self.dxy_limit)
		self.sim.step(a_clip, return_collide=True)
		return self.s(), self.reward(), self.t == self.time_limit, None

	def s(self):
		return np.concatenate([self.sim.agent_pos(), self.lidar()])

# env = SVM2dFixedXYLidarEnv()
# env.reset()
# # xs = np.linspace(-3, 3, 200)
# # ys = np.linspace(-3, 3, 200)
# # mat = env.interpolator.ev(xs, ys)
# # plt.imshow(mat)
# # plt.colorbar()
# # plt.show()
# quit()

class SVM2dFixedXYEnv():
	'''
	a 2D environment defined by SVM obstacles with a Gym-style interface
	the control is X-Y control. i.e. a = [dx, dy] clamped to [-dxy_limit, dxy_limit]
	the state space is [xy of agent, xy of target, xy of relative direction, and lidar readings]
	reward at every step is - distance - 0.1 * abs(control).sum()
	'''
	def __init__(self, svm_nobs=10, svm_nfree=40, dxy_limit=0.03, time_limit=100, img_state=True, img_res=100, boundary_penalty=False):
		self.svm_nobs = svm_nobs
		self.svm_nfree = svm_nfree
		self.dxy_limit = dxy_limit
		self.time_limit = time_limit
		self.state_dim = 2
		self.action_dim = 2
		self.img_state = img_state
		self.boundary_penalty = boundary_penalty
		self.img_res = img_res

	def get_xy_grid(self, downfactor=1):
		res = int(self.img_res / downfactor)
		x_grid, y_grid = np.meshgrid(np.linspace(-1.2, 1.2, res), np.linspace(-1.2, 1.2, res))
		x_grid = x_grid.flatten()
		y_grid = y_grid.flatten()
		xy_grid = np.stack((x_grid, y_grid), axis=1)
		return xy_grid

	def get_env_img(self):
		xy_grid = self.get_xy_grid()
		env_img = (self.env.gamma_batch(xy_grid) > 1).astype('int')
		env_img = env_img.reshape(self.img_res, self.img_res)
		return env_img

	def get_agent_img(self):
		x, y = self.sim.agent_pos()
		x_channel = np.ones((self.img_res, self.img_res)) * x
		y_channel = np.ones((self.img_res, self.img_res)) * y
		zs = np.stack((x_channel, y_channel), axis=0)
		return zs

	def rand_svm(self):
		obs = np.random.uniform(low=-0.5, high=0.5, size=(self.svm_nobs, 2))
		free = np.random.random((self.svm_nfree, 2)) * 2 - 1
		X = np.vstack((obs, free))
		y = [0] * self.svm_nobs + [1] * self.svm_nfree
		svm = SVC(gamma=10, C=100000)
		svm.fit(X, y)
		return svm

	def reset(self):
		while True:
			svm = self.rand_svm()
			# if svm.predict([[-1,-1], [1, 1]]).sum() == 2:
			if svm.predict([[0.7, 0.7], [1, 1]]).sum() == 2:
				break
		self.env = SVMEnvironment(svm)
		self.sim = PhysicsSimulator(self.env)
		self.sim.reset([0.7, 0.7])
		self.target_pos = np.array([1, 1])
		self.t = 0
		if self.img_state:
			self.env_img = self.get_env_img()
		return self.s()

	def reward(self):
		r = - np.linalg.norm(self.target_pos - self.sim.agent_pos()) #  - 0.1 * abs(np.array(a)).sum()
		if self.boundary_penalty:
			x, y = self.sim.agent_pos()
			if x < -1.2 or x > 1.2 or y < -1.2 or y > 1.2:
				r = r - 10
		return r

	def step(self, a):
		assert self.t < self.time_limit, 'time limit already exceeded'
		self.t += 1
		a = a * self.dxy_limit
		a_clip = np.clip(a, -self.dxy_limit, self.dxy_limit)
		self.sim.step(a_clip, return_collide=True)
		return self.s(), self.reward(), self.t == self.time_limit, None

	def s(self):
		img = np.concatenate((self.env_img.reshape(1, self.img_res, self.img_res), self.get_agent_img()), axis=0)
		return img, self.sim.agent_pos()

class SVM2dRThetaEnv():
	pass

def plot_lidar(p, reading, theta, plotted=None):
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

if __name__ == '__main__':
	import time

	mx, my = None, None
	def mouse_move(event):
		global mx, my
		mx, my = event.xdata, event.ydata

	# nobs = 10
	# nfree = 40
	# obs = np.random.random((nobs, 2)) - 0.5
	# free = np.random.random((nfree, 2)) * 2 - 1
	# svm = SVC(gamma=10, C=100000)
	# X = np.vstack((obs, free))
	# y = [0] * nobs + [1] * nfree
	# svm.fit(X, y)
	# e = SVMEnvironment(svm)
	# sim = PhysicsSimulator(e)

	gym_env = SVM2dFixedXYLidarEnv()
	s = gym_env.reset()
	e = gym_env.env
	sim = gym_env.sim

	plt.figure()
	plt.ion()
	plt.connect('motion_notify_event', mouse_move)

	# plot gamma value
	x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 150), np.linspace(-1.0, 1.0, 150))
	x_grid = x_grid.flatten()
	y_grid = y_grid.flatten()
	xy_grid = np.stack((x_grid, y_grid), axis=1)
	pred = (e.gamma_batch(xy_grid) > 1).astype('int')
	# plt.scatter(x_grid, y_grid, c=[f'C{p}' for p in pred], marker='.', s=5)
	pred = np.array(pred).reshape(150, 150)# [:, ::-1]
	plt.imshow(pred, extent=[-1.0, 1.0, -1.0, 1.0], origin='lower', cmap='coolwarm')

	# # plot gamma grad direction
	for x in np.linspace(-1.1, 1.1, 25):
		for y in np.linspace(-1.1, 1.1, 25):
			grad = e.gamma_grad([x, y])
			grad = grad / max(1, np.linalg.norm(grad)) * 0.05
			plt.arrow(x, y, grad[0], grad[1], head_length=0.05, head_width=0.02)

	plt.axis([-1, 1, -1, 1])
	plt.gca().set_aspect('equal')
	plt.show()
	
	line, = plt.plot([s[0]], [s[1]], 'C2o')
	reading = gym_env.lidar()
	plotted = plot_lidar(s, reading, 0)
	plt.plot([1], [1], 'C1*')

	global_start = time.time()
	total_sim = 0
	while True:
		if mx is not None and my is not None:
			d = np.array([mx - s[0], my - s[1]])
			d = d / np.linalg.norm(d)
			s, r, done, _ = gym_env.step(d)
			line.set_data([s[0]], [s[1]])
			reading = gym_env.lidar()
			plotted = plot_lidar(s, reading, 0, plotted)
		plt.pause(0.1)


# class Free2dXYEnv():
# 	def __init__(self, svm_nobs=10, svm_nfree=40, lidar_num_readings=16, lidar_dist_incr=0.005, lidar_max_range=0.5, 
# 				 dxy_limit=0.03, time_limit=100, enable_lidar=True, img_state=False, img_res=100, boundary_penalty=False):
# 		self.dxy_limit = dxy_limit
# 		self.time_limit = time_limit
# 		self.enable_lidar = enable_lidar
# 		self.state_dim = 6
# 		self.action_dim = 2
# 		self.img_state = img_state
# 		self.boundary_penalty = boundary_penalty
# 		self.img_res = img_res

# 	def get_env_img(self):
# 		return np.ones((self.img_res, self.img_res))

# 	def get_agent_img(self):
# 		x, y = self.agent_pos
# 		x_channel = np.ones((self.img_res, self.img_res)) * x
# 		y_channel = np.ones((self.img_res, self.img_res)) * y
# 		zs = np.stack((x_channel, y_channel), axis=0)
# 		return zs

# 	def get_target_img(self):
# 		x, y = self.target_pos
# 		x_channel = np.ones((self.img_res, self.img_res)) * x
# 		y_channel = np.ones((self.img_res, self.img_res)) * y
# 		zs = np.stack((x_channel, y_channel), axis=0)
# 		return zs

# 	def rand_position(self):
# 		return np.random.uniform(low=-1, high=1, size=2)

# 	def reset(self):
# 		self.agent_pos = self.rand_position()
# 		self.target_pos = self.rand_position()
# 		self.t = 0
# 		if self.img_state:
# 			self.env_img = self.get_env_img()
# 		return self.s()

# 	def step(self, a):
# 		assert self.t < self.time_limit, 'time limit already exceeded'
# 		self.t += 1
# 		a = a * self.dxy_limit
# 		a_clip = np.clip(a, -self.dxy_limit, self.dxy_limit)
# 		self.agent_pos = self.agent_pos + a_clip
# 		done = (self.t == self.time_limit)
# 		r = - np.linalg.norm(self.target_pos - self.agent_pos)
# 		if self.boundary_penalty:
# 			x, y = self.agent_pos
# 			if x < -1.2 or x > 1.2 or y < -1.2 or y > 1.2:
# 				r = r - 10
# 		return self.s(), r, done, None

# 	def s(self):
# 		state = np.concatenate((self.agent_pos, self.target_pos, self.target_pos - self.agent_pos))
# 		if not self.img_state:
# 			return state
# 		else:
# 			img = np.concatenate((self.env_img.reshape(1, self.img_res, self.img_res), self.get_agent_img(), self.get_target_img()), axis=0)
# 			return img, state

