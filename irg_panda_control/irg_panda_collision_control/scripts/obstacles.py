import matplotlib.pyplot as plt

'''
a collection of different obstacles, represented via gamma function
'''

import torch, math
import numpy as np

class Gamma(object):
	'''an abstract gamma class, can call gamma() and gamma.grad()'''
	def __init__(self):
		self.center = None
		# pt is in the workspace frame
	def __call__(pt):
		'''pt is in the workspace frame'''
		raise NotImplementedError
	def grad(self, pt):
		raise NotImplementedError
	def draw(self):
		'''draw the obstacle with matplotlib artist'''
		raise NotImplementedError

class GammaCircle2D(Gamma):
	def __init__(self, radius, center):
		super(Gamma, self).__init__()
		self.center = center
		self.radius = radius
	def __call__(self, pt):
		return np.linalg.norm(pt - self.center) / self.radius
	def grad(self, pt):
		return (pt - self.center) / (np.linalg.norm(pt - self.center) * self.radius)
	def draw(self):
		plt.gca().add_artist(plt.Circle(self.center, self.radius.item()))

class GammaRectangle2D(Gamma):
	def __init__(self, w, h, center):
		super(GammaRectangle2D, self).__init__()
		self.center = center
		self.w = w
		self.h = h
	def __call__(self, pt):
		x, y = pt - self.center
		angle = np.arctan2(y, x)
		first = np.arctan2(self.h/2, self.w/2)
		second = np.arctan2(self.h/2, -self.w/2)
		third = np.arctan2(-self.h/2, -self.w/2)
		fourth = np.arctan2(-self.h/2, self.w/2)
		if (first < angle < second) or (third < angle < fourth):
			return 2 * np.abs(y) / self.h
		else:
			return 2 * np.abs(x) / self.w
	def grad(self, pt):
		x, y = pt - self.center
		angle = np.arctan2(y, x)
		first = np.arctan2(self.h/2, self.w/2)
		second = np.arctan2(self.h/2, -self.w/2)
		third = np.arctan2(-self.h/2, -self.w/2)
		fourth = np.arctan2(-self.h/2, self.w/2)
		if (first < angle < second) or (third < angle < fourth):
			return np.stack([np.array(0.), np.sign(y)*2/self.h])
		else:
			return np.stack([np.sign(x)*2/self.w, np.array(0.)])
	def draw(self):
		ctr = self.center
		w = self.w.item()
		h = self.h.item()
		plt.gca().add_artist(plt.Rectangle((ctr[0]-w/2, ctr[1]-h/2), w, h))

class GammaCross2D(Gamma):
	def __init__(self, a, b, center):
		super(GammaCross2D, self).__init__()
		self.center = center
		self.a = a
		self.b = b
	def atan2_pos(self, x, y):
		ang = np.arctan2(x, y)
		if ang < 0:
			ang = 2 * math.pi + ang
		return ang
	def __call__(self, pt):
		c = self.a / 2
		d = self.a / 2 + self.b
		one = np.array(1.0)
		angles = [(c, d), (one, one), (d, c), (d, -c), (one, -one), (c, -d), (-c, -d),
				  (-one, -one), (-d, -c), (-d, c), (-one, one), (-c, d)]
		angles = [np.stack(ang) for ang in angles]
		angles = [self.atan2_pos(*x) for x in angles]
		x, y = pt - self.center
		angle = self.atan2_pos(y, x)
		if angles[0] <= angle < angles[1]:
			return y / c
		elif angles[1] <= angle < angles[2]:
			return x / c
		elif angles[2] <= angle < angles[3]:
			return y / d
		elif angles[3] <= angle < angles[4]:
			return - x / c
		elif angles[4] <= angle < angles[5]:
			return y / c
		elif angles[5] <= angle < angles[6]:
			return - x / d
		elif angles[6] <= angle < angles[7]:
			return - y / c
		elif angles[7] <= angle < angles[8]:
			return - x / c
		elif angles[8] <= angle < angles[9]:
			return - y / d
		elif angles[9] <= angle < angles[10]:
			return  x / c
		elif angles[10] <= angle < angles[11]:
			return - y / c
		else:
			return x / d
	def grad(self, pt):
		c = self.a / 2
		d = self.a / 2 + self.b
		one = np.array(1.0)
		angles = [(c, d), (one, one), (d, c), (d, -c), (one, -one), (c, -d), (-c, -d),
				  (-one, -one), (-d, -c), (-d, c), (-one, one), (-c, d)]
		angles = [np.stack(ang) for ang in angles]
		angles = [self.atan2_pos(*x) for x in angles]
		x, y = pt - self.center
		angle = self.atan2_pos(y, x)
		if angles[0] <= angle < angles[1]:
			return np.stack([np.array(0.), 1 / c])
		elif angles[1] <= angle < angles[2]:
			return np.stack([1 / c, np.array(0.)])
		elif angles[2] <= angle < angles[3]:
			return np.stack([np.array(0.), 1 / d])
		elif angles[3] <= angle < angles[4]:
			return np.stack([- 1 / c, np.array(0.)])
		elif angles[4] <= angle < angles[5]:
			return np.stack([np.array(0.), 1 / c])
		elif angles[5] <= angle < angles[6]:
			return np.stack([- 1 / d, np.array(0.)])
		elif angles[6] <= angle < angles[7]:
			return np.stack([np.array(0.), - 1 / c])
		elif angles[7] <= angle < angles[8]:
			return np.stack([- 1 / c, np.array(0.)])
		elif angles[8] <= angle < angles[9]:
			return np.stack([np.array(0.), - 1 / d])
		elif angles[9] <= angle < angles[10]:
			return  np.stack([1 / c, np.array(0.)])
		elif angles[10] <= angle < angles[11]:
			return np.stack([np.array(0.), - 1 / c])
		else:
			return np.stack([1 / d, np.array(0.)])
	def draw(self):
		x, y = self.center
		a = self.a.item()
		b = self.b.item()
		length = a + b * 2
		width = a
		plt.gca().add_artist(plt.Rectangle((x-length/2, y-width/2), length, width))
		plt.gca().add_artist(plt.Rectangle((x-width/2, y-length/2), width, length))
