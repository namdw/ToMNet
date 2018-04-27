
import numpy as np
import matplotlib.pyplot as plt
import time
from Agents import RandomAgent

print('hi')

class GridWorld():
	def __init__(self, height, width, num_goal):
		self.height = height
		self.width = width
		self.num_goal = num_goal

		self.walls = np.zeros([self.height, self.width])
		self.goals = np.zeros([self.num_goal, self.height, self.width])
		self.agent = None

		self.figure = None
		self.ax = None

	def generateWalls(self):
		self.walls = np.zeros([self.height, self.width])
		for r in range(self.height):
			if r==0 or r==self.height-1:
				self.walls[r,:] += 1
			else:
				self.walls[r,0] += 1
				self.walls[r,-1] += 1

	def getWalls(self):
		return self.walls

	def generateGoals(self):
		self.goals = np.zeros([self.num_goal, self.height, self.width])
		i, j = np.where(self.walls==0)
		empty_cells = np.random.permutation(len(i))
		for g in range(self.num_goal):
			self.goals[g, i[empty_cells[g]], j[empty_cells[g]]] = 1

	def getGoals(self):
		return self.goals

	def showImage(self):
		imageMat = self.walls
		for i in range(self.num_goal):
			imageMat += (i+2) * self.goals[i, :, :]
		self.figure = plt.figure()
		self.ax = self.figure.gca()
		self.figure.show()
		self.ax.matshow(imageMat)

	def updateImage(self):
		imageMat = self.walls
		for i in range(self.num_goal):
			imageMat += (i+2) * self.goals[i, :, :]
		self.ax.matshow(imageMat)
		self.figure.canvas.draw()

	def addAgent(self, agent):
		self.agent = agent

	def step(self):
		new_y, new_x = self.agent.move()
		print(new_y, new_x)

	def getState(self):
		agent_mat = np.zeros([1, self.height, self.width])
		r, c = self.agent.get_position()
		agent_mat[0,r,c] = 1
		last_action = self.agent.get_last_action()
		action_mat = np.zeros([5, self.height, self.width])
		action_mat[last_action, :, :] = np.ones([self.height, self.width])
		output = np.concatenate((np.array([self.walls]), self.goals), axis=0)
		output = np.concatenate((output, agent_mat), axis=0)
		output = np.concatenate((output, action_mat), axis=0)

		return output















