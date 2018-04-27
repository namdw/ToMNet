import numpy as np

class RandomAgent():
	def __init__(self, world):
		self.y = None
		self.x = None
		self.goal = None
		self.actions = 5 # 0:stay, 1:up, 2:down, 3:left, 4:right
		self.action_pool = np.array([[0,0], [1,0], [-1,0], [0,-1], [0,1]])
		self.last_action = 0

		self.world = world
		self.select_position()

	def select_action(self):
		return np.random.randint(self.actions)

	def get_position(self):
		return self.y, self.x

	def get_last_action(self):
		return self.last_action

	def move(self):
		action = self.select_action()
		self.last_action = action
		new_pos = np.array([self.y, self.x]) + self.action_pool[action]
		if self.world.getWalls()[new_pos[0], new_pos[1]]==1:
			new_pos = np.array([self.y, self.x])
		self.y = new_pos[0]
		self.x = new_pos[1]
		return new_pos[0], new_pos[1]

	def select_position(self):
		i, j = np.where(self.world.getWalls()+np.sum(self.world.getGoals(),axis=0)==0)
		empty_cells = np.random.permutation(len(i))
		self.y = i[empty_cells[0]]
		self.x = j[empty_cells[0]]


class AlgorithmAgent(RandomAgent):
	def __init__(self, world):
		super(AlgorithmAgent, self).__init__()
		self.y = None
		self.x = None
		self.goal = None
		self.actions = 5
		self.last_action = None

		self.world = self.world

	def select_action(self):
		return np.random.randint(self.actions)

class DeepRLAgent(RandomAgent):
	def __init__(self, world):
		super(DeepRLAgent, self).__init__()
		self.y = None
		self.x = None
		self.goal = None
		self.actions = 5
		self.last_action = None

		self.world = world

	def select_action(self):
		return np.random.randin(self.actions)