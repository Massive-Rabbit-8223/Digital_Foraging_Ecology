import numpy as np

SEED = 42

rng = np.random.default_rng(SEED)

def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1+np.exp(-x))

class Agent:
    def __init__(self, id: int, position: np.ndarray, energy: float, health: float, 
                 marker: str, weights: list, act_fn: callable, out_act_fn: callable):
        self.id = id
        self.position = position
        self.energy = energy
        self.health = health
        self.marker = marker
        self.weights = weights
        self.act_fn = act_fn
        self.out_act_fn = out_act_fn

    def get_action(self, observation):
        x = observation.copy()
        for w in self.weights[:-1]:
            x = self.act_fn(w.T@x)
        logits = self.weights[-1].T@x
        action_prob = self.out_act_fn(logits).flatten()
        print(action_prob)
        action = rng.choice(logits.size, p=action_prob)
        
        return action

    def update_position(self):
        pass

class Resource:
    def __init__(self, id: int, position: np.ndarray, energy: float, regrowth_rate: int, marker: str):
        self.id = id
        self.position = position
        self.energy = energy
        self.regrowth_rate = regrowth_rate
        self.marker = marker

class World:
    def __init__(self, size: tuple, agents: list[Agent], resources: list[Resource]):
        self.agents = agents
        self.resources = resources

        self.world = np.zeros(size)
        for objects in self.agents+self.resources:
            self.world[tuple(objects.position)] = objects.id

    def render_world(self) -> None:
        string = "." * self.world.size
        for object in self.agents+self.resources:
            index = object.position
            index_flattened = index[0]*self.world.shape[0] + index[1]
            string = string[:index_flattened] + object.marker + string[index_flattened+1:]

        for index_lb in [self.world.shape[1]*row for row in range(self.world.shape[0]-1,0,-1)]:
            string = string[:index_lb] + "\n" + string[index_lb:]

        print("\n"+string+"\n")

    def get_obs(self, position: np.ndarray) -> np.ndarray:
        obs = np.zeros((4,1))
        r, c = position[0], position[1]
        max_r, max_c = self.world.size[0]-1, self.world.size[1]-1
        min_r, min_c = 0, 0

        obs[0] = self.world[r-1, c] if r-1 >= min_r else self.world[max_r, c]
        obs[1] = self.world[r, c+1] if c+1 <= max_c else self.world[r, min_c]
        obs[2] = self.world[r+1, c] if r+1 <= max_r else self.world[min_r, c]
        obs[3] = self.world[r, c-1] if c-1 >= min_c else self.world[r, max_c]

        return obs

    def step(self) -> None:
        for index, a in enumerate(self.agents):
            obs = self.get_obs(a.position)
            action = a.get_action(obs)


if __name__ == "__main__":

    a1 = Agent(
        id=1,
        position=np.array([4,4]),
        energy=10.,
        health=10.,
        marker="A",
        weights=[
            (rng.random((4,10))*2)-1,
            (rng.random((10,4))*2)-1
        ],
        act_fn=sigmoid,
        out_act_fn=softmax
    )

    a2 = Agent(
        id=2,
        position=np.array([5,7]),
        energy=10.,
        health=10.,
        marker="B",
        weights=[
            (rng.random((4,10))*2)-1,
            (rng.random((10,4))*2)-1
        ],
        act_fn=sigmoid,
        out_act_fn=softmax
    )

    r1 = Resource(id=2,position=np.array([2,3]),energy=1,regrowth_rate=10,marker="+")
    r2 = Resource(id=2,position=np.array([5,5]),energy=1,regrowth_rate=10,marker="*")

    world = World(
        size=(10,10),
        agents=[a1, a2],
        resources=[r1, r2]
    )

    world.render_world()

    #obs = np.array([[-1],[-1],[2],[-1]])
    #action = a.get_action(obs)
    #print(action)