import numpy as np
from dataclasses import dataclass

@dataclass
class Body:
    position: np.ndarray
    energy: float
    health: float
    age: int = 0
    alive: bool = True


class Agent:
    def __init__(self, genome):
        self.genome = genome
        self.brain = build_brain(genome)
        self.memory = None

    def act(self, observation):
        return self.brain(observation)


class World:
    def __init__(self):
        self.bodies = {}
        self.agents = {}
        self.food = []

    def step(self):

        # 1. Snapshot / perception
        observations = {
            i: self.observe(i)
            for i in self.agents
        }

        # 2. Decisions — no world mutation
        actions = {
            i: agent.act(observations[i])
            for i, agent in self.agents.items()
        }

        # 3. Resolve world dynamics
        self.resolve_actions(actions)

        # 4. Environment dynamics
        self.apply_metabolism()
        self.grow_food()

        # 5. Population dynamics
        self.resolve_births()
        self.resolve_deaths()

if __name__ == "__main__":
    pass