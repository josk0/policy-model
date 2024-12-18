from mesa import Agent
import numpy as np

class PolicyAgent(Agent):
    """Individual Agent definition and its properties/interaction methods."""

    def __init__(
        self, 
        # unique_id, 
        model, 
        marginalized,
        privileged,
    ):
        super().__init__(model)

        self.opinion: int = 0  # Initialize opinion
        self.impact: int = 0  # Initialize impact
        self.marginalized = marginalized
        self.privileged = privileged
        self.num_connections: int = 0
        
        #if len(self.model.grid.get_neighbors(self.pos, include_center=False)) > 0: self.num_connections = len(self.model.grid.get_neighbors(self.pos, include_center=False))

    def step(self):
        # Gather observed impacts
        observed_impacts = [
            self.model.grid.get_cell_list_contents([neighbor.pos])[0].impact
            for neighbor in self.model.grid.get_neighbors(self.pos, include_center=True)
        ]
        observed_sum = sum(observed_impacts)

        self.num_connections = len(self.model.grid.get_neighbors(self.pos, include_center=False))

        # Logistic function to update opinion
        self.opinion = 2 / (1 + np.exp(-observed_sum)) - 1