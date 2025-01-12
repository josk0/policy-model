from mesa import Agent
import numpy as np

class PolicyAgent(Agent):
    """A citizen agent in the policy model."""

    def __init__(
        self, 
        #unique_id, 
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
        
    def count_connections(self) -> int:
        self.num_connections = len(self.model.grid.get_neighbors(self.pos, include_center=False)) 

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
        # Gather observed impacts
        observed_impacts = [n.impact for n in neighbors]
       
        # Logistic function to update opinion
        self.opinion = 2 / (1 + np.exp(-sum(observed_impacts))) - 1

        # Dynamic edge formation linking one aligend-affected with one unaligned neighbor
        if abs(self.opinion) >= self.model.trigger_level:
            
            unaligned_neighbors = [n for n in neighbors if np.sign(n.opinion) != np.sign(self.opinion)]
            affected_neighbors = [n for n in neighbors if np.sign(n.impact) == np.sign(self.opinion)]

            if unaligned_neighbors and affected_neighbors:  
                picked_unaligned_neighbor = np.random.choice(unaligned_neighbors)
                # Remove the picked unaligned neighbor from affected_neighbors if element in these
                affected_neighbors = list(set(affected_neighbors) - {picked_unaligned_neighbor})
                if affected_neighbors:  # Only proceed if we have valid targets
                    picked_affected_neighbor = np.random.choice(affected_neighbors)
                    self.model.grid.G.add_edge(picked_unaligned_neighbor.pos, picked_affected_neighbor.pos)
            
        self.count_connections()
        