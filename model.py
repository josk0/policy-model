import mesa
from mesa import Model
import networkx as nx
import numpy as np

from agents import PolicyAgent

def sum_attribute(model, attribute_name):
    return np.mean([getattr(agent, attribute_name) for agent in model.agents])

def net_opinion(model):
    return sum_attribute(model, "opinion")

def net_impact(model):
    return sum_attribute(model, "impact")

class PolicyModel(Model):
    """A opinion dynamic network model."""

    def __init__(
        self,
        num_agents: int = 10,
        privileged_fraction: float = 0.2,
        marginalized_fraction: float = 0.1,
        policy_expansion: float = 0.01,
        trigger_level: float = 0.35,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.privileged_fraction = privileged_fraction
        self.marginalized_fraction = marginalized_fraction
        self.policy_expansion = policy_expansion
        self.trigger_level = trigger_level

        # Create a network and grid
        self.graph = nx.barabasi_albert_graph(
            n=self.num_agents,
            m=2
        )
        self.grid = mesa.space.NetworkGrid(self.graph)

        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)

        # Data collector
        self.datacollector = mesa.DataCollector(
            {
                "Avg Policy Support": net_opinion,
                "Avg Social Benefit": net_impact,
            }
        )

        # Initialize agents
        sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
        num_privileged = int(self.num_agents * self.privileged_fraction)
        privileged_nodes = sorted_nodes[:num_privileged]

        for node in self.graph.nodes():
          a = PolicyAgent(
              self,
              False,
              False,
          )
          a.marginalized = np.random.random() <= self.marginalized_fraction
          #a.privileged = not a.marginalized and np.random.random() <= self.privileged_fraction
          a.privileged = node in privileged_nodes  # Assign privileged based on centrality
          self.grid.place_agent(a, node)

        # Apply initial policy intervention
        self.apply_policy(self.policy_expansion)
    

    def apply_policy(self, policy_expansion):
        eligible_nodes_nm = [
            node for node in self.graph.nodes() 
            if self.grid.get_cell_list_contents([node])[0].impact == 0 and not self.grid.get_cell_list_contents([node])[0].marginalized
        ]
        eligible_nodes_m = [
            node for node in self.graph.nodes() 
            if self.grid.get_cell_list_contents([node])[0].impact == 0 and self.grid.get_cell_list_contents([node])[0].marginalized
        ]
        
        affected_nodes = np.random.choice(
            eligible_nodes_m,
            size=int(self.num_agents * policy_expansion / 2),
            replace=False,
        )

        np.append(affected_nodes, np.random.choice(
            eligible_nodes_m,
            size=int(self.num_agents * policy_expansion / 2),
            replace=False,
        ))
        
        for node in affected_nodes:
            agent = self.grid.get_cell_list_contents([node])[0]
            agent.impact = 1 if agent.marginalized else -1


    def step(self):
        self.agents.shuffle_do("step")

        # Dynamic edge formation
        for agent in self.agents:
            if abs(agent.opinion) >= self.trigger_level:
                # Get neighbor IDs using agent.pos as key
                neighbor_ids = list(self.grid.G.neighbors(agent.pos))  
                
                aligned_neighbors = [
                    neighbor_id
                    for neighbor_id in neighbor_ids
                    if self.grid.get_cell_list_contents([neighbor_id])[0].opinion
                    == np.sign(agent.opinion)
                ]

                affected_neighbors = [
                    neighbor_id
                    for neighbor_id in neighbor_ids
                    if self.grid.get_cell_list_contents([neighbor_id])[0].impact
                    == np.sign(agent.opinion)
                ]

                unaligned_neighbors = list(set(neighbor_ids) - set(aligned_neighbors))
                unaffected_neighbors = list(set(neighbor_ids) - set(affected_neighbors))

                if unaligned_neighbors and affected_neighbors:
                    party_unaligned = np.random.choice(unaligned_neighbors)
                    party_affected = np.random.choice(affected_neighbors)
                    self.grid.G.add_edge(party_unaligned, party_affected)
        
        # is support is less than benefit, increase policy coverage
        if net_opinion(self) < net_impact(self): self.apply_policy(self.policy_expansion)
        
        # collect data
        self.datacollector.collect(self)
