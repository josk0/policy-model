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
        steps: int = 10,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.privileged_fraction = privileged_fraction
        self.marginalized_fraction = marginalized_fraction
        self.steps = steps

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
                "Net Policy Support": net_opinion,
                "Net Social Benefit": net_impact,
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
        self.apply_policy()

    def apply_policy(self):
        affected_nodes = np.random.choice(
            self.graph.nodes(),
            size=int(self.num_agents * 0.3),
            replace=False,
        )
        for node in affected_nodes:
            agent = self.grid.get_cell_list_contents([node])[0]
            if agent.marginalized:
                agent.impact = 1
            else:
                agent.impact = -1

    def step(self):
        self.agents.shuffle_do("step")

        # Dynamic edge formation
        for agent in self.agents:
            if abs(agent.opinion) >= 0.3:
                # Get neighbor IDs using agent.pos as key
                neighbor_ids = list(self.grid.G.neighbors(agent.pos))  
                
                aligned_nodes = [
                    neighbor_id
                    for neighbor_id in neighbor_ids
                    if self.grid.get_cell_list_contents([neighbor_id])[0].impact
                    == np.sign(agent.opinion)
                ]
                
                unconnected_nodes = set(neighbor_ids) - set(aligned_nodes)
                
                for node in unconnected_nodes:
                    for aligned_node in aligned_nodes:
                        # Ensure both nodes are in the grid before adding edge
                        if node in self.grid.G and aligned_node in self.grid.G:
                            self.grid.G.add_edge(node, aligned_node)

        # collect data
        self.datacollector.collect(self)
