import mesa
from mesa import Model
import networkx as nx
import numpy as np

from agents import PolicyAgent

def av_attribute(model, attribute_name):
    return np.mean([getattr(agent, attribute_name) for agent in model.agents])

def av_opinion(model):
    return av_attribute(model, "opinion")

def av_impact(model):
    return av_attribute(model, "impact")

class PolicyModel(Model):
    """A opinion dynamic network model."""

    def __init__(
        self,
        num_agents: int = 140,
        privileged_fraction: float = 0.1,
        marginalized_fraction: float = 0.3,
        rel_policy_expansion: float = 0.02,
        trigger_level: float = 0.45,
        policy_reaction: bool = True,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.num_agents = num_agents
        self.privileged_fraction = privileged_fraction
        self.marginalized_fraction = marginalized_fraction
        self.rel_policy_expansion = rel_policy_expansion
        self.trigger_level = trigger_level
        self.policy_reaction = policy_reaction

        # Data collector
        self.datacollector = mesa.DataCollector(
            {
                "Avg Policy Support": av_opinion,
                "Avg Social Benefit": av_impact,
            }
        )

        # Create a network and grid
        self.graph = nx.barabasi_albert_graph(
            n=self.num_agents,
            m=2
        )
        self.grid = mesa.space.NetworkGrid(self.graph)

        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)

        # select privileged nodes (agents will be initialized there accordingly)
        sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)
        num_privileged = max(1, int(self.num_agents * self.privileged_fraction))
        privileged_nodes = sorted_nodes[:num_privileged] # Assign privileged based on centrality
        
        # select nodes for marginalized agents
        non_privileged_nodes = set(self.graph.nodes) - set(privileged_nodes)
        num_marginalized = max(1, int(self.num_agents * self.marginalized_fraction))
        # idea: could "cluster" marginalized nodes together in some way?
        marginalized_nodes = np.random.choice(
            list(non_privileged_nodes),  
            size=min(num_marginalized, len(non_privileged_nodes)),  # Added safety check
            replace=False,
        )

        # Initialize agents
        for node in self.graph.nodes():
          a = PolicyAgent(
              model=self,
              marginalized=False,
              privileged=False,
          )
          a.marginalized = node in marginalized_nodes
          a.privileged = node in privileged_nodes  
          self.grid.place_agent(a, node)

        # Apply initial policy intervention
        self.apply_policy(self.rel_policy_expansion)
    

    def apply_policy(self, rel_policy_expansion):
        
        unaffected_privileged = self.agents.select(lambda a: a.impact == 0 and a.privileged) # agents that are both not marginalized and not (yet) affected by policy
        unaffected_marginalized = self.agents.select(lambda a: a.impact == 0 and a.marginalized) # agents that are marginalized and not (yet) affected by policy
 
        abs_policy_expansion_each_side = max(1, int(round(self.num_agents * rel_policy_expansion / 2))) # Note: the /2 entails that there will be the same number of winners and losers
        newly_affected_agents = []

        # Handle marginalized nodes
        if unaffected_marginalized:
            newly_affected_agents.extend(
                np.random.choice(
                    unaffected_marginalized,
                    size=min(abs_policy_expansion_each_side, len(unaffected_marginalized)),
                    replace=False
                )
            )
        
        # Handle non-marginalized nodes
        if unaffected_privileged:
            newly_affected_agents.extend(
                np.random.choice(
                    unaffected_privileged,
                    size=min(abs_policy_expansion_each_side, len(unaffected_privileged)),
                    replace=False
                )
        )
        
        for a in newly_affected_agents:
            a.impact = 1 if a.marginalized else -1


    def step(self):
        self.agents.shuffle_do("step")

        # if support is less than benefit, increase policy coverage
        if self.policy_reaction and av_opinion(self) < av_impact(self): 
            self.apply_policy(self.rel_policy_expansion)
        
        # collect data
        self.datacollector.collect(self)
