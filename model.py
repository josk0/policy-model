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
        num_agents: int = 80,
        privileged_fraction: float = 0.3,
        marginalized_fraction: float = 0.3,
        rel_policy_expansion: float = 0.05,
        trigger_level: float = 0.35,
        policy_reaction: bool = False,
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
              self,
              False,
              False,
          )
          a.marginalized = node in marginalized_nodes
          a.privileged = node in privileged_nodes  
          self.grid.place_agent(a, node)

        # Apply initial policy intervention
        self.apply_policy(self.rel_policy_expansion)
    

    def apply_policy(self, rel_policy_expansion):
        eligible_nodes_nm = [ # nodes that are both not marginalized and not (yet) affected by policy
            node for node in self.graph.nodes() 
            if self.grid.get_cell_list_contents([node])[0].impact == 0 and self.grid.get_cell_list_contents([node])[0].privileged
        ]
        eligible_nodes_m = [ # nodes that are marginalized and not (yet) affected by policy
            node for node in self.graph.nodes() 
            if self.grid.get_cell_list_contents([node])[0].impact == 0 and self.grid.get_cell_list_contents([node])[0].marginalized
        ]
        
        num_policy_expansion = max(1, int(round(self.num_agents * rel_policy_expansion / 2)))
        affected_nodes = []

        # Handle marginalized nodes
        if eligible_nodes_m:
            size_m = min(num_policy_expansion, len(eligible_nodes_m))
            affected_nodes.extend(
                np.random.choice(
                    eligible_nodes_m,
                    size=size_m,
                    replace=False
                )
            )
        
        # Handle non-marginalized nodes
        if eligible_nodes_nm:
            size_nm = min(num_policy_expansion, len(eligible_nodes_nm))
            affected_nodes.extend(
                np.random.choice(
                    eligible_nodes_nm,
                    size=size_nm,
                    replace=False
                )
        )
        
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
                     # Remove the selected unaligned party from affected_neighbors if present
                    available_affected = [n for n in affected_neighbors if n != party_unaligned]
                    if available_affected:  # Only proceed if we have valid targets
                        party_affected = np.random.choice(available_affected)
                        self.grid.G.add_edge(party_unaligned, party_affected)
            
        # is support is less than benefit, increase policy coverage
        if self.policy_reaction and av_opinion(self) < av_impact(self): self.apply_policy(self.rel_policy_expansion)
        
        # collect data
        self.datacollector.collect(self)
