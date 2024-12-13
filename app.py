import solara
from mesa.visualization import Slider, SolaraViz, make_plot_component, make_space_component
import networkx as nx
from model import PolicyModel
from agents import PolicyAgent

def agent_portrayal(agent):
    color = "gray"
    if agent.privileged:
        if agent.impact == 0: color = "blue"  
        else: color = "cyan"
    elif agent.marginalized:
        if agent.impact == 0: color = "red"  
        else: color = "pink"
    return {"size": 10, "color": color}


model_params = {
    "num_agents": Slider("Number of Agents", 20, 10, 500, 10),
    "privileged_fraction": Slider("Privileged Fraction", 0.2, 0.0, 1.0, 0.1),
    "marginalized_fraction": Slider("Marginalized Fraction", 0.2, 0.0, 1.0, 0.1),
    "policy_expansion": Slider("Policy Coverage Expansion Rate", 0.02, 0.0, 0.2, 0.01),
    "trigger_level": Slider("Trigger Level", 0.35, 0.1, 1.0, 0.05),
}

def post_process_plot(ax):
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

NetworkPlot = make_space_component(agent_portrayal)

StatePlot = make_plot_component(
    {
        "Avg Policy Support": "blue",
        "Avg Social Benefit": "green"
    },
    post_process=post_process_plot
)

model = PolicyModel()
page = SolaraViz(
    model,
    components=[
        NetworkPlot, 
        StatePlot,
    ],
    model_params=model_params,
    name = "Policy Opinion Dynamics",
)
