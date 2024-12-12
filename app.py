import solara
from mesa.visualization import Slider, SolaraViz, make_plot_component, make_space_component
import networkx as nx
from model import PolicyModel
from agents import PolicyAgent

def agent_portrayal(agent):
    return {
        "color": "blue" if agent.privileged else "red" if agent.marginalized else "gray",
        "size": 10
    }

model_params = {
    "num_agents": Slider("Number of Agents", 100, 10, 500, 10),
    "privileged_fraction": Slider("Privileged Fraction", 0.2, 0.0, 1.0, 0.1),
    "marginalized_fraction": Slider("Marginalized Fraction", 0.1, 0.0, 1.0, 0.1),
    "steps": Slider("Number of Steps", 10, 1, 50, 1),
}

def post_process_plot(ax):
    ax.set_ylabel("Value")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

NetworkPlot = make_space_component(agent_portrayal)

StatePlot = make_plot_component(
    {
        "Net Policy Support": "blue",
        "Net Social Benefit": "green"
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
