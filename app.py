import solara
from mesa.visualization import Slider, SolaraViz, make_plot_component, make_space_component
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure
from model import PolicyModel

def agent_portrayal(agent):
    color = "gray"
    if agent.privileged:
        if agent.impact == 0: 
            color = "blue"  
        else: 
            color = "cyan"
    elif agent.marginalized:
        if agent.impact == 0: 
            color = "red"  
        else: 
            color = "pink"
    return {"size": 15, "color": color}


model_params = {
    "num_agents": Slider("Number of Agents", 140, 20, 700, 10),
    "privileged_fraction": Slider("Privileged Fraction", 0.1, 0.05, 0.3, 0.05),
    "marginalized_fraction": Slider("Marginalized Fraction", 0.3, 0.1, 0.5, 0.05),
    "rel_policy_expansion": Slider("Policy Coverage Expansion Rate", 0.02, 0.0, 0.1, 0.01),
    "trigger_level": Slider("Trigger Level", 0.45, 0.2, 1.0, 0.05),
    "policy_reaction": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "policy maker doubles down?",
    },
    # "seed": {
    #     "type": "InputText",
    #     "value": 42,
    #     "label": "Random Seed",
    # },
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

@solara.component
def Histogram(model):
    update_counter.get() # This is required to update the counter
    # Note: you must initialize a figure using this method instead of
    # plt.figure(), for thread safety purpose
    fig = Figure()
    ax = fig.subplots()
    connection_vals = [agent.num_connections for agent in model.agents]
    # Note: you have to use Matplotlib's OOP API instead of plt.hist
    # because plt.hist is not thread-safe.
    ax.hist(connection_vals)
    solara.FigureMatplotlib(fig)

model = PolicyModel()
page = SolaraViz(
    model,
    components=[
        NetworkPlot, 
        StatePlot,
        Histogram,
    ],
    model_params=model_params,
    name = "Policy Opinion Dynamics",
)
