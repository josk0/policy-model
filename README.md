# Policy Support Network Model

An agent-based model exploring how network structure influences policy support, particularly when policy effects are unevenly distributed across social networks.

## 📊 Overview

This model investigates how public support for beneficial policies can erode due to network effects, even when the policies have positive net social benefits. It explores the hypothesis that policy support depends on direct observation of policy effects, and that network structure can lead to negative effects being more visible than positive ones.

### Key Features

- **Network Structure**: Uses Barabási-Albert preferential attachment model to create realistic social networks
- **Agent Types**: 
  - Privileged agents (more connected)
  - Marginalized agents (less connected)
  - Policy-affected agents (experiencing positive or negative impacts)
- **Dynamic Network**: Agents can create new connections based on strong opinions
- **Policy Maker Behavior**: Optional "double down" mechanism where policy coverage expands in response to low support

## 🎯 Model Dynamics

### Initialization
1. Creates a scale-free network using Barabási-Albert algorithm
2. Assigns privileged status to most central nodes
3. Randomly assigns marginalized status to remaining nodes
4. Applies initial policy effects to a subset of agents

### Agent Behavior
- Agents form opinions based on observed policy effects in their neighborhood
- Opinion formation uses a logistic function bounded between -1 and 1
- When opinions exceed a trigger threshold, agents create new connections between their neighbors

### Policy Effects
- Marginalized agents receive positive policy effects (+1)
- Privileged agents receive negative policy effects (-1)
- Effects spread through network observation

## 💻 Installation
Clone repository
```bash
git clone https://github.com/josk0/policy-model.git
cd policy-model
```

### conda
``` bash
conda env create --file environment.yml
```
or manually create environment
```bash
conda create -n policymodel python numpy pandas tqdm matplotlib solara pytest scipy ipython networkx mesa
conda activate policymodel
```

... or pick your favorite package manager

### Files

* ``model.py``: Contains the model class `PolicyModel`.
* ``agents.py``: Contains the agent class `PolicyAgent`.
* ``app.py``: Contains the code for the interactive Solara visualization.
* ``analysis.ipynb``: Jupyter notebook to run and analyze the model.

## 🚀 Usage

### Web Interface
To run the model interactively, run the following command

```bash
solara run app.py
```

### Parameter Settings

- **Number of Agents**: Total population size
- **Privileged Fraction**: Proportion of highly connected agents
- **Marginalized Fraction**: Proportion of less connected agents
- **Policy Coverage**: Initial policy coverage rate (and rate at which policy maker "doubles down")
- **Trigger Level**: Opinion threshold for network modification
- **Policy Reaction**: Enable/disable policy maker "double down" behavior

### Visualization Guide

The model provides real-time visualization with:
- Network view showing agent types and connections
  - 🔵 Blue: Privileged agents
  - 🔆 Light blue: Privileged agents affected by policy
  - 🔴 Red: Marginalized agents
  - 💗 Pink: Marginalized agents affected by policy
  - ⚪ Grey: Unaffected agents
- Time series of average policy support and social benefit
- Histogram of node degrees

## 📈 Example Results

tbc

## 🔬 Research Context

This model explores several key questions in policy dynamics:
- How does network structure influence policy support?
- Can beneficial policies lose support due to visibility bias?
- What role does social network position play in policy perception?
- How do policy maker reactions influence long-term support?

## 📋 To Dos
- Observe opinion polarization in population
- Should benefits and costs of the policy be symmetric (as they are currently)?
- tbc...

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Johannes Himmelreich ([@josk0](https://github.com/josk0))

## 🙏 Acknowledgments

This project is part of ongoing research at Syracuse University. This is my first model using [mesa](https://github.com/projectmesa/mesa). I used the "[Virus on a Network](https://github.com/projectmesa/mesa/tree/main/mesa/examples/basic/virus_on_network)" example model as a guide.
