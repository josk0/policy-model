import numpy as np
import pandas as pd
import mesa
import os

from datetime import datetime
from model import PolicyModel


parameters = {
    'num_agents': 300,  
    'privileged_fraction': 0.1,  
    'marginalized_fraction': 0.3,  
    'rel_policy_expansion': np.linspace(0.01, 0.02, 20),  
    'policy_impact_bias_pro_marginalized': np.linspace(0.5, 0.55, 10),  
    'trigger_level': 0.45,  
    'policy_reaction': False,  
}

def print_dictionary(data):
    print("\nParameters:")
    for key, value in data.items():
        print(f" {key}: {value}")

def save_results_to_csv(data):
  try:
    filename = f"output/run-{datetime.now():%Y-%m-%d-%H-%M}.csv"
    # Ensure the directory exists
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"\n Results saved to: {filename}")
  except PermissionError as e:
    print(f"Permission error: {e}. Could not write to the specified directory.")
  except FileNotFoundError as e:
    print(f"File not found error: {e}. The file path seems incorrect.")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

def main():
  print_dictionary(parameters)
  print("\nStarting batch run...")

  results = mesa.batch_run(
    PolicyModel,
    parameters=parameters,
    iterations=10,
    max_steps=60,
    number_processes=12,
    data_collection_period=1,
    display_progress=True,
  )

  save_results_to_csv(results)

if __name__ == "__main__":
    main()