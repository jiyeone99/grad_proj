"""Flower server example with partition tracking."""

from typing import List, Tuple, Dict
import os
import json
import flwr as fl
from flwr.common import Metrics
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import logging

logging.basicConfig(
    filename='server_logs.txt',  # File where logs will be saved
    level=logging.DEBUG,         # Log level (can be DEBUG, INFO, WARNING, etc.)
    filemode='w'                 # Overwrite the log file for each new run
)
logger = logging.getLogger(__name__)

# Define the save path for parameters
from pathlib import Path
SAVE_PATH = Path("client_parameters_all.json")

# Initialize a container for storing all parameters and partition IDs
all_client_data = []

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracies = [
        num_examples * m.get("accuracy", 0)  # Default to 0 if "accuracy" is missing
        for num_examples, m in metrics
    ]
    return {"accuracy": sum(accuracies) / total_examples if total_examples > 0 else 0}

def validate_parameters(results):
    expected_shapes = None
    for client_proxy, fit_res in results:  # Unpack the tuple
        parameters = parameters_to_ndarrays(fit_res.parameters)  # Extract parameters
        shapes = [param.shape for param in parameters]
        if expected_shapes is None:
            expected_shapes = shapes
        elif shapes != expected_shapes:
            print(f"Parameter shape mismatch: {shapes} vs {expected_shapes}")
            return False
    return True

def validate_and_log_shapes(results):
    base_shapes = None
    for client_idx, (client_proxy, fit_res) in enumerate(results):
        try:
            parameters = parameters_to_ndarrays(fit_res.parameters)
            shapes = [param.shape for param in parameters]
            print(f"Client {client_idx} parameter shapes: {shapes}")
            if base_shapes is None:
                base_shapes = shapes
            elif shapes != base_shapes:
                print(f"Shape mismatch detected for client {client_idx}: {shapes} vs {base_shapes}")
                return False
        except Exception as e:
            print(f"Error processing client {client_idx}: {e}")
            return False
    return True


# Define custom strategy
class CustomStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        try:
            global all_client_data
            round_data = []

            for client_proxy, fit_res in results:
                try:
                    # Convert Parameters object to list of ndarrays
                    parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                    parameters_list = [param.tolist() for param in parameters_ndarrays]
                    
                    logger.info(f"Parameters for client {client_proxy.cid}: {parameters_list[:2]}...")  # Log a snippet

                    round_data.append({
                        "round": server_round,
                        "client_id": client_proxy.cid,
                        "parameters": parameters_list,
                        "metrics": fit_res.metrics,
                    })
                except Exception as e:
                    logger.error(f"Error processing client {client_proxy.cid}: {e}")

            all_client_data.extend(round_data)
            logger.info(f"Updated all_client_data: {len(all_client_data)} entries")

            # Save to file
            with open(SAVE_PATH, "w") as f:
                json.dump(all_client_data, f, indent=4)
            logger.info(f"Client data saved to {SAVE_PATH}")

            # Proceed with normal aggregation
            return super().aggregate_fit(server_round, results, failures)
        except Exception as e:
            logger.error(f"Error in aggregate_fit: {e}")
            raise

    def save_client_data(self, results, round_number):
        global all_client_data
        try:
            for client_proxy, fit_res in results:
                parameters_ndarrays = parameters_to_ndarrays(fit_res.parameters)
                parameters_list = [param.tolist() for param in parameters_ndarrays]
                client_data = {
                    "round": round_number,
                    "client_id": client_proxy.cid,
                    "parameters": parameters_list,
                    "metrics": fit_res.metrics,
                }
                all_client_data.append(client_data)

            with open(SAVE_PATH, "w") as f:
                json.dump(all_client_data, f, indent=4)
            logger.info(f"Client data for round {round_number} saved.")
        except Exception as e:
            logger.error(f"Error saving client data for round {round_number}: {e}")


# Main server start
if __name__ == "__main__":
    try:
        strategy = CustomStrategy(
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )

        fl.server.start_server(
            server_address="192.168.0.40:8080",
            config=fl.server.ServerConfig(num_rounds=100),
            strategy=strategy,
        )
    except Exception as e:
        print(f"Client encountered an error: {e}")