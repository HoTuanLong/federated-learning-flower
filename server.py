from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics

textfile = open("accuracies.txt", "w")


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy = sum(accuracies) / sum(examples)
    textfile.write(str(accuracy) + "\n")
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

print("Start server")
# Start Flower server
fl.server.start_server(
    server_address="10.42.0.1:8081", # "10.42.0.1:8080",
    # force_final_distributed_eval=True,
    config=fl.server.ServerConfig(num_rounds=10),
    # strategy=strategy,
)
textfile.close()
