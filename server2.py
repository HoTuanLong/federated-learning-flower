from typing import List, Tuple, Optional

import flwr as fl
import wandb
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy

if __name__ == "__main__":
    wandb.init(
        project="FL-flower",
        name="2 devices locals",
    )

    class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
        def aggregate_evaluate(
                self,
                rnd: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[BaseException],
        ) -> Optional[float]:
            """Aggregate evaluation losses using weighted average."""
            if not results:
                return None

            # Weigh accuracy of each client by number of examples used
            accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
            losses = [r.loss * r.num_examples for _, r in results]
            print(results)
            examples = [r.num_examples for _, r in results]

            # Aggregate and print custom metric
            accuracy_aggregated = sum(accuracies) / sum(examples)
            print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")
            loss_aggregated = sum(losses) / sum(examples)

            # Call aggregate_evaluate from base class (FedAvg)
            wandb.log({"acc": accuracy_aggregated, "loss": loss_aggregated})
            return super().aggregate_evaluate(rnd, results, failures)

    # Define strategy
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.5,
    #     fraction_eval=0.5,
    # )

    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": 3},
        strategy=strategy,
    )

    wandb.finish()