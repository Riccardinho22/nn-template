import flwr as fl
from src.strategy.krum import Krum
from flwr.server.strategy import FedAvg, Strategy
# Start Flower server for three rounds of federated learning
if __name__ == "__main__":
    krum_strategy: Strategy = Krum()
    #fed_avg : Strategy = FedAvg()
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=krum_strategy)