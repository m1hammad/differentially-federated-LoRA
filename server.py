import flwr as fl

def main():
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,  # Fraction of clients used during training
        fraction_evaluate=0.1,  # Fraction of clients used during evaluation
        min_fit_clients=10,  # Minimum number of clients used during training
        min_evaluate_clients=10,  # Minimum number of clients used during evaluation
        min_available_clients=10,  # Minimum number of total clients in the system
    )

    # Start Flower server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()