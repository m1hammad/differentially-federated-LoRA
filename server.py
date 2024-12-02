import flwr as fl
import subprocess
import time


def start_fl_server():
    server_process = subprocess.Popen(["python", "-m", "server"])  # Adjust if server.py is not in the same folder
    time.sleep(5)  # Allow the server to initialize
    print('Server has started')
    return server_process

def stop_fl_server(server_process):
    server_process.terminate()
    return_code = server_process.wait()             # Wait for the process to terminate and capture the return code
    if return_code == 0:
        print("Server terminated successfully.")
    else:
        print(f"Server terminated with error code: {return_code}")

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