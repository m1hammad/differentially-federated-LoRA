import flwr as fl

import logging
import numpy as np
import signal

import subprocess
import time


# Configure logging
logging.basicConfig(level=logging.INFO)

import flwr as fl

def start_fl_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1, 
        fraction_evaluate=0.1, 

        # For training
        min_fit_clients=10, 
        min_evaluate_clients=10, 
        min_available_clients=10,

        # # For Debugging
        # min_fit_clients=1, 
        # min_evaluate_clients=1, 
        # min_available_clients=1,
    )
    try: 
        logging.info("Starting Flower server...")
        fl.server.start_server(
            server_address="localhost:9094",
            config=fl.server.ServerConfig(num_rounds=10, round_timeout=600),
            strategy=strategy,   # uncomment if you want to work with opacus
            # strategy=dp_strategy,
        )
    except Exception as e:
        logging.error(f"Server encountered an error: {e}")
    finally:
        logging.info("Server finished all rounds. Exiting gracefully.")

    


# # Global shutdown flag
# shutdown_flag = False  # Initialize the shutdown flag

# def signal_handler(sig, frame):
#     ## Signal handler for graceful shutdown when a signal (like SIGINT) is received.
#     global shutdown_flag
#     print("Signal received, preparing to shut down...")
#     shutdown_flag = True

# signal.signal(signal.SIGINT, signal_handler)



# # def start_fl_server(dp_mode="none", adaptive=False, noise_multiplier=1.0, clipping_norm=1.0):
# #     base_strategy = FedAvg(
# #         fraction_fit=0.1,  # Fraction of clients used during training
# #         fraction_evaluate=0.1,  # Fraction of clients used during evaluation

# #         # min_fit_clients=10,  # Minimum number of clients used during training
# #         # min_evaluate_clients=10,  # Minimum number of clients used during evaluation
# #         # min_available_clients=10,  # Minimum number of total clients in the system

# #         # Set to 1 for debugging
# #         min_fit_clients=1,
# #         min_evaluate_clients=1,
# #         min_available_clients=1,

# #         fit_metrics_aggregation_fn=lambda metrics: {
# #             "accuracy": np.mean([m[1]["accuracy"] for m in metrics if "accuracy" in m[1]])
# #         },
# #         evaluate_metrics_aggregation_fn=lambda metrics: {
# #             "accuracy": np.mean([m[1]["accuracy"] for m in metrics if "accuracy" in m[1]])
# #         },
# #     )
# #     if dp_mode == "server":
# #         if adaptive:
# #             strategy = DifferentialPrivacyServerSideAdaptiveClipping(
# #                 strategy=base_strategy,
# #                 noise_multiplier=noise_multiplier,
# #                 initial_clipping_norm=clipping_norm,
# #                 target_clipped_quantile=0.5,
# #                 clip_norm_lr=0.2,
# #                 num_sampled_clients = 10
# #             )
# #         else:
# #             strategy = DifferentialPrivacyServerSideFixedClipping(
# #                 strategy=base_strategy,
# #                 noise_multiplier=noise_multiplier,
# #                 clipping_norm=clipping_norm,
# #                 num_sampled_clients = 10
# #             )
# #     # elif dp_mode == "client":
# #     #     # Client-Side Fixed Clipping
# #     #     strategy = DifferentialPrivacyClientSideFixedClipping(
# #     #         strategy=base_strategy,
# #     #         noise_multiplier=noise_multiplier,
# #     #         clipping_norm=clipping_norm,
# #     #         num_sampled_clients = 10
# #     #     )
# #     else:
# #         strategy = base_strategy

# #     # Start Flower server
# #     try: 
# #         logging.info("Starting Flower server...")
# #         fl.server.start_server(
# #             # server_address="localhost:8080",
# #             server_address="127.0.0.1:8080",  # Allow external connections
# #             config=fl.server.ServerConfig(num_rounds=10, round_timeout=60),
# #             strategy=strategy
# #         )
# #     except Exception as e:
# #         logging.error(f"Server encountered an error: {e}")
# #     finally:
# #         if shutdown_flag:
# #             logging.info("Shutting down Flower server gracefully.")
# #         else:
# #             logging.info("Server finished all rounds. Exiting gracefully.")


# def start_fl_server():
#     server_process = subprocess.Popen(["python", "-m", "server"])  # Adjust if server.py is not in the same folder
#     time.sleep(5)  # Allow the server to initialize
#     print('Server has started')
#     return server_process

# def stop_fl_server(server_process):
#     print("Stopping the Flower server...")
#     server_process.terminate()
#     return_code = server_process.wait()             # Wait for the process to terminate and capture the return code
#     if return_code == 0:
#         print("Server terminated successfully.")
#     else:
#         print(f"Server terminated with error code: {return_code}")

# def main():

#     global shutdown_flag
#     # Define strategy
#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=0.1,  # Fraction of clients used during training
#         fraction_evaluate=0.1,  # Fraction of clients used during evaluation

#         # min_fit_clients=10,  # Minimum number of clients used during training
#         # min_evaluate_clients=10,  # Minimum number of clients used during evaluation
#         # min_available_clients=10,  # Minimum number of total clients in the system

#         # Set to 1 for debugging
#         min_fit_clients=1,  
#         min_evaluate_clients=1,
#         min_available_clients=1,

#         fit_metrics_aggregation_fn=lambda metrics: {
#             "accuracy": np.mean([m[1]["accuracy"] for m in metrics if "accuracy" in m[1]])
#         },
#         evaluate_metrics_aggregation_fn=lambda metrics: {
#             "accuracy": np.mean([m[1]["accuracy"] for m in metrics if "accuracy" in m[1]])
#         },
#     )

#     # # Wrap the strategy with Differential Privacy
#     # dp_strategy = DifferentialPrivacyServerSideFixedClipping(
#     #     strategy=strategy,
#     #     noise_multiplier=1.0,  # Adjust based on your privacy requirements
#     #     clipping_norm=1.0,     # Adjust based on your model's sensitivity
#     #     num_sampled_clients=10 # Number of clients sampled per round
#     # )

#     # Register signal handlers for graceful shutdown
#     signal.signal(signal.SIGINT, signal_handler)
#     signal.signal(signal.SIGTERM, signal_handler)


#     # Start Flower server
#     try: 
#         logging.info("Starting Flower server...")
#         fl.server.start_server(
#             server_address="localhost:8080",
#             config=fl.server.ServerConfig(num_rounds=10, round_timeout=600),
#             # strategy=strategy,   # uncomment if you want to work with opacus
#             strategy=dp_strategy,
#         )
#     except Exception as e:
#         logging.error(f"Server encountered an error: {e}")
#     finally:
#         if shutdown_flag:
#             logging.info("Shutting down Flower server gracefully.")
#         else:
#             logging.info("Server finished all rounds. Exiting gracefully.")


# if __name__ == "__main__":
#     main()