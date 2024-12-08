from device import get_device
from clients import FlowerClient
from server import start_fl_server
import flwr as fl


# def launch_server():
#     start_fl_server()


def launch_client(model, trainloader, testloader, dp_enabled, dp_params):
    client = FlowerClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=get_device(),
        dp_enabled=dp_enabled,
        dp_params=dp_params,
    )
    client.to_client()
#     fl.client.start_client(
#     server_address="localhost:8080",
#     client=client.to_client(),  # Convert NumPyClient to a standard Flower client
# )