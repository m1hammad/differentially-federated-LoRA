import torch
from collections import OrderedDict
# import flwr as fl
from flwr import client
from device import move_to_device
from differential_privacy import differential_privacy
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)

class FlowerClient(client.NumPyClient):
    def __init__(self, model, trainloader, testloader, device, dp_enabled=False, dp_params=None):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.dp_enabled = dp_enabled
        self.dp_params = dp_params if dp_params else {}
        self.privacy_engine = None

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]    # The get_parameters function lets the server get the client's parameters
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)               # set_parameters function allows the server to send its parameters to the client
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):                                              # the fit function trains the model locally for the client
        self.set_parameters(parameters)
        print("Training Started...")
        self._train()
        print("Training Finished.")
        return self.get_parameters(), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):                                         #  the evaluate function tests the model locally and returns the relevant metrics
        self.set_parameters(parameters)
        loss, accuracy = self._test()
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}
    
    def _train(self, epochs=5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)             # Initializes the AdamW optimizer, which is commonly used for training transformers.
        criterion = torch.nn.CrossEntropyLoss()                                     # Initializes the CrossEntropyLoss function, commonly used for classification tasks

        if self.dp_enabled:
            print('Running with Differential Privacy...')
            self.model, optimizer, self.trainloader = differential_privacy(         # integrate differential privacy
                model=self.model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                **self.dp_params                                                    # Additional params like noise multiplier, max gradient norm
            )

        self.model.train()
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            for batch in self.trainloader:
                optimizer.zero_grad()                                               # Reset gradient from previous batch

                # GPU acceleration
                input_ids = move_to_device(batch["input_ids"], self.device)
                attention_mask = move_to_device(batch["attention_mask"], self.device)
                labels = move_to_device(batch["label"], self.device)

                # the actual training
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

    def _test(self):
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()                                                           # Disables certain behaviors such as dropout layers and batch normalization updates
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():                                                       # Disables gradient computation
            for batch in self.testloader:

                # GPU acceleration 
                input_ids = move_to_device(batch["input_ids"], self.device)
                attention_mask = move_to_device(batch["attention_mask"], self.device)
                labels = move_to_device(batch["label"], self.device)

                # The acual evaluation/ testing part
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = 0.0 if total == 0 else correct / total
        return total_loss / (len(self.testloader) if len(self.testloader) > 0 else 1), accuracy

