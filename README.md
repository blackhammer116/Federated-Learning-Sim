# **Federated Learning Simulation (Shakespeare Text Generation)**

- This project demonstrates a simulated Federated Learning (FL) environment using Flask to train a character-level text generation model on Shakespeare's works. It highlights key FL concepts including data heterogeneity, differential privacy (DP), and robust aggregation against malicious clients.

## **Features**
FL Simulation: Flask server orchestrates FL rounds; clients are simulated as threads.

Text Generation: Keras GRU model trained to predict next characters.

Heterogeneity: Clients have varying data volumes, computation delays (including stragglers), and non-IID data distribution.

Differential Privacy (DP): L2 norm clipping and Gaussian noise added to client updates.

Malicious Clients: Configurable number of clients invert their updates to poison the model.

Robust Aggregation: Trimmed Mean aggregation mitigates malicious updates by discarding outliers.

Model Persistence: Global model weights are saved after each round.

Web UI: Simple interface to control FL rounds and generate text.

## **How it Works**
- Initialize FL: Server loads data, builds vocabulary, and initializes a global Keras model (optionally from pre-trained weights).

- Run FL Round: Server selects a random subset of clients, assigning them simulated delays, data limits, and a malicious flag. Each client (simulated in a thread) downloads the global model, trains locally, inverts updates if malicious, applies DP (clipping + noise), and uploads the noisy updates. Clients with empty datasets gracefully skip. Server aggregates noisy updates using Trimmed Mean, updates the global model, saves weights, and evaluates performance (gracefully handles empty evaluation datasets).

- Generate Text: Use the current global model to generate text from a given starting string.

- Reset FL: Clears all FL state.

## **Setup and Installation**
1. Clone the repo

2. Virtual Environment (Recommended):

```
python3 -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
```
3. Install Dependencies:

``` pip install -r requirements.txt ```
4. Running the Server
Start Server:

``` python server.py ```

(Server runs on http://127.0.0.1:5000/)

5. Interacting with the Web UI
Open http://127.0.0.1:5000/ in your browser.

Initialize FL: First step.

Run FL Round: Starts a training round. Observe client activities and aggregation in the server's console.

Reset FL State: Clears all progress.

Generate Text: Input a starting string to get model-generated text.

Configuration Parameters
Adjust constants at the top of server.py to customize simulation: NUM_CLIENTS_PER_ROUND, LOCAL_EPOCHS, L2_NORM_CLIP, DP_NOISE_MULTIPLIER, client delays, batch limits, MALICIOUS_CLIENT_COUNT, TRIM_FRACTION.