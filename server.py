import collections
import functools
import json
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from flask import Flask, request, jsonify, render_template
import threading
import time
import random
import requests # Import requests for the client simulation thread

import warnings
warnings.filterwarnings('ignore')

# --- Flask App Setup ---
app = Flask(__name__)

# --- Global Variables for FL State (in-memory simulation) ---
global_model = None
# Stores preprocessed tf.data.Dataset for each client, keyed by client_id
client_data_dict = {} 
char_to_id = {}
id_to_char = {}
vocab = []
VOCAB_SIZE = 0
SEQUENCE_LENGTH = 100
BATCH_SIZE = 10
NUM_CLIENTS_PER_ROUND = 5 # Number of clients participating in each round
LOCAL_EPOCHS = 1 # Number of local epochs for each client in a round
FL_ROUND_COUNT = 0 # To keep track of current FL round

# For storing metrics history for the dashboard
fl_metrics_history = [] # List of dictionaries: [{'round': X, 'loss': Y, 'accuracy': Z}]

# Variables for managing client uploads in a round
# Stores weights uploaded by clients in the current round
client_weights_buffer = {} 
# Event to signal when all clients have uploaded for the current round
client_completion_event = threading.Event()
# Lock to protect shared resources (client_weights_buffer, client_completion_event)
client_buffer_lock = threading.Lock()

# --- Helper Functions for Data Preprocessing and Model Handling ---

def load_and_preprocess_shakespeare_data():
    """
    Loads the Shakespeare dataset and preprocesses it for all clients.
    Initializes vocabulary and partitions data by client.
    """
    global char_to_id, id_to_char, vocab, VOCAB_SIZE, client_data_dict

    print("Loading Shakespeare dataset...")
    train_data, _ = tff.simulation.datasets.shakespeare.load_data()

    # Build vocabulary from all characters in the training data
    all_chars_set = set()
    # Iterate over a subset of client_ids to build a representative vocabulary
    # This is important for non-IID data where some chars might be rare
    # We'll use a larger subset for vocabulary building than for client simulation
    for client_id in train_data.client_ids[:100]: # Use first 100 clients to build vocab
        client_tf_dataset = train_data.create_tf_dataset_for_client(client_id)
        # Iterate through the dataset to get actual snippets
        for sample in client_tf_dataset:
            snippets_tensor = sample['snippets']
            # Ensure it's a string tensor before processing
            if snippets_tensor.dtype == tf.string:
                numpy_snippets = snippets_tensor.numpy()
                
                # Ensure numpy_snippets is always a numpy array for ndim check
                # In some TF versions/scenarios, scalar tf.string.numpy() can return raw bytes
                # rather than a 0-dim numpy array. This ensures consistency.
                if not isinstance(numpy_snippets, np.ndarray):
                    numpy_snippets = np.array(numpy_snippets)

                if numpy_snippets.ndim == 0: # Scalar tensor, single bytes object
                    # Extract the scalar bytes object from the 0-dim numpy array using .item()
                    text_content = tf.compat.as_text(numpy_snippets.item()) 
                    all_chars_set.update(text_content)
                else: # Vector tensor, array of bytes objects
                    for snippet_bytes in numpy_snippets:
                        text_content = tf.compat.as_text(snippet_bytes)
                        all_chars_set.update(text_content)
            else:
                print(f"Server: Warning: Client {client_id} has unexpected dtype for 'snippets': {snippets_tensor.dtype}. Skipping.")

    vocab = sorted(list(all_chars_set))
    vocab = ['<pad>', '<unk>'] + vocab # Add special tokens
    char_to_id = {char: i for i, char in enumerate(vocab)}
    id_to_char = {i: char for i, char in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    def to_ids(text):
        """Converts a string of characters to a list of character IDs."""
        decoded_text = text.numpy().decode('utf-8')
        ids = [char_to_id.get(c, char_to_id['<unk>']) for c in decoded_text]
        # Ensure it's never an empty list, return a placeholder if empty
        if not ids:
            return tf.constant([char_to_id['<pad>']], dtype=tf.int32) # Return a single pad token if empty
        return tf.constant(ids, dtype=tf.int32)

    def preprocess_client_dataset(dataset, num_epochs=1):
        """
        Preprocesses a single client's Shakespeare dataset and repeats it for local epochs.
        """
        def prepare_sequences(element):
            text = element['snippets']
            char_ids = tf.py_function(to_ids, [text], tf.int32)
            char_ids.set_shape([None]) # This indicates it's a 1D tensor of unknown length

            dataset = tf.data.Dataset.from_tensor_slices(char_ids)
            dataset = dataset.window(SEQUENCE_LENGTH + 1, shift=1, drop_remainder=True)
            dataset = dataset.flat_map(lambda window: window.batch(SEQUENCE_LENGTH + 1))
            
            def split_input_target(chunk):
                input_text = chunk[:-1]
                target_text = chunk[1:]
                # --- KEY CHANGE: Return a tuple (input, target) instead of OrderedDict ---
                return (input_text, target_text) 

            return dataset.map(split_input_target)

        return dataset.flat_map(prepare_sequences).shuffle(buffer_size=1000).batch(
            BATCH_SIZE, drop_remainder=True
        ).repeat(num_epochs).prefetch(tf.data.AUTOTUNE)

    # Select the first N clients for our simulation
    # These are the actual clients whose data will be used in FL rounds
    selected_client_ids = train_data.client_ids[:NUM_CLIENTS_PER_ROUND]
    for client_id in selected_client_ids:
        client_data_dict[client_id] = preprocess_client_dataset(
            train_data.create_tf_dataset_for_client(client_id), num_epochs=LOCAL_EPOCHS
        )
    print(f"Loaded data for {len(client_data_dict)} clients participating in FL.")

def create_keras_model():
    """Builds a simple character-level RNN model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(VOCAB_SIZE, 256, input_length=SEQUENCE_LENGTH),
        tf.keras.layers.GRU(512, return_sequences=True, stateful=False),
        tf.keras.layers.Dense(VOCAB_SIZE)
    ])
    return model

def get_model_weights(model):
    """Extracts weights from a Keras model."""
    return model.get_weights()

def set_model_weights(model, weights):
    """Sets weights to a Keras model."""
    model.set_weights(weights)

def weights_to_json(weights):
    """Converts Keras weights (list of numpy arrays) to JSON-serializable format."""
    return [w.tolist() for w in weights]

def json_to_weights(json_weights):
    """Converts JSON-serialized weights back to numpy arrays with correct shapes."""
    # Create a dummy model to get the correct weight shapes and dtypes
    dummy_model = create_keras_model()
    dummy_weights = dummy_model.get_weights()
    
    numpy_weights = []
    for i, w_list in enumerate(json_weights):
        # Ensure the numpy array has the correct shape and dtype from the dummy model
        numpy_weights.append(np.array(w_list, dtype=dummy_weights[i].dtype))
    return numpy_weights

def aggregate_weights(client_weights_list):
    """Performs federated averaging of client weights."""
    if not client_weights_list:
        return None

    num_clients = len(client_weights_list)
    aggregated_weights = [np.zeros_like(w) for w in client_weights_list[0]]

    for client_weights in client_weights_list:
        for i in range(len(aggregated_weights)):
            aggregated_weights[i] += client_weights[i]

    return [w / num_clients for w in aggregated_weights]

def generate_text(model, start_string, num_generate=500, temperature=0.7):
    """
    Generates text using the trained Keras model.
    `temperature` controls randomness (lower for less random, higher for more random).
    """
    if not char_to_id or not id_to_char:
        return "Error: Vocabulary not initialized. Please initialize FL first."

    input_eval = [char_to_id.get(s, char_to_id['<unk>']) for s in start_string]
    # Ensure input_eval has at least SEQUENCE_LENGTH tokens, pad if necessary
    if len(input_eval) < SEQUENCE_LENGTH:
        input_eval = [char_to_id['<pad>']] * (SEQUENCE_LENGTH - len(input_eval)) + input_eval
    input_eval = input_eval[-SEQUENCE_LENGTH:] # Take last SEQUENCE_LENGTH tokens

    input_eval = tf.expand_dims(input_eval, 0) # Add batch dimension

    text_generated = []
    
    # Keras GRU layers in Sequential model might not expose reset_states easily
    # For generation, it's often better to build a model that explicitly handles state
    # or just rely on the model processing one character at a time.
    # For simplicity, we'll use the current model as is.
    # If the model was stateful=True and built with batch_input_shape, you'd use model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0) # Remove the batch dimension

        # Get prediction for the last character in the sequence
        predictions = predictions[-1, :] # Get logits for the last time step

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0].numpy()

        # Append predicted_id to input_eval and take the last SEQUENCE_LENGTH for next step
        input_eval = tf.concat([input_eval[:, 1:], tf.expand_dims([predicted_id], 0)], axis=1)
        text_generated.append(id_to_char[predicted_id])

    return start_string + ''.join(text_generated)

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Assuming index.html is now in a 'templates' directory
    return render_template('index.html', fl_round_count=FL_ROUND_COUNT)

@app.route('/init_fl', methods=['POST'])
def init_fl():
    """Initializes the global model and loads client data."""
    global global_model, FL_ROUND_COUNT, fl_metrics_history

    try:
        # Load and preprocess data for all clients
        load_and_preprocess_shakespeare_data()

        # Initialize the global model
        global_model = create_keras_model()
        # No need to compile here, compilation happens on client side for local training
        # and when evaluating the global model.
        
        FL_ROUND_COUNT = 0
        fl_metrics_history = [] # Clear history on init
        return jsonify(status="success", message="FL initialized", fl_round_count=FL_ROUND_COUNT)
    except Exception as e:
        print(f"Error during FL initialization: {e}")
        return jsonify(status="error", message=str(e)), 500

@app.route('/download_model', methods=['GET'])
def download_model():
    """Endpoint for clients to download the current global model weights."""
    if global_model is None:
        return jsonify(status="error", message="Model not initialized."), 400
    
    weights = get_model_weights(global_model)
    return jsonify(status="success", weights=weights_to_json(weights))

@app.route('/upload_weights', methods=['POST'])
def upload_weights():
    """Endpoint for clients to upload their trained local model weights."""
    data = request.get_json()
    client_id = data.get('client_id')
    json_weights = data.get('weights')
    local_metrics = data.get('metrics', {})

    if not client_id or not json_weights:
        return jsonify(status="error", message="Missing client_id or weights."), 400

    with client_buffer_lock:
        if client_id not in client_weights_buffer: # Only accept first upload per client per round
            client_weights_buffer[client_id] = json_to_weights(json_weights)
            print(f"\nServer: Received weights from client {client_id}. Local metrics: {local_metrics}")
            # Check if all expected clients have uploaded
            if len(client_weights_buffer) == NUM_CLIENTS_PER_ROUND:
                client_completion_event.set() # Signal that all clients have uploaded
    
    return jsonify(status="success", message="Weights received.")

@app.route('/run_fl_round', methods=['POST'])
def run_fl_round():
    """
    Simulates one federated learning round by triggering client threads
    and waiting for their completion.
    """
    global global_model, FL_ROUND_COUNT, fl_metrics_history, client_weights_buffer

    if global_model is None:
        return jsonify(status="error", message="FL not initialized. Please run /init_fl first."), 400

    FL_ROUND_COUNT += 1
    print(f"\n--- Starting Federated Round {FL_ROUND_COUNT} ---")

    # Reset buffer and event for the new round
    with client_buffer_lock:
        client_weights_buffer.clear()
        client_completion_event.clear()

    # Get the list of client IDs that will participate in this round
    # In a real scenario, this would be a selection process.
    # Here, we use the pre-selected clients from client_data_dict.
    participating_client_ids = list(client_data_dict.keys())
    
    # Spawn threads for each client
    threads = []
    for client_id in participating_client_ids:
        # Pass server URL and client_id to the client script
        thread = threading.Thread(target=simulate_client_training, args=(client_id, "http://127.0.0.1:5000"))
        threads.append(thread)
        thread.start()

    # Wait for all clients to upload their weights (with a timeout)
    print(f"Server: Waiting for {NUM_CLIENTS_PER_ROUND} clients to upload weights...")
    if not client_completion_event.wait(timeout=300): # 5 minute timeout
        print("Server: Timeout waiting for all clients to upload. Aggregating from available clients.")
        # Decide how to handle missing clients: aggregate from available, or fail.
        # For this simulation, we'll aggregate from whatever we got.
        pass

    with client_buffer_lock:
        if not client_weights_buffer:
            print("No client weights received. Skipping aggregation.")
            return jsonify(status="error", message="No clients provided updates in this round."), 500
        
        # Aggregate client updates
        print(f"Server: Aggregating weights from {len(client_weights_buffer)} clients.")
        client_updates_list = list(client_weights_buffer.values())
        aggregated_weights = aggregate_weights(client_updates_list)
        
        # Update the global model with aggregated weights
        set_model_weights(global_model, aggregated_weights)
        print("Server: Global model updated.")

        # Optional: Evaluate the global model after aggregation (can be slow)
        # Create a dummy model for evaluation purposes
        eval_model = create_keras_model()
        set_model_weights(eval_model, get_model_weights(global_model))
        eval_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        # Using a dummy client's test data for evaluation for simplicity
        # This is not ideal for true global model evaluation, but serves for simulation
        # of seeing *some* metric change.
        if participating_client_ids:
            sample_client_id_for_eval = participating_client_ids[0]
            # Create a small test dataset from the *original* test_data for a client
            # to avoid using the same data used for training.
            _, raw_test_data = tff.simulation.datasets.shakespeare.load_data()
            preprocessed_test_dataset = preprocess_client_dataset(
                raw_test_data.create_tf_dataset_for_client(sample_client_id_for_eval), num_epochs=1
            )
            
            # Evaluate only a few batches to keep it fast
            eval_batches = preprocessed_test_dataset.take(5) 
            
            try:
                eval_results = eval_model.evaluate(eval_batches, verbose=0)
                eval_loss = eval_results[0]
                eval_accuracy = eval_results[1]
                print(f"Server: Global Model Evaluation (Round {FL_ROUND_COUNT}): Loss={eval_loss:.4f}, Accuracy={eval_accuracy:.4f}")
                fl_metrics_history.append({'round': FL_ROUND_COUNT, 'loss': float(eval_loss), 'accuracy': float(eval_accuracy)})
            except Exception as e:
                print(f"Server: Error during global model evaluation: {e}")
                fl_metrics_history.append({'round': FL_ROUND_COUNT, 'loss': np.nan, 'accuracy': np.nan})
        else:
            print("Server: No clients to evaluate on.")
            fl_metrics_history.append({'round': FL_ROUND_COUNT, 'loss': np.nan, 'accuracy': np.nan})


    return jsonify(status="success", message="Federated round complete", fl_round_count=FL_ROUND_COUNT)

@app.route('/reset_fl', methods=['POST'])
def reset_fl():
    """Resets the federated learning state."""
    global global_model, client_data_dict, char_to_id, id_to_char, vocab, VOCAB_SIZE, FL_ROUND_COUNT, fl_metrics_history
    global_model = None
    client_data_dict = {}
    char_to_id = {}
    id_to_char = {}
    vocab = []
    VOCAB_SIZE = 0
    FL_ROUND_COUNT = 0
    fl_metrics_history = []
    print("Federated Learning state reset.")
    return jsonify(status="success", message="FL state reset", fl_round_count=FL_ROUND_COUNT)


@app.route('/generate', methods=['POST'])
def generate_text_endpoint():
    """Generates text using the current global model."""
    global global_model

    if global_model is None:
        return jsonify(status="error", message="Model not trained. Please initialize and run FL rounds."), 400

    data = request.get_json()
    start_string = data.get('start_string', "ROMEO:")
    num_generate = int(data.get('num_generate', 300))
    temperature = float(data.get('temperature', 0.7))

    try:
        # Ensure the global model is compiled for prediction if it hasn't been
        if not global_model.optimizer: # Check if compiled
             global_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
        generated_text = generate_text(global_model, start_string, num_generate, temperature)
        return jsonify(status="success", generated_text=generated_text)
    except Exception as e:
        print(f"Error during text generation: {e}")
        return jsonify(status="error", message=f"Failed to generate text: {str(e)}"), 500

@app.route('/metrics', methods=['GET'])
def get_metrics_history():
    """Returns the history of FL metrics for the dashboard."""
    return jsonify(metrics_history=fl_metrics_history)

# --- Client Simulation Function (run by threads on server side) ---
def simulate_client_training(client_id, server_url):
    """
    Simulates a single client's federated learning process.
    This function will be run in a separate thread.
    """
    print(f"Client: {client_id}: Starting local training simulation.")
    try:
        # 1. Download global model
        response = requests.get(f"{server_url}/download_model")
        response.raise_for_status() # Raise an exception for HTTP errors
        server_weights_json = response.json()['weights']
        server_weights = json_to_weights(server_weights_json)

        # 2. Create local model and set global weights
        local_model = create_keras_model()
        set_model_weights(local_model, server_weights)
        local_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        # 3. Get client's local data
        local_dataset = client_data_dict.get(client_id)
        if local_dataset is None:
            print(f"Client: {client_id}: No data found for this client. Skipping.")
            return

        # 4. Train locally
        # Changed verbose=0 to verbose=1 for console output during training
        history = local_model.fit(local_dataset, epochs=LOCAL_EPOCHS, verbose=1) 
        local_loss = history.history['loss'][-1]
        local_accuracy = history.history['sparse_categorical_accuracy'][-1]
        print(f"\nClient: {client_id}: Local training complete. Loss={local_loss:.4f}, Acc={local_accuracy:.4f}")

        # 5. Upload updated weights
        updated_weights = get_model_weights(local_model)
        upload_payload = {
            'client_id': client_id,
            'weights': weights_to_json(updated_weights),
            'metrics': {'loss': float(local_loss), 'accuracy': float(local_accuracy)}
        }
        response = requests.post(f"{server_url}/upload_weights", json=upload_payload)
        response.raise_for_status()
        print(f"Client {client_id}: Weights uploaded successfully.")

    except requests.exceptions.ConnectionError as e:
        print(f"Client {client_id}: Connection error to server: {e}")
    except Exception as e:
        print(f"Client {client_id}: An error occurred during simulation: {e}")

if __name__ == '__main__':
    # Run the Flask app
    # Use threaded=True to allow multiple client threads to run concurrently
    # debug=True is useful for development but should be False in production
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
