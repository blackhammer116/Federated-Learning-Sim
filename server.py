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
import os # Import os to check for file existence

# --- Flask App Setup ---
app = Flask(__name__)

# --- Global Constants for FL State (in-memory simulation) ---
global_model = None
# Stores preprocessed tf.data.Dataset for each client, keyed by client_id
client_data_dict = {} 
char_to_id = {}
id_to_char = {}
vocab = []
VOCAB_SIZE = 0
SEQUENCE_LENGTH = 100
BATCH_SIZE = 10
NUM_CLIENTS_PER_ROUND = 2 # Number of clients participating in each round
LOCAL_EPOCHS = 1 # Number of local epochs for each client in a round
FL_ROUND_COUNT = 0 # To keep track of current FL round

# --- Security Constants for Differential Privacy Simulation ---
L2_NORM_CLIP = 1.0 
DP_NOISE_MULTIPLIER = 0.1 

# --- Simulation Heterogeneity Parameters ---
MIN_CLIENT_DELAY_SECONDS = 1
MAX_CLIENT_DELAY_SECONDS = 5
STRAGGLER_DELAY_SECONDS = 15 # Significantly longer delay for stragglers
STRAGGLER_COUNT = 0 # Number of stragglers per round
MIN_CLIENT_BATCHES = 5 # Minimum number of batches a client will train on
MAX_CLIENT_BATCHES = 20 # Maximum number of batches a client will train on

# --- Malicious Client Simulation & Robust Aggregation Parameters ---
MALICIOUS_CLIENT_COUNT = 0 # Number of clients to simulate as malicious per round
TRIM_FRACTION = 0.1 # Fraction of updates to trim from each end for trimmed mean (e.g., 0.1 trims 10% from top and 10% from bottom)


# For storing metrics history for the dashboard
fl_metrics_history = [] # List of dictionaries: [{'round': X, 'loss': Y, 'accuracy': Z}]

# Variables for managing client uploads in a round
client_updates_buffer = {} 
client_completion_event = threading.Event()
client_buffer_lock = threading.Lock()

# --- Helper Functions for Data Preprocessing and Model Handling ---

def to_ids(text):
    """Converts a string of characters to a list of character IDs."""
    # Ensure text is converted from EagerTensor to numpy bytes before decoding
    decoded_text = text.numpy().decode('utf-8')
    ids = [char_to_id.get(c, char_to_id['<unk>']) for c in decoded_text]
    # Ensure it's never an empty list, return a placeholder if empty
    if not ids:
        return tf.constant([char_to_id['<pad>']], dtype=tf.int32)
    return tf.constant(ids, dtype=tf.int32)

def preprocess_client_dataset(dataset, num_epochs=1, max_batches_to_use=None):
    """
    Preprocesses a single client's Shakespeare dataset and repeats it for local epochs.
    Can limit the number of batches used.
    """
    def prepare_sequences(element):
        text = element['snippets']
        char_ids = tf.py_function(to_ids, [text], tf.int32)
        char_ids.set_shape([None])

        dataset = tf.data.Dataset.from_tensor_slices(char_ids)
        dataset = dataset.window(SEQUENCE_LENGTH + 1, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda window: window.batch(SEQUENCE_LENGTH + 1))
        
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return (input_text, target_text) 

        return dataset.map(split_input_target)

    processed_dataset = dataset.flat_map(prepare_sequences).shuffle(buffer_size=1000).batch(
        BATCH_SIZE, drop_remainder=True
    ).repeat(num_epochs).prefetch(tf.data.AUTOTUNE)

    if max_batches_to_use is not None:
        processed_dataset = processed_dataset.take(max_batches_to_use)

    return processed_dataset

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
                if not isinstance(numpy_snippets, np.ndarray):
                    numpy_snippets = np.array(numpy_snippets)

                if numpy_snippets.ndim == 0: # Scalar tensor, single bytes object
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

    # Preload data for all potential clients, but don't limit batches yet.
    # Batch limiting will happen when clients are selected for a round.
    all_available_client_ids = train_data.client_ids 
    for client_id in all_available_client_ids:
        # Pass None for max_batches_to_use here, actual limiting happens per round
        client_data_dict[client_id] = preprocess_client_dataset(
            train_data.create_tf_dataset_for_client(client_id), num_epochs=LOCAL_EPOCHS, max_batches_to_use=None
        )
    print(f"Loaded data for {len(client_data_dict)} total available clients.")

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
    """
    Converts JSON-serialized weights back to numpy arrays with correct shapes and dtypes.
    Performs strict structural validation.
    """
    dummy_model = create_keras_model()
    dummy_weights = dummy_model.get_weights()
    
    if len(json_weights) != len(dummy_weights):
        raise ValueError(
            f"Uploaded weights have {len(json_weights)} layers, "
            f"but expected {len(dummy_weights)} layers based on model architecture."
        )

    numpy_weights = []
    for i, w_list in enumerate(json_weights):
        try:
            converted_w = np.array(w_list)
        except Exception as e:
            raise ValueError(f"Could not convert uploaded weight list for layer {i} to numpy array: {e}")

        if converted_w.shape != dummy_weights[i].shape:
            raise ValueError(
                f"Shape mismatch for layer {i}: Uploaded shape {converted_w.shape}, "
                f"expected shape {dummy_weights[i].shape}."
            )
        
        if converted_w.dtype != dummy_weights[i].dtype:
             print(f"Warning: Dtype mismatch for layer {i}. Uploaded: {converted_w.dtype}, Expected: {dummy_weights[i].dtype}. Attempting conversion.")
             converted_w = converted_w.astype(dummy_weights[i].dtype)
             if converted_w.dtype != dummy_weights[i].dtype:
                 raise ValueError(f"Dtype mismatch for layer {i}: Uploaded dtype {converted_w.dtype}, expected dtype {dummy_weights[i].dtype}. Conversion failed.")

        numpy_weights.append(converted_w)
    return numpy_weights

def updates_to_json(updates):
    """Converts a list of numpy update arrays to JSON-serializable format."""
    return [u.tolist() for u in updates]

def json_to_updates(json_updates):
    """Converts JSON-serialized updates back to numpy arrays with correct shapes and dtypes."""
    # Updates should have the same structure as weights
    return json_to_weights(json_updates)

def calculate_l2_norm(arrays):
    """Calculates the L2 norm of a flattened list of numpy arrays."""
    flat_arrays = np.concatenate([arr.flatten() for arr in arrays])
    return np.linalg.norm(flat_arrays)

def clip_l2_norm(updates, clip_norm):
    """Clips the L2 norm of the updates."""
    current_norm = calculate_l2_norm(updates)
    if current_norm > clip_norm:
        scaling_factor = clip_norm / current_norm
        return [u * scaling_factor for u in updates]
    return updates

def add_gaussian_noise(updates, noise_multiplier, clip_norm):
    """Adds Gaussian noise to updates for differential privacy."""
    noisy_updates = []
    for u in updates:
        noise_std = noise_multiplier * clip_norm
        noise = np.random.normal(loc=0.0, scale=noise_std, size=u.shape).astype(u.dtype)
        noisy_updates.append(u + noise)
    return noisy_updates

def aggregate_updates(client_updates_list):
    """Performs federated averaging of client updates."""
    if not client_updates_list:
        return None

    num_clients = len(client_updates_list)
    # Initialize aggregated_updates with zeros, using the shape of the first client's update
    aggregated_updates = [np.zeros_like(u) for u in client_updates_list[0]]

    for client_updates in client_updates_list:
        for i in range(len(aggregated_updates)):
            aggregated_updates[i] += client_updates[i]

    return [u / num_clients for u in aggregated_updates]

def trimmed_mean_aggregate_updates(client_updates_list, trim_fraction):
    """
    Performs trimmed mean aggregation of client updates.
    Removes a fraction of outliers from both ends before averaging.
    """
    if not client_updates_list:
        return None

    num_clients = len(client_updates_list)
    num_to_trim = int(num_clients * trim_fraction)
    
    # If not enough clients to trim, fall back to simple averaging
    if num_clients < 2 * num_to_trim + 1:
        print(f"Warning: Not enough clients ({num_clients}) for trimmed mean with trim_fraction={trim_fraction}. "
              f"Falling back to simple averaging. Need at least {2 * num_to_trim + 1} clients for trimming.")
        return aggregate_updates(client_updates_list)

    # Initialize aggregated_updates with zeros, using the shape of the first client's update
    aggregated_updates = [np.zeros_like(u) for u in client_updates_list[0]]

    # Iterate through each weight tensor (layer)
    for i in range(len(client_updates_list[0])):
        # Stack the i-th weight tensor from all clients
        # This creates a (num_clients, ...) shaped array for the current weight
        stacked_weights = np.array([client_updates[i] for client_updates in client_updates_list])
        
        # Flatten the current weight tensor for sorting and trimming
        original_shape = stacked_weights.shape[1:] # Store original shape for reshaping later
        flattened_weights = stacked_weights.reshape(num_clients, -1) # (num_clients, num_elements_in_weight)

        trimmed_flattened_weights = np.zeros_like(flattened_weights[0], dtype=flattened_weights.dtype) # Initialize with shape of one flattened weight

        # Apply trimmed mean element-wise across the flattened weights
        for j in range(flattened_weights.shape[1]): # Iterate over each element in the flattened weight
            column_values = flattened_weights[:, j]
            
            # Sort the values and trim
            sorted_values = np.sort(column_values)
            trimmed_values = sorted_values[num_to_trim : len(sorted_values) - num_to_trim]
            
            # Calculate the mean of the trimmed values
            if len(trimmed_values) > 0:
                trimmed_mean = np.mean(trimmed_values)
            else:
                trimmed_mean = 0.0 
            
            trimmed_flattened_weights[j] = trimmed_mean
        
        # Reshape back to original weight tensor shape and store
        aggregated_updates[i] = trimmed_flattened_weights.reshape(original_shape)

    return aggregated_updates


def generate_text(model, start_string, num_generate=500, temperature=0.7):
    """
    Generates text using the trained Keras model.
    `temperature` controls randomness (lower for less random, higher for more random).
    """
    if not char_to_id or not id_to_char:
        return "Error: Vocabulary not initialized. Please initialize FL first."

    input_eval = [char_to_id.get(s, char_to_id['<unk>']) for s in start_string]
    if len(input_eval) < SEQUENCE_LENGTH:
        input_eval = [char_to_id['<pad>']] * (SEQUENCE_LENGTH - len(input_eval)) + input_eval
    input_eval = tf.constant(input_eval[-SEQUENCE_LENGTH:], dtype=tf.int32)
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions[-1:] 

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0]
        predicted_id = tf.cast(predicted_id, tf.int32)

        input_eval = tf.concat([input_eval[:, 1:], tf.expand_dims(predicted_id, 0)], axis=1)
        text_generated.append(id_to_char[predicted_id.numpy()[0]])

    return start_string + ''.join(text_generated)

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html', fl_round_count=FL_ROUND_COUNT)

@app.route('/init_fl', methods=['POST'])
def init_fl():
    """Initializes the global model and loads client data."""
    global global_model, FL_ROUND_COUNT, fl_metrics_history

    try:
        load_and_preprocess_shakespeare_data()
        global_model = create_keras_model()

        pretrain_weights_path = 'initial_model_weights.json'
        if os.path.exists(pretrain_weights_path):
            print(f"Server: Found pre-trained weights at {pretrain_weights_path}. Loading...")
            with open(pretrain_weights_path, 'r') as f:
                pretrain_weights_json = json.load(f)
            
            try:
                pretrain_weights = json_to_weights(pretrain_weights_json)
                set_model_weights(global_model, pretrain_weights)
                print("Server: Pre-trained weights loaded successfully.")
            except ValueError as ve:
                print(f"Server: Error loading pre-trained weights due to structural mismatch: {ve}. Initializing with random weights.")
            except Exception as e:
                print(f"Server: Unexpected error loading pre-trained weights: {e}. Initializing with random weights.")
                import traceback
                traceback.print_exc()
        else:
            print("Server: No pre-trained weights found. Initializing model with random weights.")
        
        FL_ROUND_COUNT = 0
        fl_metrics_history = []
        return jsonify(status="success", message="FL initialized", fl_round_count=FL_ROUND_COUNT)
    except Exception as e:
        print(f"Error during FL initialization: {e}")
        import traceback
        traceback.print_exc()
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
    """
    Endpoint for clients to upload their trained local model updates (deltas).
    These updates are expected to be clipped and noised for differential privacy.
    """
    data = request.get_json()
    client_id = data.get('client_id')
    json_updates = data.get('updates') # Now expecting 'updates' not 'weights'
    local_metrics = data.get('metrics', {})

    if not client_id or not json_updates:
        return jsonify(status="error", message="Missing client_id or updates."), 400

    try:
        client_updates = json_to_updates(json_updates)
        
        with client_buffer_lock:
            if client_id not in client_updates_buffer:
                client_updates_buffer[client_id] = client_updates
                print(f"Server: Received (noised) updates from client {client_id}. Local metrics: {local_metrics}")
                
                if len(client_updates_buffer) == NUM_CLIENTS_PER_ROUND:
                    client_completion_event.set()
        
        return jsonify(status="success", message="Updates received and validated.")
    except ValueError as ve:
        print(f"Server: Client {client_id} uploaded malformed updates: {ve}. Rejecting.")
        return jsonify(status="error", message=f"Malformed updates: {ve}"), 400
    except Exception as e:
        print(f"Server: An unexpected error occurred during update upload from client {client_id}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(status="error", message=f"Server error during update upload: {e}"), 500

@app.route('/run_fl_round', methods=['POST'])
def run_fl_round():
    """
    Simulates one federated learning round by triggering client threads
    and waiting for their completion.
    Includes partial participation, straggler simulation, varying data volumes,
    and malicious client simulation with robust aggregation.
    """
    global global_model, FL_ROUND_COUNT, fl_metrics_history, client_updates_buffer

    if global_model is None:
        return jsonify(status="error", message="FL not initialized. Please run /init_fl first."), 400

    FL_ROUND_COUNT += 1
    print(f"\n--- Starting Federated Round {FL_ROUND_COUNT} ---")

    with client_buffer_lock:
        client_updates_buffer.clear()
        client_completion_event.clear()

    all_available_client_ids = list(client_data_dict.keys())
    
    # 1. Simulate partial participation: Randomly select NUM_CLIENTS_PER_ROUND
    if len(all_available_client_ids) < NUM_CLIENTS_PER_ROUND:
        print(f"Warning: Not enough clients ({len(all_available_client_ids)}) to meet NUM_CLIENTS_PER_ROUND ({NUM_CLIENTS_PER_ROUND}). Using all available clients.")
        participating_client_ids = all_available_client_ids
    else:
        participating_client_ids = random.sample(all_available_client_ids, NUM_CLIENTS_PER_ROUND)
    
    print(f"Server: Participating clients for round {FL_ROUND_COUNT}: {participating_client_ids}")

    # 2. Assign varying delays, data volumes, and malicious flags
    client_configs = {}
    
    # Select stragglers
    straggler_indices = random.sample(range(len(participating_client_ids)), min(STRAGGLER_COUNT, len(participating_client_ids)))
    
    # Select malicious clients (ensure not more than available clients, and not more than can be trimmed)
    # Malicious clients should ideally be distinct from stragglers for clearer observation
    num_potential_malicious = len(participating_client_ids) - len(straggler_indices)
    malicious_client_indices = []
    if num_potential_malicious > 0:
        malicious_client_indices = random.sample(
            [i for i in range(len(participating_client_ids)) if i not in straggler_indices],
            min(MALICIOUS_CLIENT_COUNT, num_potential_malicious)
        )

    for i, client_id in enumerate(participating_client_ids):
        simulated_delay = random.uniform(MIN_CLIENT_DELAY_SECONDS, MAX_CLIENT_DELAY_SECONDS)
        max_batches = random.randint(MIN_CLIENT_BATCHES, MAX_CLIENT_BATCHES)
        is_malicious = False

        if i in straggler_indices:
            simulated_delay = STRAGGLER_DELAY_SECONDS # Make this client a straggler
            print(f"Server: Client {client_id} assigned as STRAGGLER (delay={simulated_delay}s, batches={max_batches}).")
        
        if i in malicious_client_indices:
            is_malicious = True
            print(f"Server: Client {client_id} assigned as MALICIOUS (delay={simulated_delay}s, batches={max_batches}).")
            
        client_configs[client_id] = {
            'simulated_delay_seconds': simulated_delay,
            'max_batches_to_use': max_batches,
            'is_malicious': is_malicious
        }

    # Spawn threads for each participating client
    threads = []
    for client_id in participating_client_ids:
        config = client_configs[client_id]
        thread = threading.Thread(
            target=simulate_client_training, 
            args=(
                client_id, 
                "http://127.0.0.1:5000", 
                config['simulated_delay_seconds'], 
                config['max_batches_to_use'],
                config['is_malicious'] # Pass malicious flag
            )
        )
        threads.append(thread)
        thread.start()

    print(f"Server: Waiting for {len(participating_client_ids)} clients to upload (noised) updates (timeout: 300s)...")
    if not client_completion_event.wait(timeout=300):
        print("Server: Timeout waiting for all clients to upload. Aggregating from available clients.")
        pass

    with client_buffer_lock:
        if not client_updates_buffer:
            print("No client updates received. Skipping aggregation.")
            return jsonify(status="error", message="No clients provided updates in this round."), 500
        
        print(f"Server: Aggregating (noised) updates from {len(client_updates_buffer)} clients using Trimmed Mean (trim_fraction={TRIM_FRACTION}).")
        client_updates_list = list(client_updates_buffer.values())
        
        # --- Use Trimmed Mean Aggregation ---
        averaged_update = trimmed_mean_aggregate_updates(client_updates_list, TRIM_FRACTION)
        if averaged_update is None: # Fallback if trimmed mean cannot be performed (e.g., too few clients)
            print("Server: Trimmed mean aggregation failed or returned None. Falling back to simple averaging.")
            averaged_update = aggregate_updates(client_updates_list)


        current_global_weights = get_model_weights(global_model)
        new_global_weights = [
            global_w + avg_u for global_w, avg_u in zip(current_global_weights, averaged_update)
        ]
        set_model_weights(global_model, new_global_weights)
        print("Server: Global model updated with aggregated (noised) updates.")

        output_filename = 'latest_global_model_weights.json'
        try:
            weights_to_save = get_model_weights(global_model)
            with open(output_filename, 'w') as f:
                json.dump(weights_to_json(weights_to_save), f)
            print(f"Server: Global model weights saved to {output_filename} after round {FL_ROUND_COUNT}.")
        except Exception as e:
            print(f"Server: Error saving global model weights after round {FL_ROUND_COUNT}: {e}")
            import traceback
            traceback.print_exc()

        eval_model = create_keras_model()
        set_model_weights(eval_model, get_model_weights(global_model))
        eval_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        
        if participating_client_ids:
            sample_client_id_for_eval = participating_client_ids[0]
            _, raw_test_data = tff.simulation.datasets.shakespeare.load_data()
            
            # For evaluation, we use the full preprocessed dataset of a sample client
            # to get a more stable evaluation metric.
            preprocessed_test_dataset = preprocess_client_dataset(
                raw_test_data.create_tf_dataset_for_client(sample_client_id_for_eval), num_epochs=1, max_batches_to_use=None
            )
            
            eval_batches = preprocessed_test_dataset.take(5) 
            
            # --- Debugged: Check if eval_batches is empty before evaluating ---
            try:
                # Attempt to get one batch to check if the dataset is empty
                _ = next(iter(eval_batches))
                # If successful, proceed with evaluation
                eval_results = eval_model.evaluate(eval_batches, verbose=0)
                eval_loss = eval_results[0]
                eval_accuracy = eval_results[1]
                print(f"Server: Global Model Evaluation (Round {FL_ROUND_COUNT}): Loss={eval_loss:.4f}, Accuracy={eval_accuracy:.4f}")
                fl_metrics_history.append({'round': FL_ROUND_COUNT, 'loss': float(eval_loss), 'accuracy': float(eval_accuracy)})
            except StopIteration:
                print(f"Server: Evaluation dataset for client {sample_client_id_for_eval} is empty after taking 5 batches. Skipping global model evaluation for this round.")
                fl_metrics_history.append({'round': FL_ROUND_COUNT, 'loss': np.nan, 'accuracy': np.nan})
            except Exception as e:
                print(f"Server: Error during global model evaluation: {e}")
                import traceback
                traceback.print_exc()
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
        if not global_model.optimizer:
             global_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )
        generated_text = generate_text(global_model, start_string, num_generate, temperature)
        return jsonify(status="success", generated_text=generated_text)
    except Exception as e:
        print(f"Error during text generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify(status="error", message=f"Failed to generate text: {str(e)}"), 500

@app.route('/metrics', methods=['GET'])
def get_metrics_history():
    """Returns the history of FL metrics for the dashboard."""
    return jsonify(metrics_history=fl_metrics_history)

# --- Client Simulation Function (run by threads on server side) ---
def simulate_client_training(client_id, server_url, simulated_delay_seconds=0, max_batches_to_use=None, is_malicious=False):
    """
    Simulates a single client's federated learning process.
    This function will be run in a separate thread.
    Includes simulated delay, data volume limits, and malicious behavior.
    """
    status_msg = "MALICIOUS" if is_malicious else "NORMAL"
    print(f"Client {client_id}: Starting ({status_msg}) with delay={simulated_delay_seconds}s, max_batches={max_batches_to_use}...")
    try:
        # 1. Download global model weights
        print(f"Client {client_id}: Attempting to download global model weights.")
        response = requests.get(f"{server_url}/download_model")
        response.raise_for_status()
        server_weights_json = response.json()['weights']
        server_weights = json_to_weights(server_weights_json)
        print(f"Client {client_id}: Global model weights downloaded successfully.")

        # Simulate computation delay
        if simulated_delay_seconds > 0:
            print(f"Client {client_id}: Simulating {simulated_delay_seconds} seconds of computation delay...")
            time.sleep(simulated_delay_seconds)

        # 2. Create local model and set global weights
        local_model = create_keras_model()
        set_model_weights(local_model, server_weights)
        local_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        print(f"Client {client_id}: Local model created and compiled with global weights.")

        # 3. Get client's local data (now with potential batch limit)
        # Re-preprocess the dataset for this client with the specific max_batches_to_use
        # This ensures the client thread gets a dataset with the correct batch limit.
        _, raw_train_data = tff.simulation.datasets.shakespeare.load_data()
        local_dataset = preprocess_client_dataset(
            raw_train_data.create_tf_dataset_for_client(client_id), 
            num_epochs=LOCAL_EPOCHS, 
            max_batches_to_use=max_batches_to_use
        )
        
        # Check if the dataset is empty after limiting
        try:
            _ = next(iter(local_dataset))
            print(f"Client {client_id}: Local dataset retrieved and limited to {max_batches_to_use} batches.")
        except StopIteration:
            print(f"Client {client_id}: Local dataset is empty after limiting batches. Skipping training and upload.")
            return # Exit the function, as there's no data to train on

        # 4. Train locally
        print(f"Client {client_id}: Starting local training for {LOCAL_EPOCHS} epoch(s)...")
        history = local_model.fit(local_dataset, epochs=LOCAL_EPOCHS, verbose=1) 
        local_loss = history.history['loss'][-1]
        local_accuracy = history.history['sparse_categorical_accuracy'][-1]
        print(f"Client {client_id}: Local training complete. Loss={local_loss:.4f}, Acc={local_accuracy:.4f}")

        # 5. Calculate update (delta) and apply Differential Privacy
        local_weights = get_model_weights(local_model)
        updates = [
            local_w - server_w for local_w, server_w in zip(local_weights, server_weights)
        ]
        
        # --- Malicious Behavior: Invert updates ---
        if is_malicious:
            print(f"Client {client_id}: Applying malicious update inversion!")
            updates = [u * -1 for u in updates] # Invert the updates

        clipped_updates = clip_l2_norm(updates, L2_NORM_CLIP)
        dp_updates = add_gaussian_noise(clipped_updates, DP_NOISE_MULTIPLIER, L2_NORM_CLIP)
        
        print(f"Client {client_id}: Updates calculated, clipped, and noised for DP.")

        # 6. Upload noised updates
        upload_payload = {
            'client_id': client_id,
            'updates': updates_to_json(dp_updates),
            'metrics': {'loss': float(local_loss), 'accuracy': float(local_accuracy)}
        }
        print(f"Client {client_id}: Attempting to upload (noised) updates to {server_url}/upload_weights")
        response = requests.post(f"{server_url}/upload_weights", json=upload_payload)
        response.raise_for_status()
        print(f"Client {client_id}: (Noised) updates uploaded successfully. Server response: {response.json()}")

    except requests.exceptions.ConnectionError as e:
        print(f"Client {client_id}: Connection error to server: {e}. Make sure the server is running at {server_url}")
    except requests.exceptions.RequestException as e:
        print(f"Client {client_id}: HTTP Request error: {e}. Response text: {e.response.text if e.response else 'N/A'}")
    except ValueError as e:
        print(f"Client {client_id}: Data or model error: {e}")
    except Exception as e:
        print(f"Client {client_id}: An unexpected error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
