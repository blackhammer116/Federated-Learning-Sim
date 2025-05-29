import json
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import requests
import sys
import time
import os
import random # Import random for potential future client-side randomness

# --- Global Constants (must match server.py for consistency) ---
SEQUENCE_LENGTH = 100
BATCH_SIZE = 10
LOCAL_EPOCHS = 1 

# --- Security Constants for Differential Privacy Simulation (must match server.py) ---
L2_NORM_CLIP = 1.0 
DP_NOISE_MULTIPLIER = 0.1 

# Global variables for vocabulary (will be populated during data loading)
char_to_id = {}
id_to_char = {}
vocab = []
VOCAB_SIZE = 0

# --- Helper Functions (copied directly from server.py for consistency) ---

def to_ids(text):
    """Converts a string of characters to a list of character IDs."""
    decoded_text = text.numpy().decode('utf-8')
    ids = [char_to_id.get(c, char_to_id['<unk>']) for c in decoded_text]
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
        print(f"Client: Limiting dataset to {max_batches_to_use} batches.")

    return processed_dataset

def load_client_data(client_id, max_batches_to_use=None):
    """
    Loads the Shakespeare dataset, builds vocabulary, and extracts data for a specific client.
    Can limit the number of batches used for this client.
    """
    global char_to_id, id_to_char, vocab, VOCAB_SIZE

    print(f"Client {client_id}: Loading Shakespeare dataset and building vocabulary...")
    train_data, _ = tff.simulation.datasets.shakespeare.load_data()

    # Build vocabulary from a representative subset of client data (consistent with server)
    all_chars_set = set()
    for cid in train_data.client_ids[:100]: # Use first 100 clients to build vocab, same as server
        client_tf_dataset = train_data.create_tf_dataset_for_client(cid)
        for sample in client_tf_dataset:
            snippets_tensor = sample['snippets']
            if snippets_tensor.dtype == tf.string:
                numpy_snippets = snippets_tensor.numpy()
                if not isinstance(numpy_snippets, np.ndarray):
                    numpy_snippets = np.array(numpy_snippets)

                if numpy_snippets.ndim == 0:
                    text_content = tf.compat.as_text(numpy_snippets.item()) 
                    all_chars_set.update(text_content)
                else:
                    for snippet_bytes in numpy_snippets:
                        text_content = tf.compat.as_text(snippet_bytes)
                        all_chars_set.update(text_content)
            else:
                print(f"Client {client_id}: Warning: Client {cid} has unexpected dtype for 'snippets': {snippets_tensor.dtype}. Skipping.")

    vocab = sorted(list(all_chars_set))
    vocab = ['<pad>', '<unk>'] + vocab
    char_to_id = {char: i for i, char in enumerate(vocab)}
    id_to_char = {i: char for i, char in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)
    print(f"Client {client_id}: Vocabulary size: {VOCAB_SIZE}")

    # Get data for the specific client_id
    if client_id not in train_data.client_ids:
        raise ValueError(f"Client ID '{client_id}' not found in the dataset.")
    
    client_dataset = preprocess_client_dataset(
        train_data.create_tf_dataset_for_client(client_id), 
        num_epochs=LOCAL_EPOCHS,
        max_batches_to_use=max_batches_to_use # Pass the limit here
    )
    print(f"Client {client_id}: Data loaded and preprocessed.")
    return client_dataset

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
            # converted_w = np.array(w_list)
            converted_w = np.array(w_list, dtype=np.float32)
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

# --- Main Client Execution Logic ---
def run_client(client_id, server_url, simulated_delay_seconds=0, max_batches_to_use=None):
    """
    Main function to run a single federated learning client.
    Includes simulated delay and data volume limits.
    """
    print(f"Client {client_id}: Starting with delay={simulated_delay_seconds}s, max_batches={max_batches_to_use}...")
    
    try:
        # Load and preprocess data for this specific client
        response = requests.get(f"{server_url}/get_client_behavior")
        max_batches_to_use = response.json()['max_batches']
        local_dataset = load_client_data(client_id, max_batches_to_use=max_batches_to_use)
        
        # --- Crucial Check: Ensure dataset is not empty ---
        temp_dataset_for_check = local_dataset.take(1)
        try:
            next(iter(temp_dataset_for_check))
            print(f"Client {client_id}: Local dataset ready for training.")
        except StopIteration:
            print(f"Client {client_id}: Local dataset is empty or has no usable batches after preprocessing. Skipping training and upload.")
            sys.exit(0) # Exit cleanly if no data
        # --- End of Crucial Check ---

        # 1. Download global model weights
        print(f"Client {client_id}: Attempting to download global model weights from {server_url}/download_model")
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

        # 3. Train locally
        print(f"Client {client_id}: Starting local training for {LOCAL_EPOCHS} epoch(s)...")
        history = local_model.fit(local_dataset, epochs=LOCAL_EPOCHS, verbose=1) 
        local_loss = history.history['loss'][-1]
        local_accuracy = history.history['sparse_categorical_accuracy'][-1]
        print(f"Client {client_id}: Local training complete. Loss={local_loss:.4f}, Acc={local_accuracy:.4f}")

        # 4. Calculate update (delta) and apply Differential Privacy
        local_weights = get_model_weights(local_model)
        updates = [
            local_w - server_w for local_w, server_w in zip(local_weights, server_weights)
        ]
        
        # Apply L2 norm clipping to the updates
        clipped_updates = clip_l2_norm(updates, L2_NORM_CLIP)
        
        # Add Gaussian noise to the clipped updates for differential privacy
        dp_updates = add_gaussian_noise(clipped_updates, DP_NOISE_MULTIPLIER, L2_NORM_CLIP)
        
        print(f"Client {client_id}: Updates calculated, clipped, and noised for DP.")

        # 5. Upload noised updates
        upload_payload = {
            'client_id': client_id,
            'updates': updates_to_json(dp_updates), # Sending 'updates' now
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
    # When running client.py directly, it expects client_id and server_url.
    # The simulated_delay_seconds and max_batches_to_use will default to 0 and None respectively
    # unless you explicitly modify the command below for manual testing of heterogeneity.
    # For server-driven simulation, these parameters are passed by the server.
    if len(sys.argv) != 3:
        print("Usage: python client.py <client_id> <server_url>")
        print("Example: python client.py CLIENT_0 http://127.0.0.1:5000")
        sys.exit(1)

    client_id = sys.argv[1]
    server_url = sys.argv[2]
    run_client(client_id, server_url)
