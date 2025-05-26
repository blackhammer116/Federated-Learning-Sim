import collections
import functools
import json
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import requests
import time
import sys

# --- Configuration (should match server's config) ---
SERVER_URL = "http://127.0.0.1:5000" # Address of your Flask server
SEQUENCE_LENGTH = 100
BATCH_SIZE = 10
LOCAL_EPOCHS = 1

# Global variables for client-side data and model
client_data = None
char_to_id = {}
id_to_char = {}
vocab = []
VOCAB_SIZE = 0

# --- Helper Functions (copied from server for consistency) ---

def load_global_vocab():
    """
    Loads the global vocabulary from the server.
    In a real system, this might be part of initial setup.
    For this simulation, we'll try to infer it or get it from server if available.
    """
    global char_to_id, id_to_char, vocab, VOCAB_SIZE
    print("Client: Loading Shakespeare dataset for vocabulary inference...")
    train_data, _ = tff.simulation.datasets.shakespeare.load_data()
    all_chars_set = set()
    for client_id in train_data.client_ids[:100]: # Use first 100 clients to build vocab
        client_tf_dataset = train_data.create_tf_dataset_for_client(client_id)
        # Iterate through the dataset to get actual snippets
        for sample in client_tf_dataset:
            snippets_tensor = sample['snippets']
            # Ensure it's a string tensor before processing
            if snippets_tensor.dtype == tf.string:
                # Convert the tensor of bytes to a list of Python strings
                # tf.compat.as_text handles decoding bytes to unicode strings
                numpy_snippets = snippets_tensor.numpy()
                if numpy_snippets.ndim == 0: # Scalar tensor, single bytes object
                    text_content = tf.compat.as_text(numpy_snippets)
                    all_chars_set.update(text_content)
                else: # Vector tensor, array of bytes objects
                    for snippet_bytes in numpy_snippets:
                        text_content = tf.compat.as_text(snippet_bytes)
                        all_chars_set.update(text_content)
            else:
                print(f"Client: Warning: Client {client_id} has unexpected dtype for 'snippets': {snippets_tensor.dtype}. Skipping.")

    vocab = sorted(list(all_chars_set))
    vocab = ['<pad>', '<unk>'] + vocab
    char_to_id = {char: i for i, char in enumerate(vocab)}
    id_to_char = {i: char for i, char in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)
    print(f"Client: Vocabulary size: {VOCAB_SIZE}")


def to_ids(text):
    """Converts a string of characters to a list of character IDs."""
    return [char_to_id.get(c, char_to_id['<unk>']) for c in text.numpy().decode('utf-8')]
def preprocess_client_dataset_for_client(raw_dataset, num_epochs=1):
    """
    Preprocesses a single client's Shakespeare dataset.
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
            return collections.OrderedDict(x=input_text, y=target_text)

        return dataset.map(split_input_target)

    return raw_dataset.flat_map(prepare_sequences).shuffle(buffer_size=1000).batch(
        BATCH_SIZE, drop_remainder=True
    ).repeat(num_epochs).prefetch(tf.data.AUTOTUNE)

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
        numpy_weights.append(np.array(w_list, dtype=dummy_weights[i].dtype))
    return numpy_weights

# --- Main Client Logic ---

def run_client(client_id):
    """
    Main function for a single client to run its FL cycle.
    """
    print(f"Client {client_id}: Starting FL process.")
    
    # Load global vocabulary (needed for preprocessing)
    load_global_vocab()

    # Load client's specific data
    print(f"Client {client_id}: Loading local Shakespeare data...")
    train_data, _ = tff.simulation.datasets.shakespeare.load_data()
    raw_client_dataset = train_data.create_tf_dataset_for_client(client_id)
    
    # Preprocess the client's data
    global client_data
    client_data = preprocess_client_dataset_for_client(raw_client_dataset, num_epochs=LOCAL_EPOCHS)
    
    # Check if client_data is empty after preprocessing
    try:
        next(iter(client_data))
    except tf.errors.OutOfRangeError:
        print(f"Client {client_id}: Local dataset is empty after preprocessing. Exiting.")
        return

    while True:
        try:
            print(f"Client {client_id}: Requesting global model from server...")
            response = requests.get(f"{SERVER_URL}/download_model")
            response.raise_for_status() # Raise an exception for HTTP errors
            server_weights_json = response.json()['weights']
            server_weights = json_to_weights(server_weights_json)
            print(f"Client {client_id}: Global model downloaded.")

            # Create local model and set global weights
            local_model = create_keras_model()
            set_model_weights(local_model, server_weights)
            local_model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

            # Train locally
            print(f"Client {client_id}: Starting local training for {LOCAL_EPOCHS} epochs...")
            history = local_model.fit(client_data, epochs=LOCAL_EPOCHS, verbose=0)
            local_loss = history.history['loss'][-1]
            local_accuracy = history.history['sparse_categorical_accuracy'][-1]
            print(f"Client {client_id}: Local training complete. Loss={local_loss:.4f}, Acc={local_accuracy:.4f}")

            # Upload updated weights
            updated_weights = get_model_weights(local_model)
            upload_payload = {
                'client_id': client_id,
                'weights': weights_to_json(updated_weights),
                'metrics': {'loss': float(local_loss), 'accuracy': float(local_accuracy)}
            }
            print(f"Client {client_id}: Uploading weights to server...")
            response = requests.post(f"{SERVER_URL}/upload_weights", json=upload_payload)
            response.raise_for_status()
            print(f"Client {client_id}: Weights uploaded successfully. Waiting for next round...")

            # In a real system, clients would wait for server's signal for next round.
            # Here, we'll just wait a bit before trying again.
            time.sleep(5) # Wait 5 seconds before checking for next round (polling)

        except requests.exceptions.ConnectionError:
            print(f"Client {client_id}: Server not reachable at {SERVER_URL}. Retrying in 5 seconds...")
            time.sleep(5)
        except requests.exceptions.RequestException as e:
            print(f"Client {client_id}: HTTP error during FL round: {e}. Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"Client {client_id}: An unexpected error occurred: {e}. Retrying in 5 seconds...")
            time.sleep(5)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python client.py <client_id>")
        print("Example: python client.py 'MUCH_ADO_ABOUT_NOTHING_BENEDICK'")
        # You can find client IDs from the Shakespeare dataset, e.g.,
        # train_data, _ = tff.simulation.datasets.shakespeare.load_data()
        # print(train_data.client_ids[:5])
        sys.exit(1)
    
    client_id = sys.argv[1]
    run_client(client_id)
