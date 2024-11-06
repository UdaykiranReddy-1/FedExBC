import hashlib
import time
import json
import torch
import random
from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("Federation", "weights")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class Block:
    def __init__(self, index, previous_hash, model_weights, timestamp=None):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp or time.time()
        self.model_weights = model_weights
        self.hash = self.compute_hash()

    def compute_hash(self):
        """
        A function that returns the hash of the block contents.
        """
        block_string = json.dumps({
            "index": self.index,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "model_weights": str(self.model_weights),
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Generates the first block in the chain.
        """
        genesis_block = Block(0, "0", {})
        self.chain.append(genesis_block)

    def add_block(self, model_weights):
        """
        Adds a new block containing model weights to the chain.
        """
        previous_block = self.chain[-1]
        new_block = Block(index=previous_block.index + 1,
                          previous_hash=previous_block.hash,
                          model_weights=model_weights)
        self.chain.append(new_block)

    def is_chain_valid(self):
        """
        Validates the integrity of the blockchain.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check if the current block's hash is correct
            if current.hash != current.compute_hash():
                return False

            # Check if the current block's previous_hash matches the hash of the previous block
            if current.previous_hash != previous.hash:
                return False
        return True

    def get_chain_data(self):
        """
        Returns the blockchain data as a list of dictionaries.
        """
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                'previous_hash': block.previous_hash,
                'timestamp': block.timestamp,
                'hash': block.hash,
                'model_weights': str(block.model_weights)
            })
        return chain_data


# Initialize the blockchain
blockchain = Blockchain()


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint to upload a .pth file and add it to the blockchain.
    """
    if 'model' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['model']

    # Check if the file is a .pth file
    if file.filename == '' or not file.filename.endswith('.pth'):
        return jsonify({'error': 'File must be a .pth file'}), 400

    # Save the file
    file_path = os.path.join(UPLOAD_FOLDER, str(random.randint(10, 100000)) + ".pth")
    file.save(file_path)

    # Load the model weights
    # try:
    #     model_weights = torch.load(file_path)
    # except Exception as e:
    #     return jsonify({'error': f'Failed to load model weights: {str(e)}'}), 500

    # Add the model weights to the blockchain
    blockchain.add_block(file)
    return jsonify({'message': f'Model weights from {file.filename} added to blockchain'}), 200


@app.route('/download', methods=['GET'])
async def send_global_model_route():
    if os.path.exists(os.path.join("Federation", "federated_model.pth")):
        return send_file(os.path.join("Federation", "federated_model.pth"))
    else:
        return send_file(os.path.join("PreTraining", "pre_trained_model.pth"))


@app.route('/chain', methods=['GET'])
def get_chain():
    """
    Endpoint to retrieve the blockchain data.
    """
    chain_data = blockchain.get_chain_data()
    return jsonify(chain_data), 200


@app.route('/validate', methods=['GET'])
def validate_chain():
    """
    Endpoint to validate the blockchain integrity.
    """
    is_valid = blockchain.is_chain_valid()
    return jsonify({'is_valid': is_valid}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
