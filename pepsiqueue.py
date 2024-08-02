import os
import sys
from flask import Flask, request, jsonify
import boto3
import json

app = Flask(__name__)

# Set up AWS credentials

# Initialize SQS client
sqs = boto3.client('sqs')
queue_url = 'https://sqs.us-east-1.amazonaws.com/220959411709/pepsiqueue.fifo'

@app.route('/generate', methods=['POST'])
def generate_image_request():
    data = request.json
    user_input = data.get('prompt', '')
    uuid = data.get('uuid', '')

    if not user_input or not uuid:
        return jsonify({'error': 'No prompt or UUID provided'}), 400

    message = {
        'prompt': user_input,
        'uuid': uuid
    }

    # Send message to SQS queue
    sqs.send_message(
        QueueUrl=queue_url,
        MessageBody=json.dumps(message),
        MessageGroupId='image-generation-group'
    )

    return jsonify({'message': 'Request received and added to queue'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4000)
