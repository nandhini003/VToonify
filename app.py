from flask import Flask, request, jsonify
from google.cloud import storage
from vtoonify_model import Model
import numpy as np
from PIL import Image
from io import BytesIO
import torch

app = Flask(__name__)
storage_client = storage.Client()
model = Model(device='cuda' if torch.cuda.is_available() else 'cpu')
model.load_model('cartoon1')

@app.route('/image_toonify', methods=['POST'])
def image_toonify():
    input_image_path = request.json['input_image_path']
    instyle = 'default_instyle'
    exstyle = 'default_exstyle'
    style_degree = 0.5
    style_type = 'cartoon1'

    # Read the image from GCS
    bucket_name, blob_name = input_image_path.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    input_image = np.array(Image.open(BytesIO(blob.download_as_bytes())))

    result_face = model.image_toonify(input_image, instyle, exstyle, style_degree, style_type)
    output_path = request.json['output_image_path']

    # Save the result to GCS
    output_bucket_name, output_blob_name = output_path.replace("gs://", "").split("/", 1)
    output_bucket = storage_client.bucket(output_bucket_name)
    output_blob = output_bucket.blob(output_blob_name)
    output_blob.upload_from_string(Image.fromarray(result_face).tobytes())

    return jsonify({'message': 'Image saved to {}'.format(output_path)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)