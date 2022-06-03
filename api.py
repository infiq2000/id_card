import os
import json
import time
import threading
import cv2
from queue import Empty, Queue
from flask import Flask, request
from util import NumpyEncoder, byte_to_image, filename_gen
# Import model here
from kyc_model import KYCModel

app = Flask(__name__)
# Change model here
kyc_model = KYCModel("model/frozen_model.pb")
requestQueue = Queue()
CHECK_INTERVAL = 1
BATCH_SIZE = 10
BATCH_TIMEOUT = 2


def request_handler():
    while True:
        batch = []
        while not (
                len(batch) > BATCH_SIZE or
                (len(batch) > 0 and time.time() - batch[0]['time'] > BATCH_TIMEOUT)
        ):
            try:
                batch.append(requestQueue.get(timeout=CHECK_INTERVAL))
            except Empty:
                continue
        for req in batch:
            # Model call here
            out_img, out_heatmap, out_status = kyc_model.infer_one_img(req['image'])
            out = {'cropped_img': out_img, 'heatmap': out_heatmap, "status": out_status}
            req['output'] = out


threading.Thread(target=request_handler).start()

@app.route('/isAlive', methods=['GET'])
def is_alive():
    return {'responses': "Alive"}

@app.route('/imageSegment', methods=['POST'])
def crop_image():
    file = request.files['image']
    img = byte_to_image(file)
    data = {'image': img, 'time': time.time()}
    requestQueue.put(data)
    response = {"filename": ''}
    count = 10
    while 'output' not in data and count > 0:
        time.sleep(CHECK_INTERVAL)
    if data['output']['status'] == 0:
        file_name = filename_gen()
        response = {"filename": file_name}
        count -= 1
    return json.dumps(response, cls=NumpyEncoder)


if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)
