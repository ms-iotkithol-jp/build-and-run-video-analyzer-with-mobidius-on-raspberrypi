import threading
from PIL import Image
import os
import io
import json
import logging
import linecache
import sys
import datetime
import asyncio

from flask import Flask, request, jsonify, Response
from openvino_movidius import OpenvinoMovidius
from file_uploader import FileUploader

import requests
import tarfile
from six.moves import input
from azure.iot.device import IoTHubModuleClient

def update_reported_properties(client, status, model=None):
    reportingStatus = "unknown"
    if status == 0:
        reportingStatus = "initialized"
    elif status == 1:
        reportingStatus = "model-loaded"

    current_status = {"current_status": {'status':reportingStatus, 'model':''}}
    if model:
        current_status['current_status']['model'] = model
    client.patch_twin_reported_properties(current_status)

def parse_desired_properties_request(scoringSpec, ovmv, modelLock):
    modelFileKey = 'model'
    if modelFileKey in scoringSpec:
        modelUrl = scoringSpec[modelFileKey]['url']
        modelFileName = scoringSpec[modelFileKey]['filename']
        modelName = scoringSpec[modelFileKey]['name']
        labelName = None
        if 'label' in scoringSpec[modelFileKey]:
            labelName = scoringSpec[modelFileKey]['label']
        logging.info(f'Receive request of model update - {modelFileName} - {modelUrl}')
        response = requests.get(modelUrl)
        if response.status_code == 200:
            modelFolderName = 'model'
            os.makedirs(modelFolderName, exist_ok=True)
            saveFileName = os.path.join(modelFolderName, modelFileName)
            with open(saveFileName, 'wb') as saveFile:
                saveFile.write(response.content)
                logging.info('Succeeded to download new model.')
                if modelFileName.endswith('.tgz'):
                    tar = tarfile.open(saveFileName, 'r:gz')
                    tar.extractall(modelFolderName)
                    os.remove(saveFileName)
                modelFolderPath = os.path.join(os.getcwd(), modelFolderName)
                modelPath = os.path.join(modelFolderPath, modelName)
                logging.info(f'Loading {modelPath}')
                labelPath = None
                if labelName:
                    labelPath = os.path.join(modelFolderPath, labelName)

                modelLock.acquire()
                res = ovmv.LoadModel(modelPath, labelPath)
                modelLock.release()
                if res == 0:
                    logging.info('model load succeeded')
                else:
                    logging.warning('model load failed')
        else:
            logging.info(f'Failed to download {modelUrl}')
        
        return modelName

def twin_update_listener(client, ovmv, modelLock):
    while True:
        patch = client.receive_twin_desired_properties_patch()  # blocking call
        logging.info(f'Twin desired properties patch updated. - {patch}')
        modelName = parse_desired_properties_request(patch, ovmv, modelLock)
        logging.info(f'loaded - {modelName}')
        update_reported_properties(client, ovmv.GetStatus(), modelName)


def setup_iot(ovmv, modelLock):
    try:
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        logging.info( "IoT Hub Client for Python" )

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()

        # connect the client.
        module_client.connect()
        logging.info("Connected to Edge Runtime.")

        
        currentTwin = module_client.get_twin()
        scoringSpec = currentTwin['desired']
        modelName = parse_desired_properties_request(scoringSpec, ovmv, modelLock)
        logging.info(f'loaded - {modelName}')
        update_reported_properties(module_client, ovmv.GetStatus(), modelName)

        twin_update_listener_thread = threading.Thread(target=twin_update_listener, args=(module_client, ovmv, modelLock))
        twin_update_listener_thread.daemon = True
        twin_update_listener_thread.start()
        logging.info('IoT Edge settings done.')

    except Exception as e:
        logging.warning( "Unexpected error %s " % e )
        raise

logging.basicConfig(level=logging.INFO)

async def main():
    IOTEDGE_DEVICEID=os.environ['IOTEDGE_DEVICEID']
    fileUploader = None
    if 'BLOB_ON_EDGE_MODULE' in os.environ and 'BLOB_ON_EDGE_ACCOUNT_NAME' in os.environ and 'BLOB_ON_EDGE_ACCOUNT_KEY' in os.environ and 'BLOB_CONTAINER_NAME':
        BLOB_ON_EDGE_MODULE = os.environ['BLOB_ON_EDGE_MODULE']
        BLOB_ON_EDGE_ACCOUNT_NAME = os.environ['BLOB_ON_EDGE_ACCOUNT_NAME']
        BLOB_ON_EDGE_ACCOUNT_KEY = os.environ['BLOB_ON_EDGE_ACCOUNT_KEY']
        BLOB_CONTAINER_NAME = os.environ['BLOB_CONTAINER_NAME']
        logging.info(f'Blob Service specified. BLOB_ON_EDGE_MODULE={BLOB_ON_EDGE_MODULE},BLOB_ON_EDGE_ACCOUNT_NAME={BLOB_ON_EDGE_ACCOUNT_NAME},BLOB_ON_EDGE_ACCOUNT_KEY={BLOB_ON_EDGE_ACCOUNT_KEY},BLOB_CONTAINER_NAME={BLOB_CONTAINER_NAME}')
        fileUploader = FileUploader(BLOB_ON_EDGE_MODULE, BLOB_ON_EDGE_ACCOUNT_NAME, BLOB_ON_EDGE_ACCOUNT_KEY, BLOB_CONTAINER_NAME)
        fileUploader.initialize()
        logging.info('FileUploader initialized.')

    app = Flask(__name__)
    ovmv = OpenvinoMovidius()
    # res = ovmv.LoadModel('/result/open_model_zoo/tools/downloader/intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml')
    modelLock = threading.Lock()
    setup_iot(ovmv, modelLock)
    time_delta_sec = 60

    @app.route("/score", methods = [ 'POST'])
    async def score():
        try:
            logging.info('received request')
            imageData = io.BytesIO(request.get_data())
            pilImage = Image.open((imageData))
            logging.info('Scoring...')

            modelLock.acquire()
            nowTime = datetime.datetime.now()
            timeDelta = nowTime - ovmv.GetScoredTime()
            detectedObjects = ovmv.Score(pilImage)
            modelLock.release()

            logging.info('Scored.')
            if len(detectedObjects) > 0 :
                if timeDelta.seconds > time_delta_sec:
                    logging.info('scored long again.')
                    if fileUploader:
                        imageData = io.BytesIO(request.get_data())
                        fileUploader.upload(imageData, IOTEDGE_DEVICEID, '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()), pilImage.format.lower())
                scored_time = nowTime
                respBody = {
                    'inferences' : detectedObjects
                }
                respBody = json.dumps(respBody)
                logging.info(f'Sending response - {respBody}')
                return Response(respBody, status=200, mimetype='application/json')
            else :
                logging.info('Sending empty response')
                return Response(status=204)
        except Exception as e:
            logging.error(f'exception - {e}')
            return Response(response='Exception occured while processing the image.', status=500)

    @app.route("/")
    def healty():
        return "Healthy"

    app.run(host='0.0.0.0', port=8888)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
