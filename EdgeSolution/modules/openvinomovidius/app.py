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

intervalSec = 60
inferenceMark = False
sendDetection = False
modelLoaded = False
modelName = ''

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

def parse_desired_properties_request(configSpec, ovmv, configLock):
    sendKey = 'send-telemetry'
    global sendDetection, modelLoaded
    if sendKey in configSpec:
        configLock.acquire()
        sendDetection = bool(configSpec[sendKey])
        configLock.release()
        logging.info(f'new send detection telemetry = {sendDetection}')
    uploadImageKey = 'upload'
    global intervalSec, inferenceMark
    if uploadImageKey in configSpec:
        configLock.acquire()
        if 'interval-sec' in configSpec[uploadImageKey]:
            intervalSec = int(configSpec[uploadImageKey]['interval-sec'])
        if 'inference-mark' in configSpec[uploadImageKey]:
            inferenceMark = bool(configSpec[uploadImageKey]['inference-mark'])
        configLock.release()
        logging.info(f'new interval-sec = {intervalSec}')
        logging.info(f'new inference-mark = {inferenceMark}')
    modelFileKey = 'model'
    global modelName
    if modelFileKey in configSpec:
        if modelLoaded:
            logging.info('Model update will be done at next starting.')
        else:
            modelUrl = configSpec[modelFileKey]['url']
            modelFileName = configSpec[modelFileKey]['filename']
            modelName = configSpec[modelFileKey]['name']
            labelName = None
            if 'label' in configSpec[modelFileKey]:
                labelName = configSpec[modelFileKey]['label']
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

                    configLock.acquire()
                    res = ovmv.LoadModel(modelPath, labelPath)
                    configLock.release()
                    if res == 0:
                        logging.info('model load succeeded')
                        modelLoaded = True
                    else:
                        logging.warning('model load failed')
            else:
                logging.info(f'Failed to download {modelUrl}')
        

def twin_update_listener(client, ovmv, configLock):
    while True:
        global modelName, intervalSec, inferenceMark
        patch = client.receive_twin_desired_properties_patch()  # blocking call
        logging.info(f'Twin desired properties patch updated. - {patch}')
        parse_desired_properties_request(patch, ovmv, configLock)
        logging.info(f'loaded - {modelName}')
        update_reported_properties(client, ovmv.GetStatus(), modelName)


def setup_iot(ovmv, configLock):
    try:
        global inferenceMark, intervalSec, modelName
        if not sys.version >= "3.5.3":
            raise Exception( "The sample requires python 3.5.3+. Current version of Python: %s" % sys.version )
        logging.info( "IoT Hub Client for Python" )

        # The client object is used to interact with your Azure IoT hub.
        module_client = IoTHubModuleClient.create_from_edge_environment()

        # connect the client.
        module_client.connect()
        logging.info("Connected to Edge Runtime.")

        
        currentTwin = module_client.get_twin()
        configSpec = currentTwin['desired']
        parse_desired_properties_request(configSpec, ovmv, configLock)
        logging.info(f'loaded - {modelName}')
        update_reported_properties(module_client, ovmv.GetStatus(), modelName)

        twin_update_listener_thread = threading.Thread(target=twin_update_listener, args=(module_client, ovmv, configLock))
        twin_update_listener_thread.daemon = True
        twin_update_listener_thread.start()
        logging.info('IoT Edge settings done.')

        return module_client

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
    configLock = threading.Lock()
    moduleClient = setup_iot(ovmv, configLock)
    # time_delta_sec = 60

    @app.route("/score", methods = [ 'POST'])
    async def score():
        try:
            global intervalSec, inferenceMark, sendDetection
            logging.info('received request')
            imageData = io.BytesIO(request.get_data())
            pilImage = Image.open((imageData))
            logging.info('Scoring...')

            configLock.acquire()
            nowTime = datetime.datetime.now()
            timeDelta = nowTime - ovmv.GetScoredTime()
            detectedObjects, inferencedImageFile = ovmv.Score(pilImage, inferenceMark)
            isSendTelemetry = sendDetection
            configLock.release()

            logging.info('Scored.')
            if isSendTelemetry:
                totalDtected = 0
                detected = {}
                for d in detectedObjects:
                    label = d['entity']['tag']['value']
                    if label in detected:
                        detected[label] = detected[label] + 1
                    else:
                        detected[label] = 1
                    totalDtected = totalDtected + 1
                telemetry = {'timestamp':'{0:%Y-%m-%dT%H:%M:%S.%fZ}'.format(nowTime), 'totaldetection': totalDtected, 'detected':detected }
                sendMsg = json.dumps(telemetry)
                moduleClient.send_message_to_output(sendMsg, 'detection_monitor')
                logging.info('Send detection message to detection_monitor of IoT Edge Runtime')

            if len(detectedObjects) > 0 :
                if timeDelta.seconds > intervalSec:
                    logging.info('scored long again.')
                    if fileUploader:
                        imageData = None
                        if inferenceMark:
                            imageData = open(inferencedImageFile, 'rb')
                        else:
                            imageData = io.BytesIO(request.get_data())
                        fileUploader.upload(imageData, IOTEDGE_DEVICEID, '{0:%Y%m%d%H%M%S%f}'.format(datetime.datetime.now()), pilImage.format.lower())
                        if inferenceMark:
                            imageData.close()
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
