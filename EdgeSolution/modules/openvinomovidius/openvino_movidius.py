import argparse
import logging as log
import sys
import datetime

import cv2
import numpy as np
from openvino.inference_engine import IECore
import json

class OpenvinoMovidius:
    def __init__(self, config = None, extension = None):
        self.device = 'MYRIAD'
        self.net = None
        self.exec_net = None
        self.input_blob = None
        self.output_blob = None
        self.config = config
        self.extension = extension
        self.labelinfo = None
        self.labels = None
        self.status = 0
        self.scored_time = datetime.datetime.strptime('2000-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')

        log.info('Creating Inference Engine')
        self.ie = IECore()
        if extension and device == 'CPU':
            log.info(f'Loading the {device} extension: {extension}')
            ie.add_extension(extension, device)

        if config and device in ('GPU', 'MYRIAD', 'HDDL'):
            log.info(f'Loading the {device} configuration: {config}')
            ie.set_config({'CONFIG_FILE': config}, device)

    def LoadModel(self, modelinfo, labelinfo = None):
    # ---------------------------Step 2. Read a model in OpenVINO Intermediate Representation or ONNX format---------------
        # args.model -> net
        self.status = 0
        log.info(f'Reading the network: {modelinfo}')
        # (.xml and .bin files) or (.onnx file)
        self.net = self.ie.read_network(model=modelinfo)

        if len(self.net.input_info) != 1:
            log.error('The sample supports only single input topologies')
            return -1

        if len(self.net.outputs) != 1 and not ('boxes' in self.net.outputs or 'labels' in self.net.outputs):
            log.error('The sample supports models with 1 output or with 2 with the names "boxes" and "labels"')
            return -1

         # ---------------------------Step 4. Loading model to the device-------------------------------------------------------
        log.info('Loading the model to the plugin')
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device)

        self.status = 1

        # ---------------------------Step 3. Configure input & output----------------------------------------------------------
        # net, args.device, args.input, args.lables

        log.info('Configuring input and output blobs')
        # Get name of input blob
        self.input_blob = next(iter(self.net.input_info))

        # Set input and output precision manually
        self.net.input_info[self.input_blob].precision = 'U8'

        if len(self.net.outputs) == 1:
            self.output_blob = next(iter(self.net.outputs))
            self.net.outputs[self.output_blob].precision = 'FP32'
        else:
            self.net.outputs['boxes'].precision = 'FP32'
            self.net.outputs['labels'].precision = 'U16'

        # Generate a label list
        self.labelinfo = labelinfo
        if labelinfo:
            with open(labelinfo, 'r') as f:
                self.labels = [line.split(',')[0].strip() for line in f]

        return 0

    def Score(self, input, inferenceMark):
        # ---------------------------Step 6. Prepare input---------------------------------------------------------------------
        if self.status == 0:
            log.warning('model has not been loaded!')
            return []

        original_image = None
        if (type(input) == str):
            log.info(f'Read image from {input}')
            original_image = cv2.imread(input)
        else:
            log.info('Convert octed-stream to array')
            original_image = np.asarray(input)
        log.info('Reading image Done.')
        
        image = original_image.copy()
        _, _, net_h, net_w = self.net.input_info[self.input_blob].input_data.shape

        if image.shape[:-1] != (net_h, net_w):
            log.warning(f'Image {input} is resized from {image.shape[:-1]} to {(net_h, net_w)}')
            image = cv2.resize(image, (net_w, net_h))

        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        # Add N dimension to transform to NCHW
        image = np.expand_dims(image, axis=0)
        # ---------------------------Step 7. Do inference----------------------------------------------------------------------
        log.info('Starting inference in synchronous mode')
        res = self.exec_net.infer(inputs={self.input_blob: image})

        # ---------------------------Step 8. Process output--------------------------------------------------------------------
        output_image = original_image.copy()
        h, w, _ = output_image.shape

        if len(self.net.outputs) == 1:
            res = res[self.output_blob]
            # Change a shape of a numpy.ndarray with results ([1, 1, N, 7]) to get another one ([N, 7]),
            # where N is the number of detected bounding boxes
            detections = res.reshape(-1, 7)
        else:
            detections = res['boxes']
            self.labels = res['labels']
            # Redefine scale coefficients
            w, h = w / net_w, h / net_h
    
        detectedObjects = []

        for i, detection in enumerate(detections):
            if len(self.net.outputs) == 1:
                _, class_id, confidence, xmin, ymin, xmax, ymax = detection
            else:
                class_id = self.labels[i]
                xmin, ymin, xmax, ymax, confidence = detection

            if confidence > 0.5:
                # label = int(self.labels[class_id]) if self.labelinfo else int(class_id)
                label = self.labels[int(class_id)] if self.labelinfo else int(class_id)

                xmin = int(xmin * w)
                ymin = int(ymin * h)
                xmax = int(xmax * w)
                ymax = int(ymax * h)

                detected = { 'type':'entity', 'entity': { 'tag': { 'value': label, 'confidence' : float(confidence) }, 'box': { 'l': xmin, 't':ymin, 'w':xmax-xmin, 'h':ymax-ymin } } }
                detectedObjects.append(detected)

                log.info(f'Found: label = {label}, confidence = {confidence:.2f}, ' f'coords = ({xmin}, {ymin}), ({xmax}, {ymax})')

                # Draw a bounding box on a output image
                if inferenceMark:
                    cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        log.info(f'ovmv - detections={len(detections)} - ovmv.scored_time={self.scored_time}')
        if len(detectedObjects)>0:
            self.scored_time = datetime.datetime.now()
            log.info('scored_time updated.')

        output_filename = 'out.bmp'
        if inferenceMark:
            cv2.imwrite(output_filename, output_image)
            log.info('Image out.bmp created!')
        
        return detectedObjects, output_filename

    def GetStatus(self):
        return self.status

    def GetScoredTime(self):
        return self.scored_time