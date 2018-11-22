from __future__ import print_function

import torch
import torch.utils.data
from torchvision import transforms

from ocrd.utils import getLogger

from .constants import classes
from .vraec import vraec18

log = getLogger('ocrd_typegroups_classifier')

class TypegroupsClassifier():

    def __init__(self, network_file, stride):
        """
        Create a new classifier

        Arguments:
            network_file (string): Path to a network file
            stride (number): Stride applied to the CNN on the image. Should be between 1 and 224. Smaller values increase the computation time.
        """
        self.network_file = network_file
        self.stride = stride
        log.debug('Using network: %s', network_file)
        log.debug('Using stride: %s', stride)

    def run(self, pil_image):
        """
        Classifiy types on an image

        Arguments:
            pil_image (PIL.Image): PIL image
        """

        log.debug('Loading image...')

        log.debug('Loading network...')
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        vraec = vraec18(layer_size=96, output_channels=8)
        vraec.load_state_dict(torch.load(self.network_file, map_location='cpu'))
        vraec.to(dev)
        for l in range(2, 6):
            vraec.set_variational(l, False)

        tensorize = transforms.ToTensor()
        batch_size = 64 if torch.cuda.is_available() else 2
        nb_classes = 8
        score = torch.zeros(1, nb_classes).to(dev)
        processed_samples = 0
        batch = []
        with torch.no_grad():
            for x in range(0, pil_image.size[0], self.stride):
                for y in range(0, pil_image.size[1], self.stride):
                    crop = tensorize(pil_image.crop((x, y, x+224, y+224)))
                    batch.append(crop)
                    if len(batch) >= batch_size:
                        tensors = torch.stack(batch).to(dev)
                        out, _ = vraec(tensors)
                        score += out.sum(0)
                        processed_samples += len(batch)
                        batch = []
            if batch:
                tensors = torch.stack(batch).to(dev)
                out, _ = vraec(tensors)
                score += out.sum(0)
                processed_samples += len(batch)
                batch = []
        ssum = 0
        for k in classes:
            ssum += score[0, k]

        conf = {}
        for k in classes:
            conf[score[0, k]] = classes[k]

        # Result generation
        # TODO: output it correctly, not as a string output to stdio
        result = 'result'
        for c in sorted(conf, reverse=True):
            result = '%s:%s=%2.2f' % (result, conf[c], 100 / ssum * c)
        log.debug(result)
        return result
