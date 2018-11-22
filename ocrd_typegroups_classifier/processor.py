"""
Wrap TypegroupsClassifier as an ocrd.Processor
"""

from ocrd import Processor
from ocrd.model.ocrd_page import (
    from_file,
    to_xml
)

from .classifier import TypegroupsClassifier
from .constants import OCRD_TOOL

class TypegroupsClassifierProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-typegroups-classifier']
        kwargs['version'] = OCRD_TOOL['version']
        super(TypegroupsClassifierProcessor, self).__init__(*args, **kwargs)

    def process(self):
        network_file = self.parameter['network']
        stride = self.parameter['stride']
        classifier = TypegroupsClassifier(network_file, stride)
        for (_, input_file) in enumerate(self.input_files):
            pcgts = from_file(self.workspace.download_file(input_file))
            image_url = pcgts.get_Page().imageFilename
            pil_image = self.workspace.resolve_image_as_pil(image_url)
            print(classifier.run(pil_image))
