"""
Wrap TypegroupsClassifier as an ocrd.Processor
"""

from ocrd import Processor
from ocrd.model.ocrd_page import (
    from_file,
    to_xml
)

from .typegroups_classifier import TypegroupsClassifier
from .constants import OCRD_TOOL

from ocrd.utils import getLogger


class TypegroupsClassifierProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-typegroups-classifier']
        kwargs['version'] = OCRD_TOOL['version']
        super(TypegroupsClassifierProcessor, self).__init__(*args, **kwargs)
        self.log = getLogger('ocrd_typegroups_classifier')

    def process(self):
        network_file = self.parameter['network']
        stride = self.parameter['stride']
        classifier = TypegroupsClassifier.load(network_file)
        
        ignore_type = ('Adornment', 'Book covers and other irrelevant data', 'Empty Pages', 'Woodcuts - Engravings')
        
        self.log.debug('Processing: ', self.input_files)
        for (_, input_file) in enumerate(self.input_files):
            pcgts = from_file(self.workspace.download_file(input_file))
            image_url = pcgts.get_Page().imageFilename
            pil_image = self.workspace.resolve_image_as_pil(image_url)
            result = classifier.run(pil_image, stride)
            score_sum = 0
            for typegroup in classifier.classMap.cl2id:
                if not typegroup in ignore_type:
                    score_sum += max(0, result[typegroup])
            
            script_highscore = 0
            noise_highscore = 0
            result_map = {}
            output = ''
            for typegroup in classifier.classMap.cl2id:
                score = result[typegroup]
                if typegroup in ignore_type:
                    noise_highscore = max(noise_highscore, score)
                else:
                    script_highscore = max(script_highscore, score)
                    normalised_score = max(0, score / score_sum)
                    result_map[normalised_score] = typegroup
            if noise_highscore>script_highscore:
                pcgts.get_Page().set_primaryScript(None)
                self.log.debug('Detected only noise (such as empty page or book cover)')
            else:
                for k in sorted(result_map, reverse=True):
                    if output!='':
                        output = '%s, ' % output
                    output = '%s%s:%d' % (output, result_map[k], round(100*k))
                pcgts.get_Page().set_primaryScript(output)
                self.log.debug('Detected %s' % output)
