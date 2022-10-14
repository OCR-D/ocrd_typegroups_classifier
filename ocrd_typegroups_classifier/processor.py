"""
Wrap TypegroupsClassifier as an ocrd.Processor
"""
import os
from pkg_resources import resource_filename

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    image_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_utils import getLogger, make_file_id, MIMETYPE_PAGE
from ocrd_models.ocrd_page import (
    to_xml,
    TextStyleType
)
from ocrd_modelfactory import page_from_file

from .typegroups_classifier import TypegroupsClassifier
from .constants import OCRD_TOOL

IGNORE_TYPE = [
    'Adornment',
    'Book covers and other irrelevant data',
    'Empty Pages',
    'Woodcuts - Engravings'
]

class TypegroupsClassifierProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-typegroups-classifier']
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        if hasattr(self, 'output_file_grp'):
            # processing context
            self.setup()

    def setup(self):
        if 'network' not in self.parameter:
            self.parameter['network'] = resource_filename(__name__, 'models/densenet121.tgc')
        network_file = self.resolve_resource(self.parameter['network'])
        self.classifier = TypegroupsClassifier.load(network_file)

    def _process_segment(self, segment, image):
        LOG = getLogger('ocrd_typegroups_classifier')
        result = self.classifier.run(image, self.parameter['stride'])
        score_sum = 0
        for typegroup in self.classifier.classMap.cl2id:
            if not typegroup in IGNORE_TYPE:
                score_sum += max(0, result[typegroup])
        script_highscore = 0
        noise_highscore = 0
        result_map = {}
        for typegroup in self.classifier.classMap.cl2id:
            score = result[typegroup]
            if typegroup in IGNORE_TYPE:
                noise_highscore = max(noise_highscore, score)
            else:
                script_highscore = max(script_highscore, score)
                normalised_score = max(0, score / score_sum)
                result_map[normalised_score] = typegroup
        output = ''
        if noise_highscore > script_highscore:
            segment.set_primaryScript(None)
            LOG.warning('Detected only noise on "%s": noise_highscore=%s > script_highscore=%s',
                        segment.id, noise_highscore, script_highscore)
        else:
            for k in sorted(result_map, reverse=True):
                intk = round(100 * k)
                if intk <= 0:
                    continue
                if output != '':
                    output = '%s, ' % output
                output = '%s%s:%d' % (output, result_map[k], intk)
            LOG.debug('Detected %s on "%s"', output, segment.id)
            textStyle = segment.get_TextStyle()
            if not textStyle:
                textStyle = TextStyleType()
                segment.set_TextStyle(textStyle)
            textStyle.set_fontFamily(output)

    def process(self):
        """Classify historic script in pages and annotate as font style.

        Open and deserialize PAGE input files and their respective images
        down to the hierarchy ``level``.

        Then for each segment, retrieve the raw image and feed it to the font
        classifier. 

        Post-process detections by filtering classes and thresholding scores.
        Annotate the good predictions by name and score as a comma-separated
        list under ``./TextStyle/@fontFamily``, if any.

        Produce a new PAGE output file by serialising the resulting hierarchy.
        """
        LOG = getLogger('ocrd_typegroups_classifier')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        level = self.parameter['level']
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            LOG.info('Processing: %d / %s', n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            page_image, page_coords, image_info = self.workspace.image_from_page(
                page, page_id,
                # prefer raw image (to meet expectation of the models, which
                # have been trained on RGB images with both geometry and color
                # transform random augmentation)
                # maybe even: dewarped,deskewed ?
                feature_filter='binarized,normalized,grayscale_normalized,despeckled')
            # todo: use image_info.resolution
            if level == 'page':
                self._process_segment(page, page_image)
            else:
                for region in page.get_AllRegions(classes=['Text']):
                    region_polygon = coordinates_of_segment(region, page_image, page_coords)
                    region_image = image_from_polygon(page_image, region_polygon)
                    self._process_segment(region, region_image)
            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                mimetype=MIMETYPE_PAGE,
                local_filename=os.path.join(self.output_file_grp, file_id + '.xml'),
                content=to_xml(pcgts)
            )
