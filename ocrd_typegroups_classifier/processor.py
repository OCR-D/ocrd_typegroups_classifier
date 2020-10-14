"""
Wrap TypegroupsClassifier as an ocrd.Processor
"""

from ocrd import Processor
from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import (
    to_xml,

    TextStyleType
)
from ocrd_modelfactory import page_from_file

from .typegroups_classifier import TypegroupsClassifier
from .constants import OCRD_TOOL


class TypegroupsClassifierProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-typegroups-classifier']
        kwargs['version'] = OCRD_TOOL['version']
        super(TypegroupsClassifierProcessor, self).__init__(*args, **kwargs)

    def process(self):
        log = getLogger('ocrd_typegroups_classifier')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        network_file = self.parameter['network']
        stride = self.parameter['stride']
        classifier = TypegroupsClassifier.load(network_file)

        ignore_type = ('Adornment', 'Book covers and other irrelevant data',
                       'Empty Pages', 'Woodcuts - Engravings')

        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            log.info('Processing: %d / %s', n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page = pcgts.get_Page()
            pil_image, _, image_info = self.workspace.image_from_page(page, page_id)
            # todo: use image_info.resolution
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
            if noise_highscore > script_highscore:
                page.set_primaryScript(None)
                log.warning(
                    'Detected only noise on page %s. noise_highscore=%s > script_highscore=%s',
                    page_id, noise_highscore, script_highscore)
            else:
                for k in sorted(result_map, reverse=True):
                    intk = round(100*k)
                    if intk<=0:
                        continue
                    if output != '':
                        output = '%s, ' % output
                    output = '%s%s:%d' % (output, result_map[k], intk)
                log.debug('Detected %s' % output)
                textStyle = page.get_TextStyle()
                if not textStyle:
                    textStyle = TextStyleType()
                    page.set_TextStyle(textStyle)
                textStyle.set_fontFamily(output)
                file_id = make_file_id(input_file, self.output_file_grp)
                self.workspace.add_file(
                    ID=file_id,
                    file_grp=self.output_file_grp,
                    pageId=input_file.pageId,
                    mimetype=MIMETYPE_PAGE,
                    local_filename="%s/%s.xml" % (self.output_file_grp, file_id),
                    content=to_xml(pcgts)
                )
