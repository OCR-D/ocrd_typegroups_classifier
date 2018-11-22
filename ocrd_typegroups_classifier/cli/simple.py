"""
OCR-D conformant command line interface
"""
import sys
from ..classifier import TypegroupsClassifier

def cli():
    """
    Run on sys.args
    """
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Syntax: %s network-file image-file [stride]' % sys.argv[0])
        quit(1)
    network_file = sys.argv[1]
    image_file = sys.argv[2]
    stride = sys.argv[3] if len(sys.argv) > 3 else 96
    classifier = TypegroupsClassifier()
    classifier.run(network_file, image_file, stride)
