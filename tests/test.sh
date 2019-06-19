#!/bin/bash
set -ex

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

net="${SCRIPTDIR}/../ocrd_typegroups_classifier/models/classifier.tgc"

cd "$SCRIPTDIR/assets/pembroke_werke_1766/data"
ocrd-typegroups-classifier \
    -l DEBUG \
    -g PHYS_0011 \
    -m mets.xml \
    -I DEFAULT \
    -O OCR-D-FONTIDENT \
    -p <(echo '{"network": "'"$net"'", "stride":143}')

expected_fontfamily='fraktur'
if ! egrep -q "<pc:TextStyle fontFamily=\"$expected_fontfamily.*" OCR-D-FONTIDENT/OCR-D-FONTIDENT_FILE_0010_DEFAULT; then
    echo 'Bad classification result:'
    grep '<pc:TextStyle fontFamily=' OCR-D-FONTIDENT/OCR-D-FONTIDENT_FILE_0010_DEFAULT
    exit 1
else
    echo 'Good classification result:'
    grep '<pc:TextStyle fontFamily=' OCR-D-FONTIDENT/OCR-D-FONTIDENT_FILE_0010_DEFAULT
fi
