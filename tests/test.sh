#!/bin/bash
set -ex

SCRIPTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

net="${SCRIPTDIR}/../ocrd_typegroups_classifier/models/classifier-1.tgc"

cd "$SCRIPTDIR/assets/pembroke_werke_1766/data"
ocrd-typegroups-classifier \
    -l DEBUG \
    -g FILE_0010_DEFAULT \
    -m mets.xml \
    -I DEFAULT \
    -O "OCR-D-FONTIDENT" \
    -p <(echo '{"network": "'"$net"'", "stride":143}')
