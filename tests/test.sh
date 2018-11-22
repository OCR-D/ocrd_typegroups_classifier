#!/bin/bash
export assets="$PWD/assets"
export workspace_dir="/tmp/test-ocrd_typegroups_classifier"

BINDIR=NOT_SET_YET
export PATH="${BINDIR}:${PATH}"


# Get some data
mkdir -p "$assets"
if [ ! -f "$assets/85249078X-0010.tif" ]
then
    wget -O "$assets/85249078X-0010.tif" "https://content.staatsbibliothek-berlin.de/dc/85249078X-0010/full/full/0/default.tif"
fi

# Init workspace
rm -rf "$workspace_dir"
ocrd workspace init "$workspace_dir"
ocrd workspace -d "$workspace_dir" add -G OCR-D-IMG -i orig -m image/jpg "$assets/85249078X-0010.tif"

PATH="$PATH:/home/ms/Documents/ocr-d/publish/15th-classifier/tg15th/local/bin"

SHAREDIR=NOT_SET_YET
net="${SHAREDIR}/network-epoch-99-settings-011.pth"
echo "{\"network\": \"$net\", \"stride\":143}" > test-params.json

ocrd_typegroups_classifier \
    -m "$workspace_dir"/mets.xml \
    -I OCR-D-IMG \
    -w "$workspace_dir" \
    -O "output-group" \
    -p "$(pwd)/test-params.json"
