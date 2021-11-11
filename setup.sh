# ----------------------------------------------------------------------
# Main script for textbook translation
#
# Jeudi 11 Novembre 2021
# @author :jeremylhour
# ----------------------------------------------------------------------

INSTALL=false


##### DO NO MODIFY BELOW #####
if [ "$INSTALL" = true ] ; then
    echo INSTALLING PACKAGES
    pip install -r requirements.txt
fi

echo RUNNING TRANSLATION
echo WARNING : ALL FILES TO BE TRANSLATED NEED TO BE UPLOADED TO data/raw/
for entry in data/raw/*
do
  echo CURRENTLY PROCESSING "$(basename $entry)"
  python3 src/pipeline.py "$(basename $entry)"
done