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

python3 src/pipeline.py