INSTALL=true


### DO NO MODIFY BELOW
if [ "$INSTALL" = true ] ; then
    echo INSTALLING PACKAGES
    pip install -r requirements.txt
fi

echo RUNNING TRANSLATION
python3 src/pipeline.py