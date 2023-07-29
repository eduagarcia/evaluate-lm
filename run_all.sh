MODEL=$1

model_name=${MODEL##*/}
echo $model_name

sh run.sh cls cnj $MODEL
sh run.sh ner cnj $MODEL
sh run.sh cls ju $MODEL
sh run.sh ner ju $MODEL
