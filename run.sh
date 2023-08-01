TYPE=$1
DATASET_GROUP=$2
MODEL=$3

model_name=${MODEL##*/}
echo $model_name

PROJECT_NAME="evaluate_lm_tese"
RUN_NAME="${DATASET_GROUP}_${TYPE}_focus_${model_name}"

export WANDB_PROJECT=${PROJECT_NAME}
export WANDB_TAGS=${RUN_NAME},${model_name},${DATASET_GROUP},${TYPE}
HYP_PATH="./hypsearch/${TYPE}"
mkdir -p /workspace/models/${PROJECT_NAME}
datalawyer_tune tune huggingface_script \
  ${HYP_PATH}/tune-${DATASET_GROUP}.yaml \
  ${HYP_PATH}/hparams-${DATASET_GROUP}-focus-without-model.json \
  --optuna-param-path ${HYP_PATH}/config.json \
  --serialization-dir /workspace/models/${PROJECT_NAME}/${model_name}/${DATASET_GROUP}_${TYPE}_focus \
  --metrics "eval_f1" \
  --study-name ${RUN_NAME} \
  --direction maximize \
  --skip-if-exists \
  --storage sqlite:////workspace/models/${PROJECT_NAME}/db.sqlite3 \
  --overrides "{\"model_name_or_path\": \"${MODEL}\"}" \
  --skip-exception \
  --delete-checkpoints
