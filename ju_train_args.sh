PROJECT_NAME="evaluate_lm_tese"
RUN_NAME="ju_${1}_focus_v2"
TYPE=$1
export CUDA_VISIBLE_DEVICES=3
export WANDB_PROJECT=${PROJECT_NAME}
export WANDB_TAGS=${RUN_NAME}
HYP_PATH="./hypsearch/${TYPE}"
mkdir -p /workspace/models/${PROJECT_NAME}
datalawyer_tune tune huggingface_script \
  ${HYP_PATH}/tune.yaml \
  ${HYP_PATH}/hparams-focus-v2.json \
  --optuna-param-path ${HYP_PATH}/config.json \
  --serialization-dir /workspace/models/${PROJECT_NAME}/${RUN_NAME} \
  --metrics "eval_f1" \
  --study-name ${RUN_NAME} \
  --direction maximize \
  --skip-if-exists \
  --storage sqlite:////workspace/models/${PROJECT_NAME}/db.sqlite3 \
  --skip-exception \
  --delete-checkpoints
