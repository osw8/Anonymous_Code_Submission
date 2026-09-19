#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${PROJECT_ROOT}}
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
PYTHON=${PYTHON:-python3}
PREPROCESS_ROOT=${PREPROCESS_ROOT:-auto}
RESULT_ROOT=${RESULT_ROOT:-auto}
CROSS_TIME_PREPROCESS_ROOT=${CROSS_TIME_PREPROCESS_ROOT:-${DATA_ROOT}/preprocessed/cross_time}
CROSS_MODAL_RESULT_ROOT=${CROSS_MODAL_RESULT_ROOT:-${DATA_ROOT}/results/cross_modal}
NEW_BASELINE_RESULT_ROOT=${NEW_BASELINE_RESULT_ROOT:-${DATA_ROOT}/results/baselines}

MISSION=${MISSION:-dataset_transfer}
GPU=${GPU:-0}
PHYSICAL_GPU=${GPU}
RUNTIME_GPU=0
RAND_SEED=${RAND_SEED:-1}
BUDGET_SEED=${BUDGET_SEED:-2026}
UNDERSAMPLE_SEED=${UNDERSAMPLE_SEED:-2026}
SOURCE_REHEARSAL_FRACTION=${SOURCE_REHEARSAL_FRACTION:-0.25}
DETERMINISTIC=${DETERMINISTIC:-1}
NUMERIC_MODE=${NUMERIC_MODE:-fp32}
FAST_ALGORITHMS=${FAST_ALGORITHMS:-0}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}
STATS_NUM_WORKERS=${STATS_NUM_WORKERS:-8}
BOOTSTRAP_RESAMPLES=${BOOTSTRAP_RESAMPLES:-2000}
if [[ -n ${EPOCH:-} ]]
then
  EPOCHS=${EPOCH}
else
  EPOCHS=${EPOCHS:-100}
fi
PATIENCE=${PATIENCE:-20}
BUDGET_EPOCHS=${BUDGET_EPOCHS:-50}
BUDGET_PATIENCE=${BUDGET_PATIENCE:-10}
BUDGET_HEAD_ONLY_EPOCHS=${BUDGET_HEAD_ONLY_EPOCHS:-5}
MIN_DELTA=${MIN_DELTA:-0.001}
INTERPRETABILITY=${INTERPRETABILITY:-0}
INTERPRETABILITY_MAX_CLIPS=${INTERPRETABILITY_MAX_CLIPS:-2000}
INTERPRETABILITY_ONLY=${INTERPRETABILITY_ONLY:-0}
USE_PRETRAINED=${USE_PRETRAINED:-auto}
DRY_RUN=${DRY_RUN:-0}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
STOP_ON_ERROR=${STOP_ON_ERROR:-0}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
PROCESS_NAME=${PROCESS_NAME:-auto}
SAMPLING_AUDIT_MODE=${SAMPLING_AUDIT_MODE:-compact}
EEGNET_PYTHON=${EEGNET_PYTHON:-${PYTHON}}

if [[ -n ${MODEL:-} ]]
then
  MODELS_TEXT=${MODEL}
else
  MODELS_TEXT=${MODELS_TEXT:-}
fi
if [[ -n ${TASK:-} ]]
then
  TASKS_TEXT=${TASK}
else
  TASKS_TEXT=${TASKS_TEXT:-}
fi
if [[ -n ${DIRECTION:-} ]]
then
  DIRECTIONS_TEXT=${DIRECTION}
else
  DIRECTIONS_TEXT=${DIRECTIONS_TEXT:-}
fi
if [[ -n ${WINDOW:-} ]]
then
  WINDOWS_TEXT=${WINDOW}
else
  WINDOWS_TEXT=${WINDOWS_TEXT:-}
fi
if [[ -n ${BUDGET:-} ]]
then
  BUDGETS_TEXT=${BUDGET}
else
  BUDGETS_TEXT=${BUDGETS_TEXT:-}
fi
SVM_C_GRID_TEXT=${SVM_C_GRID_TEXT:-0.01 0.1 1.0 10.0}
SEEDS_TEXT=${SEEDS_TEXT:-${RAND_SEED}}
EPISHIFT_CROSS_DATASET_RESULT_ROOT=${EPISHIFT_CROSS_DATASET_RESULT_ROOT:-${DATA_ROOT}/results/cross_dataset}
EPISHIFT_CROSS_MODAL_RESULT_ROOT=${EPISHIFT_CROSS_MODAL_RESULT_ROOT:-${CROSS_MODAL_RESULT_ROOT}}
EPISHIFT_ECE_BINS=${EPISHIFT_ECE_BINS:-15}
EPISHIFT_SAVE_FIGURES=${EPISHIFT_SAVE_FIGURES:-1}

if [[ -z ${MODELS_TEXT} ]]
then
  echo ERROR_MODELS_NOT_SET_USE_MODEL_OR_MODELS_TEXT
  exit 2
fi
if [[ -z ${WINDOWS_TEXT} ]]
then
  echo ERROR_WINDOWS_NOT_SET_USE_WINDOW_OR_WINDOWS_TEXT
  exit 2
fi
if [[ -z ${BUDGETS_TEXT} ]]
then
  echo ERROR_BUDGETS_NOT_SET_USE_BUDGET_OR_BUDGETS_TEXT
  exit 2
fi

LR_OVERRIDE=${LR_OVERRIDE:-none}
TARGET_LR_OVERRIDE=${TARGET_LR_OVERRIDE:-none}
TARGET_BACKBONE_LR_OVERRIDE=${TARGET_BACKBONE_LR_OVERRIDE:-none}
WEIGHT_DECAY_OVERRIDE=${WEIGHT_DECAY_OVERRIDE:-none}
MAX_GRAD_NORM_OVERRIDE=${MAX_GRAD_NORM_OVERRIDE:-none}
MRI_TEMPLATE=${MRI_TEMPLATE:-none}
EEGNET_CUDA_ROOT=${EEGNET_CUDA_ROOT:-none}
REUSE_INCOMPLETE_SOURCE_CHECKPOINT=${REUSE_INCOMPLETE_SOURCE_CHECKPOINT:-0}

BIOT_TOKEN_SIZE=${BIOT_TOKEN_SIZE:-200}
BIOT_HOP_LENGTH=${BIOT_HOP_LENGTH:-100}
CBRAMOD_OPTIMIZER=${CBRAMOD_OPTIMIZER:-AdamW}
CBRAMOD_CLIP_VALUE=${CBRAMOD_CLIP_VALUE:-1.0}
CBRAMOD_MULTI_LR=${CBRAMOD_MULTI_LR:-1}
CBRAMOD_PREFETCH_FACTOR=${CBRAMOD_PREFETCH_FACTOR:-4}
BENDR_PREFETCH_FACTOR=${BENDR_PREFETCH_FACTOR:-4}
BENDR_PROGRESS_UPDATE_INTERVAL=${BENDR_PROGRESS_UPDATE_INTERVAL:-10}
EEGPT_PREFETCH_FACTOR=${EEGPT_PREFETCH_FACTOR:-4}
EEGPT_FORWARD_CHUNK_SIZE=${EEGPT_FORWARD_CHUNK_SIZE:-64}
EEGPT_ACTIVATION_CHECKPOINTING=${EEGPT_ACTIVATION_CHECKPOINTING:-0}
STEEGFORMER_PREFETCH_FACTOR=${STEEGFORMER_PREFETCH_FACTOR:-4}
CST_RESIZE_HEADS=${CST_RESIZE_HEADS:-2}
CST_RESIZE_LAYERS=${CST_RESIZE_LAYERS:-2}
CST_RESIZE_FEEDFORWARD=${CST_RESIZE_FEEDFORWARD:-128}
CST_KD_WEIGHT=${CST_KD_WEIGHT:-1.0}
CST_MCC_WEIGHT=${CST_MCC_WEIGHT:-1.0}
CST_KD_TEMPERATURE=${CST_KD_TEMPERATURE:-4.0}
CST_MCC_TEMPERATURE=${CST_MCC_TEMPERATURE:-2.5}
EVOBRAIN_MAX_GRAD_NORM=${EVOBRAIN_MAX_GRAD_NORM:-5.0}
EVOBRAIN_RNN_UNITS=${EVOBRAIN_RNN_UNITS:-64}
EVOBRAIN_AGG=${EVOBRAIN_AGG:-max}
EVOBRAIN_TOP_K=${EVOBRAIN_TOP_K:-3}
RIVER_DESCRIPTOR_HIDDEN_DIM=${RIVER_DESCRIPTOR_HIDDEN_DIM:-64}
RIVER_DESCRIPTOR_WINDOW_SECONDS=${RIVER_DESCRIPTOR_WINDOW_SECONDS:-4.0}
RIVER_DESCRIPTOR_SLOW_SECONDS=${RIVER_DESCRIPTOR_SLOW_SECONDS:-16.0}
RIVER_LATENT_ELECTRODES=${RIVER_LATENT_ELECTRODES:-12}
RIVER_EMBED_DIM=${RIVER_EMBED_DIM:-128}
RIVER_DEPTH=${RIVER_DEPTH:-6}
RIVER_DROPOUT=${RIVER_DROPOUT:-0.0}
RIVER_DESCRIPTOR_CHANNEL_CHUNK_SIZE=${RIVER_DESCRIPTOR_CHANNEL_CHUNK_SIZE:-16384}
RIVER_CHANNEL_DROP_ENABLED=${RIVER_CHANNEL_DROP_ENABLED:-1}
RIVER_RDROP_ENABLED=${RIVER_RDROP_ENABLED:-1}
RIVER_HEADS=${RIVER_HEADS:-8}
RIVER_FF_DIM=${RIVER_FF_DIM:-384}
RIVER_FOURIER_MODES=${RIVER_FOURIER_MODES:-32}
RIVER_FOURIER_RANK=${RIVER_FOURIER_RANK:-8}
RIVER_LOCAL_KERNEL_SIZE=${RIVER_LOCAL_KERNEL_SIZE:-5}
RIVER_SLOT_TEMPERATURE=${RIVER_SLOT_TEMPERATURE:-0.50}
RIVER_EVIDENCE_SMOOTHING_KERNEL=${RIVER_EVIDENCE_SMOOTHING_KERNEL:-3}
RIVER_SINKHORN_ITERATIONS=${RIVER_SINKHORN_ITERATIONS:-4}
RIVER_TRANSPORT_EPSILON=${RIVER_TRANSPORT_EPSILON:-0.20}
RIVER_ELECTRODE_MASS_RELAXATION=${RIVER_ELECTRODE_MASS_RELAXATION:-0.80}
RIVER_ELECTRODE_MASS_TEMPERATURE=${RIVER_ELECTRODE_MASS_TEMPERATURE:-0.50}
RIVER_POOLING_TEMPERATURE=${RIVER_POOLING_TEMPERATURE:-0.70}
RIVER_TRANSPORT_MODE=${RIVER_TRANSPORT_MODE:-temporal_uot}
RIVER_TEMPORAL_MIXER=${RIVER_TEMPORAL_MIXER:-spectral_local}
RIVER_STATE_ENCODER_MODE=${RIVER_STATE_ENCODER_MODE:-descriptor}
RIVER_DIVERSITY_WEIGHT=${RIVER_DIVERSITY_WEIGHT:-0.01}
RIVER_ALIGNMENT_WEIGHT=${RIVER_ALIGNMENT_WEIGHT:-0.05}
RIVER_MONTAGE_CONSISTENCY_WEIGHT=${RIVER_MONTAGE_CONSISTENCY_WEIGHT:-0.02}
RIVER_CHANNEL_DROP_RATIO=${RIVER_CHANNEL_DROP_RATIO:-0.15}
RIVER_CHANNEL_DROP_START_RATIO=${RIVER_CHANNEL_DROP_START_RATIO:-0.05}
RIVER_TEMPORAL_PATCH_MASK_ENABLED=${RIVER_TEMPORAL_PATCH_MASK_ENABLED:-0}
RIVER_TEMPORAL_PATCH_MASK_RATIO=${RIVER_TEMPORAL_PATCH_MASK_RATIO:-0.10}
RIVER_TEMPORAL_PATCH_MASK_BLOCKS=${RIVER_TEMPORAL_PATCH_MASK_BLOCKS:-1}
RIVER_TEMPORAL_PATCH_MASK_MIN_KEEP_RATIO=${RIVER_TEMPORAL_PATCH_MASK_MIN_KEEP_RATIO:-0.75}
RIVER_PROTOTYPE_MOMENTUM=${RIVER_PROTOTYPE_MOMENTUM:-0.95}
RIVER_RDROP_WEIGHT=${RIVER_RDROP_WEIGHT:-0.1}
RIVER_PRECISION=${RIVER_PRECISION:-fp32}
RIVER_EMA_DECAY=${RIVER_EMA_DECAY:-0.999}
RIVER_WARMUP_RATIO=${RIVER_WARMUP_RATIO:-0.05}
RIVER_MIN_LR_RATIO=${RIVER_MIN_LR_RATIO:-0.01}
RIVER_PROGRESS_UPDATE_INTERVAL=${RIVER_PROGRESS_UPDATE_INTERVAL:-10}
export RIVER_PROGRESS_UPDATE_INTERVAL
export RIVER_PRECISION
RIVER_MASS_AWARE_TRANSPORT=${RIVER_MASS_AWARE_TRANSPORT:-1}
RIVER_SELECTED_MASS_AWARE_TRANSPORT=${RIVER_MASS_AWARE_TRANSPORT}
RIVER_RESIDUAL_BOTTLENECK_DIM=${RIVER_RESIDUAL_BOTTLENECK_DIM:-32}
RIVER_SLOT_COMPETITION_STRENGTH=${RIVER_SLOT_COMPETITION_STRENGTH:-0.65}
RIVER_SLOT_COMPETITION_TEMPERATURE=${RIVER_SLOT_COMPETITION_TEMPERATURE:-0.35}
RIVER_FUSE_CONSISTENCY_VIEWS=${RIVER_FUSE_CONSISTENCY_VIEWS:-1}
RIVER_FUSE_AUGMENTATION_VIEWS=${RIVER_FUSE_AUGMENTATION_VIEWS:-1}
export RIVER_MASS_AWARE_TRANSPORT
export RIVER_RESIDUAL_BOTTLENECK_DIM
export RIVER_SLOT_COMPETITION_STRENGTH
export RIVER_SLOT_COMPETITION_TEMPERATURE
export RIVER_FUSE_CONSISTENCY_VIEWS
export RIVER_CHANNEL_DROP_ENABLED
export RIVER_TEMPORAL_PATCH_MASK_ENABLED
export RIVER_TEMPORAL_PATCH_MASK_RATIO
export RIVER_TEMPORAL_PATCH_MASK_BLOCKS
export RIVER_TEMPORAL_PATCH_MASK_MIN_KEEP_RATIO
export RIVER_FUSE_AUGMENTATION_VIEWS
export RIVER_RDROP_ENABLED
export RIVER_RDROP_WEIGHT
export RIVER_STATE_ENCODER_MODE
SVM_MAX_ITER=${SVM_MAX_ITER:-10000}
RF_N_ESTIMATORS=${RF_N_ESTIMATORS:-500}
RF_MAX_DEPTH=${RF_MAX_DEPTH:-none}
RF_MAX_FEATURES=${RF_MAX_FEATURES:-sqrt}
RF_MIN_SAMPLES_LEAF=${RF_MIN_SAMPLES_LEAF:-2}
CMMN_C=${CMMN_C:-1.0}
CMMN_MAX_ITER=${CMMN_MAX_ITER:-1000}

read -r -a MODELS <<< ${MODELS_TEXT}
read -r -a WINDOWS <<< ${WINDOWS_TEXT}
read -r -a BUDGETS <<< ${BUDGETS_TEXT}
read -r -a SVM_C_GRID <<< ${SVM_C_GRID_TEXT}
read -r -a SEEDS <<< ${SEEDS_TEXT}

validate_python_runtime() {
  local interpreter=$1
  local label=$2
  local probe_output

  if ! command -v ${interpreter} >/dev/null 2>&1
  then
    echo ERROR_${label}_PYTHON_NOT_EXECUTABLE=${interpreter}
    exit 2
  fi
  if ! probe_output=$("${interpreter}" -c 'import encodings,sys;print(sys.prefix)' 2>&1)
  then
    echo ERROR_${label}_PYTHON_RUNTIME_BROKEN=${interpreter}
    printf '%s\n' "${probe_output}"
    exit 2
  fi
}

validate_python_runtime "${PYTHON}" BENCHMARK
for selected_model in "${MODELS[@]}"
do
  if [[ ${selected_model} = EEGNet ]]
  then
    validate_python_runtime "${EEGNET_PYTHON}" EEGNET
    break
  fi
done

case ${MISSION} in
  dataset_transfer)
    MISSION_MODULE=eeg_benchmark.tasks.cross_dataset
    SUBCOMMAND=dataset-transfer
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${DATA_ROOT}/PreprocessResults_eeg_cross_dataset
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${DATA_ROOT}/Results
    DEFAULT_TASKS_TEXT='detection prediction'
    DEFAULT_DIRECTIONS_TEXT='tusz:chbmit chbmit:tusz siena:chbmit chbmit:siena'
    ;;
  cross_time|chbmit_cross_time)
    MISSION_MODULE=eeg_benchmark.tasks.cross_time
    SUBCOMMAND=
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${CROSS_TIME_PREPROCESS_ROOT}
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${DATA_ROOT}/Results_cross_time
    DEFAULT_TASKS_TEXT='detection prediction'
    DEFAULT_DIRECTIONS_TEXT='chbmit_historical:chbmit_future'
    ;;
  eeg_ieeg_detection)
    MISSION_MODULE=eeg_benchmark.tasks.cross_modal
    SUBCOMMAND=eeg-ieeg-transfer
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${DATA_ROOT}/PreprocessResults_eeg_ieeg_cross_modal
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${CROSS_MODAL_RESULT_ROOT}
    DEFAULT_TASKS_TEXT='detection'
    DEFAULT_DIRECTIONS_TEXT='tusz:epilepsy_ieeg'
    ;;
  eeg_ieeg_prediction)
    MISSION_MODULE=eeg_benchmark.tasks.cross_modal
    SUBCOMMAND=eeg-ieeg-transfer
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${DATA_ROOT}/PreprocessResults_eeg_ieeg_cross_modal
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${CROSS_MODAL_RESULT_ROOT}
    DEFAULT_TASKS_TEXT='prediction'
    DEFAULT_DIRECTIONS_TEXT='tusz:thalamocortical_ieeg siena:thalamocortical_ieeg chbmit:thalamocortical_ieeg'
    ;;
  eeg_ieeg_transfer)
    MISSION_MODULE=eeg_benchmark.tasks.cross_modal
    SUBCOMMAND=eeg-ieeg-transfer
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${DATA_ROOT}/PreprocessResults_eeg_ieeg_cross_modal
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${CROSS_MODAL_RESULT_ROOT}
    DEFAULT_TASKS_TEXT='detection prediction'
    DEFAULT_DIRECTIONS_TEXT='tusz:epilepsy_ieeg tusz:thalamocortical_ieeg siena:epilepsy_ieeg siena:thalamocortical_ieeg chbmit:epilepsy_ieeg chbmit:thalamocortical_ieeg'
    ;;
  eeg_ieeg_localization)
    MISSION_MODULE=eeg_benchmark.tasks.cross_modal
    SUBCOMMAND=eeg-ieeg-localization
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${DATA_ROOT}/PreprocessResults_eeg_ieeg_cross_modal
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${DATA_ROOT}/results/localization
    LOCALIZATION_REFERENCE_RESULT_ROOT=${LOCALIZATION_REFERENCE_RESULT_ROOT:-${CROSS_MODAL_RESULT_ROOT}}
    INTERPRETABILITY=0
    DEFAULT_TASKS_TEXT='localization'
    DEFAULT_DIRECTIONS_TEXT='tusz:epilepsy_ieeg'
    ;;
  ieeg_indomain)
    MISSION_MODULE=eeg_benchmark.tasks.cross_modal
    SUBCOMMAND=ieeg-indomain
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=${DATA_ROOT}/PreprocessResults_eeg_ieeg_cross_modal
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${DATA_ROOT}/Results_ieeg_indomain
    BUDGETS_TEXT='0'
    DEFAULT_TASKS_TEXT='detection prediction'
    DEFAULT_DIRECTIONS_TEXT='epilepsy_ieeg:epilepsy_ieeg thalamocortical_ieeg:thalamocortical_ieeg'
    ;;
  epishift_reliability)
    MISSION_MODULE=eeg_benchmark.analysis.reliability
    SUBCOMMAND=epishift-reliability
    [ ${PREPROCESS_ROOT} = auto ] && PREPROCESS_ROOT=none
    [ ${RESULT_ROOT} = auto ] && RESULT_ROOT=${DATA_ROOT}/results/reliability
    DEFAULT_TASKS_TEXT='detection prediction'
    DEFAULT_DIRECTIONS_TEXT='all'
    ;;
  *)
    echo Unsupported_MISSION=${MISSION}
    exit 2
    ;;
esac

[[ -z ${TASKS_TEXT} ]] && TASKS_TEXT=${DEFAULT_TASKS_TEXT}
[[ -z ${DIRECTIONS_TEXT} ]] && DIRECTIONS_TEXT=${DEFAULT_DIRECTIONS_TEXT}
read -r -a TASKS <<< ${TASKS_TEXT}
read -r -a DIRECTIONS <<< ${DIRECTIONS_TEXT}

validate_mission_cache_pair() {
  case ${MISSION} in
    dataset_transfer|cross_time|chbmit_cross_time)
      if [[ ${PREPROCESS_ROOT} == *PreprocessResults_eeg_ieeg_cross_modal* ]]
      then
        echo ERROR_MISSION_PREPROCESS_CONTRACT_MISMATCH
        echo MISSION=${MISSION}
        echo PREPROCESS_ROOT=${PREPROCESS_ROOT}
        if [[ ${MISSION} = cross_time || ${MISSION} = chbmit_cross_time ]]
        then
          echo EXPECTED_PREPROCESS_ROOT_FOR_CROSS_TIME=${CROSS_TIME_PREPROCESS_ROOT}
        else
          echo EXPECTED_PREPROCESS_ROOT_FOR_DATASET_TRANSFER=${DATA_ROOT}/PreprocessResults_eeg_cross_dataset
        fi
        echo HINT_USE_MISSION=eeg_ieeg_transfer_FOR_CROSS_MODAL_CACHE
        exit 2
      fi
      ;;
    eeg_ieeg_detection|eeg_ieeg_prediction|eeg_ieeg_transfer|eeg_ieeg_localization|ieeg_indomain)
      if [[ ${PREPROCESS_ROOT} == *PreprocessResults_eeg_cross_dataset* ]]
      then
        echo ERROR_MISSION_PREPROCESS_CONTRACT_MISMATCH
        echo MISSION=${MISSION}
        echo PREPROCESS_ROOT=${PREPROCESS_ROOT}
        echo EXPECTED_PREPROCESS_ROOT_FOR_EEG_IEEG=${DATA_ROOT}/PreprocessResults_eeg_ieeg_cross_modal
        echo HINT_USE_MISSION=dataset_transfer_FOR_EEG_EEG_CACHE
        exit 2
      fi
      ;;
  esac
}

validate_mission_directions() {
  local direction
  local source_dataset
  local target_dataset
  case ${MISSION} in
    eeg_ieeg_detection|eeg_ieeg_prediction|eeg_ieeg_transfer|eeg_ieeg_localization)
      for direction in "${DIRECTIONS[@]}"
      do
        source_dataset=${direction%%:*}
        target_dataset=${direction#*:}
        if [[ ${target_dataset} != epilepsy_ieeg && ${target_dataset} != hup_ieeg && ${target_dataset} != thalamocortical_ieeg ]]
        then
          echo ERROR_EEG_IEEG_DIRECTION_TARGET_MUST_BE_IEEG
          echo MISSION=${MISSION}
          echo BAD_DIRECTION=${direction}
          echo VALID_DETECTION_EXAMPLE=tusz:epilepsy_ieeg
          echo VALID_PREDICTION_EXAMPLE=chbmit:thalamocortical_ieeg
          exit 2
        fi
      done
      ;;
    ieeg_indomain)
      for direction in "${DIRECTIONS[@]}"
      do
        source_dataset=${direction%%:*}
        target_dataset=${direction#*:}
        if [[ ${source_dataset} != ${target_dataset} ]]
        then
          echo ERROR_IEEG_INDOMAIN_DIRECTION_MUST_MATCH
          echo MISSION=${MISSION}
          echo BAD_DIRECTION=${direction}
          echo VALID_DETECTION_EXAMPLE=epilepsy_ieeg:epilepsy_ieeg
          echo VALID_PREDICTION_EXAMPLE=thalamocortical_ieeg:thalamocortical_ieeg
          exit 2
        fi
        if [[ ${source_dataset} != epilepsy_ieeg && ${source_dataset} != thalamocortical_ieeg ]]
        then
          echo ERROR_IEEG_INDOMAIN_DATASET_UNSUPPORTED
          echo MISSION=${MISSION}
          echo BAD_DIRECTION=${direction}
          echo VALID_DATASETS=epilepsy_ieeg,thalamocortical_ieeg
          exit 2
        fi
      done
      ;;
    dataset_transfer)
      for direction in "${DIRECTIONS[@]}"
      do
        source_dataset=${direction%%:*}
        target_dataset=${direction#*:}
        if [[ ${source_dataset} == *_ieeg || ${target_dataset} == *_ieeg ]]
        then
          echo ERROR_DATASET_TRANSFER_DIRECTION_MUST_BE_EEG_TO_EEG
          echo MISSION=${MISSION}
          echo BAD_DIRECTION=${direction}
          echo VALID_EXAMPLE=tusz:chbmit
          echo HINT_USE_MISSION=eeg_ieeg_transfer_FOR_IEEG_TARGETS
          exit 2
        fi
      done
      ;;
  esac
}

validate_mission_cache_pair
validate_mission_directions

export PYTHONHASHSEED=${RAND_SEED}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BENCHMARK_NUMERIC_MODE=${NUMERIC_MODE}
export BENCHMARK_FAST_ALGORITHMS=${FAST_ALGORITHMS}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${PHYSICAL_GPU}
export BENCHMARK_PHYSICAL_GPU=${PHYSICAL_GPU}
export EEGPT_FORWARD_CHUNK_SIZE
export EEGPT_ACTIVATION_CHECKPOINTING
export DATA_AUGMENTATION=0
export DROPOUT=0
export PDFORMER_ATTN_DROPOUT=0
export PDFORMER_DROP_PATH=0
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/benchmark_numba_cache
export MPLCONFIGDIR=/tmp/benchmark_mpl_cache
export SAMPLING_AUDIT_MODE=${SAMPLING_AUDIT_MODE}

EXTRA_ARGS=()
[ ${LR_OVERRIDE} != none ] && EXTRA_ARGS+=(--lr ${LR_OVERRIDE})
[ ${TARGET_LR_OVERRIDE} != none ] && EXTRA_ARGS+=(--target-lr ${TARGET_LR_OVERRIDE})
[ ${TARGET_BACKBONE_LR_OVERRIDE} != none ] && EXTRA_ARGS+=(--target-backbone-lr ${TARGET_BACKBONE_LR_OVERRIDE})
[ ${WEIGHT_DECAY_OVERRIDE} != none ] && EXTRA_ARGS+=(--weight-decay ${WEIGHT_DECAY_OVERRIDE})
[ ${MAX_GRAD_NORM_OVERRIDE} != none ] && EXTRA_ARGS+=(--max-grad-norm ${MAX_GRAD_NORM_OVERRIDE})
[ ${MRI_TEMPLATE} != none ] && EXTRA_ARGS+=(--mri-template-path ${MRI_TEMPLATE})
[ ${EEGNET_CUDA_ROOT} != none ] && EXTRA_ARGS+=(--eegnet-cuda-root ${EEGNET_CUDA_ROOT})
if [ ${MISSION} = eeg_ieeg_localization ]
then
  LOCALIZATION_REFERENCE_WINDOW=${LOCALIZATION_REFERENCE_WINDOW:-${WINDOWS_TEXT}}
  if [[ ${LOCALIZATION_REFERENCE_WINDOW} == *' '* ]]
  then
    echo ERROR_LOCALIZATION_REFERENCE_WINDOW_MUST_BE_SINGLE_VALUE
    exit 2
  fi
  EXTRA_ARGS+=(--localization-reference-window-seconds ${LOCALIZATION_REFERENCE_WINDOW})
fi
[ ${DRY_RUN} -eq 1 ] && EXTRA_ARGS+=(--dry-run)
[ ${PREFLIGHT_ONLY} -eq 1 ] && EXTRA_ARGS+=(--preflight-only)
if [ ${INTERPRETABILITY_ONLY} -eq 1 ]
then
  if [ ${#MODELS[@]} -ne 1 ] || [ ${MODELS[0]} != EEGNet ]
  then
    echo ERROR_INTERPRETABILITY_ONLY_REQUIRES_SINGLE_EEGNET_MODEL
    exit 2
  fi
  SKIP_COMPLETED=0
  EXTRA_ARGS+=(--interpretability-only)
fi
EXTRA_ARGS+=(--stop-on-error ${STOP_ON_ERROR})
EXTRA_ARGS+=(--skip-completed ${SKIP_COMPLETED})
EXTRA_ARGS+=(--reuse-incomplete-source-checkpoint ${REUSE_INCOMPLETE_SOURCE_CHECKPOINT})

cd ${BENCHMARK_ROOT}

echo GPU=${GPU}
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo NUMERIC_MODE=${NUMERIC_MODE}
echo FAST_ALGORITHMS=${FAST_ALGORITHMS}
echo DETERMINISTIC=${DETERMINISTIC}
echo MISSION=${MISSION}
echo MODELS=${MODELS_TEXT}
echo TASKS=${TASKS_TEXT}
echo DIRECTIONS=${DIRECTIONS_TEXT}
echo WINDOWS=${WINDOWS_TEXT}
echo BUDGETS=${BUDGETS_TEXT}
echo RESULT_ROOT=${RESULT_ROOT}
echo CROSS_MODAL_RESULT_ROOT=${CROSS_MODAL_RESULT_ROOT}
echo NEW_BASELINE_RESULT_ROOT=${NEW_BASELINE_RESULT_ROOT}
echo LOCALIZATION_REFERENCE_RESULT_ROOT=${LOCALIZATION_REFERENCE_RESULT_ROOT:-none}
echo LOCALIZATION_REFERENCE_WINDOW=${LOCALIZATION_REFERENCE_WINDOW:-${WINDOWS_TEXT}}
echo BUDGET_UNIT=patient
echo BUDGET_FINETUNE_STRATEGY=model_appropriate_budget_adaptation
echo DEEP_BUDGET_OPTIMIZATION=linear_probe_then_full_model_discriminative_learning_rates
echo SVM_BUDGET_OPTIMIZATION=budget_refit_with_source_rehearsal
echo BUDGET_EPOCHS=${BUDGET_EPOCHS}
echo BUDGET_PATIENCE=${BUDGET_PATIENCE}
echo BUDGET_HEAD_ONLY_EPOCHS=${BUDGET_HEAD_ONLY_EPOCHS}
echo PROCESS_NAME=${PROCESS_NAME}
echo EEGNET_PYTHON=${EEGNET_PYTHON}
echo EEGNET_CUDA_ROOT=${EEGNET_CUDA_ROOT}
echo DETECTION_TRAIN_SAMPLING=TUSZ_Siena_CHB-MIT_epoch_dynamic_ictal_1_non_ictal_2
echo DETECTION_TARGET_BUDGET_SAMPLING=selected_patients_all_ictal_dynamic_hard_far_one_to_two
echo PREDICTION_TRAIN_SAMPLING=dataset_aware_epoch_dynamic_preictal_anchored_one_to_one
echo PREDICTION_TARGET_BUDGET_SAMPLING=selected_patients_all_preictal_dynamic_patient_balanced_interictal_one_to_one
echo SOURCE_REHEARSAL_SAMPLING=prediction_one_to_one_detection_one_to_two
echo TARGET_DEV_POLICY=fixed_full_target_dev_excluded_from_budget
echo SOURCE_REHEARSAL_FRACTION=${SOURCE_REHEARSAL_FRACTION}
echo SAMPLING_AUDIT_MODE=${SAMPLING_AUDIT_MODE}
echo EARLY_STOPPING_MIN_DELTA=${MIN_DELTA}
echo STATISTICS_CPU_WORKERS=${STATS_NUM_WORKERS}
echo BOOTSTRAP_RESAMPLES=${BOOTSTRAP_RESAMPLES}
echo PREFLIGHT_ONLY=${PREFLIGHT_ONLY}
echo INTERPRETABILITY_ONLY=${INTERPRETABILITY_ONLY}
echo CBRAMOD_PREFETCH_FACTOR=${CBRAMOD_PREFETCH_FACTOR}
echo BENDR_PREFETCH_FACTOR=${BENDR_PREFETCH_FACTOR}
echo BENDR_PROGRESS_UPDATE_INTERVAL=${BENDR_PROGRESS_UPDATE_INTERVAL}
echo EEGPT_PREFETCH_FACTOR=${EEGPT_PREFETCH_FACTOR}
echo EEGPT_FORWARD_CHUNK_SIZE=${EEGPT_FORWARD_CHUNK_SIZE}
echo EEGPT_ACTIVATION_CHECKPOINTING=${EEGPT_ACTIVATION_CHECKPOINTING}
echo STEEGFORMER_PREFETCH_FACTOR=${STEEGFORMER_PREFETCH_FACTOR}
if [[ ${MISSION} == epishift_reliability ]]
then
  echo EPISHIFT_CROSS_DATASET_RESULT_ROOT=${EPISHIFT_CROSS_DATASET_RESULT_ROOT}
  echo EPISHIFT_CROSS_MODAL_RESULT_ROOT=${EPISHIFT_CROSS_MODAL_RESULT_ROOT}
  echo EPISHIFT_ECE_BINS=${EPISHIFT_ECE_BINS}
  echo EPISHIFT_SAVE_FIGURES=${EPISHIFT_SAVE_FIGURES}
  ${PYTHON} -m ${MISSION_MODULE} \
    --models ${MODELS[@]} \
    --tasks ${TASKS[@]} \
    --directions ${DIRECTIONS[@]} \
    --window-seconds ${WINDOWS[@]} \
    --budget-percent ${BUDGETS[@]} \
    --seeds ${SEEDS[@]} \
    --cross-dataset-result-root ${EPISHIFT_CROSS_DATASET_RESULT_ROOT} \
    --cross-modal-result-root ${EPISHIFT_CROSS_MODAL_RESULT_ROOT} \
    --output-dir ${RESULT_ROOT} \
    --ece-bins ${EPISHIFT_ECE_BINS} \
    --save-figures ${EPISHIFT_SAVE_FIGURES} \
    --stop-on-error ${STOP_ON_ERROR}
  exit $?
fi
echo CST_RESIZE_HEADS=${CST_RESIZE_HEADS}
echo CST_RESIZE_LAYERS=${CST_RESIZE_LAYERS}
echo CST_RESIZE_FEEDFORWARD=${CST_RESIZE_FEEDFORWARD}
echo CST_KD_WEIGHT=${CST_KD_WEIGHT}
echo CST_MCC_WEIGHT=${CST_MCC_WEIGHT}
echo CST_KD_TEMPERATURE=${CST_KD_TEMPERATURE}
echo CST_MCC_TEMPERATURE=${CST_MCC_TEMPERATURE}
echo RIVER_LATENT_ELECTRODES=${RIVER_LATENT_ELECTRODES}
echo RIVER_EMBED_DIM=${RIVER_EMBED_DIM}
echo RIVER_DEPTH=${RIVER_DEPTH}
echo RIVER_DROPOUT=${RIVER_DROPOUT}
echo RIVER_RDROP_ENABLED=${RIVER_RDROP_ENABLED}
echo RIVER_RDROP_WEIGHT=${RIVER_RDROP_WEIGHT}
echo RIVER_EMA_DECAY=${RIVER_EMA_DECAY}
echo RIVER_WARMUP_RATIO=${RIVER_WARMUP_RATIO}
echo RIVER_MIN_LR_RATIO=${RIVER_MIN_LR_RATIO}
echo RIVER_CHANNEL_DROP_RATIO=${RIVER_CHANNEL_DROP_RATIO}
echo RIVER_TEMPORAL_PATCH_MASK_ENABLED=${RIVER_TEMPORAL_PATCH_MASK_ENABLED}
echo RIVER_TEMPORAL_PATCH_MASK_RATIO=${RIVER_TEMPORAL_PATCH_MASK_RATIO}
echo RIVER_TEMPORAL_PATCH_MASK_BLOCKS=${RIVER_TEMPORAL_PATCH_MASK_BLOCKS}
echo RIVER_TEMPORAL_PATCH_MASK_MIN_KEEP_RATIO=${RIVER_TEMPORAL_PATCH_MASK_MIN_KEEP_RATIO}
echo RIVER_PRECISION=${RIVER_PRECISION}
echo RIVER_RESIDUAL_BOTTLENECK_DIM=${RIVER_RESIDUAL_BOTTLENECK_DIM}
echo RIVER_CHANNEL_DROP_ENABLED=${RIVER_CHANNEL_DROP_ENABLED}
echo RIVER_FUSE_AUGMENTATION_VIEWS=${RIVER_FUSE_AUGMENTATION_VIEWS}
echo RIVER_FOURIER_RANK=${RIVER_FOURIER_RANK}
echo RIVER_TRANSPORT_MODE=${RIVER_TRANSPORT_MODE}
echo RIVER_TEMPORAL_MIXER=${RIVER_TEMPORAL_MIXER}
echo RIVER_ALIGNMENT_WEIGHT=${RIVER_ALIGNMENT_WEIGHT}
echo RIVER_MONTAGE_CONSISTENCY_WEIGHT=${RIVER_MONTAGE_CONSISTENCY_WEIGHT}
echo RIVER_MASS_AWARE_TRANSPORT=${RIVER_MASS_AWARE_TRANSPORT}
echo RIVER_SLOT_COMPETITION_STRENGTH=${RIVER_SLOT_COMPETITION_STRENGTH}
echo RIVER_SLOT_COMPETITION_TEMPERATURE=${RIVER_SLOT_COMPETITION_TEMPERATURE}
echo RIVER_FUSE_CONSISTENCY_VIEWS=${RIVER_FUSE_CONSISTENCY_VIEWS}
echo RF_N_ESTIMATORS=${RF_N_ESTIMATORS}
echo RF_MAX_DEPTH=${RF_MAX_DEPTH}
echo RF_MAX_FEATURES=${RF_MAX_FEATURES}
echo RF_MIN_SAMPLES_LEAF=${RF_MIN_SAMPLES_LEAF}
echo CMMN_C=${CMMN_C}
echo CMMN_MAX_ITER=${CMMN_MAX_ITER}

RF_MAX_DEPTH_ARGS=()
if [[ ${RF_MAX_DEPTH} != none ]]
then
  RF_MAX_DEPTH_ARGS=(--rf-max-depth ${RF_MAX_DEPTH})
fi

${PYTHON} -m ${MISSION_MODULE} ${SUBCOMMAND} \
  --models ${MODELS[@]} \
  --tasks ${TASKS[@]} \
  --directions ${DIRECTIONS[@]} \
  --window-seconds ${WINDOWS[@]} \
  --budget-percent ${BUDGETS[@]} \
  --budget-seed ${BUDGET_SEED} \
  --undersample-seed ${UNDERSAMPLE_SEED} \
  --source-rehearsal-fraction ${SOURCE_REHEARSAL_FRACTION} \
  --use-pretrained ${USE_PRETRAINED} \
  --epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --budget-epochs ${BUDGET_EPOCHS} \
  --budget-patience ${BUDGET_PATIENCE} \
  --budget-head-only-epochs ${BUDGET_HEAD_ONLY_EPOCHS} \
  --min-delta ${MIN_DELTA} \
  --seed ${RAND_SEED} \
  --batch-size ${BATCH_SIZE} \
  --num-workers ${NUM_WORKERS} \
  --stats-num-workers ${STATS_NUM_WORKERS} \
  --bootstrap-resamples ${BOOTSTRAP_RESAMPLES} \
  --gpu ${RUNTIME_GPU} \
  --physical-gpu ${PHYSICAL_GPU} \
  --deterministic ${DETERMINISTIC} \
  --generate-interpretability ${INTERPRETABILITY} \
  --interpretability-max-clips ${INTERPRETABILITY_MAX_CLIPS} \
  --biot-token-size ${BIOT_TOKEN_SIZE} \
  --biot-hop-length ${BIOT_HOP_LENGTH} \
  --cbramod-optimizer ${CBRAMOD_OPTIMIZER} \
  --cbramod-clip-value ${CBRAMOD_CLIP_VALUE} \
  --cbramod-multi-lr ${CBRAMOD_MULTI_LR} \
  --cbramod-prefetch-factor ${CBRAMOD_PREFETCH_FACTOR} \
  --bendr-prefetch-factor ${BENDR_PREFETCH_FACTOR} \
  --bendr-progress-update-interval ${BENDR_PROGRESS_UPDATE_INTERVAL} \
  --eegpt-prefetch-factor ${EEGPT_PREFETCH_FACTOR} \
  --steegformer-prefetch-factor ${STEEGFORMER_PREFETCH_FACTOR} \
  --cst-resize-heads ${CST_RESIZE_HEADS} \
  --cst-resize-layers ${CST_RESIZE_LAYERS} \
  --cst-resize-feedforward ${CST_RESIZE_FEEDFORWARD} \
  --cst-kd-weight ${CST_KD_WEIGHT} \
  --cst-mcc-weight ${CST_MCC_WEIGHT} \
  --cst-kd-temperature ${CST_KD_TEMPERATURE} \
  --cst-mcc-temperature ${CST_MCC_TEMPERATURE} \
  --evobrain-max-grad-norm ${EVOBRAIN_MAX_GRAD_NORM} \
  --evobrain-rnn-units ${EVOBRAIN_RNN_UNITS} \
  --evobrain-agg ${EVOBRAIN_AGG} \
  --evobrain-top-k ${EVOBRAIN_TOP_K} \
  --river-descriptor-hidden-dim ${RIVER_DESCRIPTOR_HIDDEN_DIM} \
  --river-descriptor-window-seconds ${RIVER_DESCRIPTOR_WINDOW_SECONDS} \
  --river-descriptor-slow-seconds ${RIVER_DESCRIPTOR_SLOW_SECONDS} \
  --river-descriptor-channel-chunk-size ${RIVER_DESCRIPTOR_CHANNEL_CHUNK_SIZE} \
  --river-latent-electrodes ${RIVER_LATENT_ELECTRODES} \
  --river-embed-dim ${RIVER_EMBED_DIM} \
  --river-depth ${RIVER_DEPTH} \
  --river-heads ${RIVER_HEADS} \
  --river-ff-dim ${RIVER_FF_DIM} \
  --river-fourier-modes ${RIVER_FOURIER_MODES} \
  --river-fourier-rank ${RIVER_FOURIER_RANK} \
  --river-local-kernel-size ${RIVER_LOCAL_KERNEL_SIZE} \
  --river-slot-temperature ${RIVER_SLOT_TEMPERATURE} \
  --river-evidence-smoothing-kernel ${RIVER_EVIDENCE_SMOOTHING_KERNEL} \
  --river-sinkhorn-iterations ${RIVER_SINKHORN_ITERATIONS} \
  --river-transport-epsilon ${RIVER_TRANSPORT_EPSILON} \
  --river-electrode-mass-relaxation ${RIVER_ELECTRODE_MASS_RELAXATION} \
  --river-electrode-mass-temperature ${RIVER_ELECTRODE_MASS_TEMPERATURE} \
  --river-pooling-temperature ${RIVER_POOLING_TEMPERATURE} \
  --river-transport-mode ${RIVER_TRANSPORT_MODE} \
  --river-temporal-mixer ${RIVER_TEMPORAL_MIXER} \
  --river-state-encoder-mode ${RIVER_STATE_ENCODER_MODE} \
  --river-diversity-weight ${RIVER_DIVERSITY_WEIGHT} \
  --river-alignment-weight ${RIVER_ALIGNMENT_WEIGHT} \
  --river-montage-consistency-weight ${RIVER_MONTAGE_CONSISTENCY_WEIGHT} \
  --river-channel-drop-ratio ${RIVER_CHANNEL_DROP_RATIO} \
  --river-channel-drop-start-ratio ${RIVER_CHANNEL_DROP_START_RATIO} \
  --river-temporal-patch-mask-enabled ${RIVER_TEMPORAL_PATCH_MASK_ENABLED} \
  --river-temporal-patch-mask-ratio ${RIVER_TEMPORAL_PATCH_MASK_RATIO} \
  --river-temporal-patch-mask-blocks ${RIVER_TEMPORAL_PATCH_MASK_BLOCKS} \
  --river-temporal-patch-mask-min-keep-ratio ${RIVER_TEMPORAL_PATCH_MASK_MIN_KEEP_RATIO} \
  --river-channel-drop-enabled ${RIVER_CHANNEL_DROP_ENABLED} \
  --river-fuse-augmentation-views ${RIVER_FUSE_AUGMENTATION_VIEWS} \
  --river-prototype-momentum ${RIVER_PROTOTYPE_MOMENTUM} \
  --river-mass-aware-transport ${RIVER_SELECTED_MASS_AWARE_TRANSPORT} \
  --river-residual-bottleneck-dim ${RIVER_RESIDUAL_BOTTLENECK_DIM} \
  --river-slot-competition-strength ${RIVER_SLOT_COMPETITION_STRENGTH} \
  --river-slot-competition-temperature ${RIVER_SLOT_COMPETITION_TEMPERATURE} \
  --river-dropout ${RIVER_DROPOUT} \
  --river-rdrop-enabled ${RIVER_RDROP_ENABLED} \
  --river-rdrop-weight ${RIVER_RDROP_WEIGHT} \
  --river-ema-decay ${RIVER_EMA_DECAY} \
  --river-precision ${RIVER_PRECISION} \
  --river-warmup-ratio ${RIVER_WARMUP_RATIO} \
  --river-min-lr-ratio ${RIVER_MIN_LR_RATIO} \
  --svm-c-grid ${SVM_C_GRID[@]} \
  --svm-max-iter ${SVM_MAX_ITER} \
  --rf-n-estimators ${RF_N_ESTIMATORS} \
  ${RF_MAX_DEPTH_ARGS[@]-} \
  --rf-max-features ${RF_MAX_FEATURES} \
  --rf-min-samples-leaf ${RF_MIN_SAMPLES_LEAF} \
  --cmmn-c ${CMMN_C} \
  --cmmn-max-iter ${CMMN_MAX_ITER} \
  --benchmark-root ${BENCHMARK_ROOT} \
  --preprocess-root ${PREPROCESS_ROOT} \
  --result-root ${RESULT_ROOT} \
  --new-baseline-result-root ${NEW_BASELINE_RESULT_ROOT} \
  --cross-modal-result-root ${LOCALIZATION_REFERENCE_RESULT_ROOT:-${CROSS_MODAL_RESULT_ROOT}} \
  --process-name ${PROCESS_NAME} \
  --eegnet-python ${EEGNET_PYTHON} \
  ${EXTRA_ARGS[@]-}
