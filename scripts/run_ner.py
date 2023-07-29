#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import Sequence, ClassLabel, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from transformers_crf.trainer import CRFTrainer, CRFTrainingArguments
from transformers_crf.bert_crf import AutoModelForEmbedderCRFTokenClassification
from transformers_crf.util import get_offsets_from_offsets_mapping
from transformers_crf.data_collator import DataCollatorForTokenClassificationWithCRF

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.31.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    use_crf: bool = field(
        default=False,
        metadata={"help": "Will enable to use CRF layer."},
    )
    constrain_crf: bool = field(
        default=True,
        metadata={"help": "If constrain crf outputs"},
    )
    dropout_task: Optional[float] = field(
        default=None,
        metadata={"help": "Task layer dropout for Roberta and Deberta models"}
    )
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "early_stopping_patience"}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "early_stopping_threshold"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    convert_to_iobes: bool = field(
        default=False,
        metadata={"help": "Convert a IOB2 input to IOBES."},
    )
    break_docs_to_max_length: bool = field(
        default=False,
        metadata={"help": "Whether to chunck docs into sentences with the max seq length of tokenizer."},
    )
    break_docs_not_split_entities: bool = field(
        default=True,
        metadata={"help": "Whether to mantain entity integrity when breaking documents."},
    )
    break_docs_fix_splitted_entites: bool = field(
        default=True,
        metadata={"help": "Whether to fix IOB2/IOBES tags of splitted entities."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CRFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_ner", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

    if data_args.text_column_name is not None:
        text_column_name = data_args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if data_args.label_column_name is not None:
        label_column_name = data_args.label_column_name
    elif f"{data_args.task_name}_tags" in column_names:
        label_column_name = f"{data_args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    end_last_tag = 'E'
    single_unique_tag = 'S'
    is_iobes = False

    if data_args.convert_to_iobes:
        is_iobes = True
        label_list = features[label_column_name].feature.names
        def convert_to_iobes(examples, end_last_tag=end_last_tag, single_unique_tag=single_unique_tag):
            examples['new_tags'] = [[label_list[label] for label in label_seq] for label_seq in examples[label_column_name]]
            
            for i in range(len(examples['new_tags'])):
                for j in range(len(examples['new_tags'][i])):
                    if examples['new_tags'][i][j] == 'O':
                        continue
                    elif examples['new_tags'][i][j].startswith('B-'):
                        if j == len(examples['new_tags'][i])-1:
                            examples['new_tags'][i][j] = f'{single_unique_tag}-'+examples['new_tags'][i][j][2:]
                        elif examples['new_tags'][i][j+1].startswith('I-'):
                            continue
                        else:
                            examples['new_tags'][i][j] = f'{single_unique_tag}-'+examples['new_tags'][i][j][2:]
                    elif examples['new_tags'][i][j].startswith('I-'):
                        if j == len(examples['new_tags'][i])-1:
                            examples['new_tags'][i][j] = f'{end_last_tag}-'+examples['new_tags'][i][j][2:]
                        elif examples['new_tags'][i][j+1].startswith('I-'):
                            continue
                        else:
                            examples['new_tags'][i][j] = f'{end_last_tag}-'+examples['new_tags'][i][j][2:]
            return examples
        
        iobes_classes = []
        for label in features[label_column_name].feature.names:
            if label == 'O':
                iobes_classes.append(label)
            else:
                label = label[2:]
                if 'B-'+label not in iobes_classes:
                    iobes_classes.append('B-'+label)
                    iobes_classes.append('I-'+label)
                    iobes_classes.append(f'{end_last_tag}-'+label)
                    iobes_classes.append(f'{single_unique_tag}-'+label)
        features[label_column_name] = Sequence(ClassLabel(names=iobes_classes))

        for split in raw_datasets:
            split_dataset = raw_datasets[split].map(convert_to_iobes, batched=True, remove_columns=[label_column_name])
            raw_datasets[split] = split_dataset.rename_column('new_tags', label_column_name)
            raw_datasets[split] = raw_datasets[split].cast(features)

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    # If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
    # Otherwise, we have to get the list of labels manually.
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}

    num_labels = len(label_list)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    config.dropout_task = model_args.dropout_task
    
    if 'deberta' in model_args.model_name_or_path:
        config.cls_dropout = model_args.dropout_task
    else:
        # bert, roberta
        config.classifier_dropout = model_args.dropout_task

    tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if config.model_type in {"bloom", "gpt2", "roberta", "deberta-v2"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Model has labels -> use them.
    if config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(config.label2id.keys()) == sorted(label_list):
            # Reorganize `label_list` to match the ordering of the model;
            if labels_are_int:
                label_to_id = {i: int(config.label2id[l]) for i, l in enumerate(label_list)}
                label_list = [config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for i, l in enumerate(label_list)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(config.label2id.keys())}, dataset labels:"
                f" {sorted(label_list)}.\nIgnoring the model labels as a result.",
            )

    # Set the correspondences label/ID inside the model config
    config.label2id = {l: i for i, l in enumerate(label_list)}
    config.id2label = dict(enumerate(label_list))

    if model_args.use_crf: 
        config.use_crf = model_args.use_crf
        config.constrain_crf = model_args.constrain_crf
        model = AutoModelForEmbedderCRFTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models at"
            " https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet"
            " this requirement"
        )

    print('model.config', model.config)

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if data_args.label_all_tokens:
                        if is_iobes:
                            raise Exception('label_all_tokens not implemented for IOBES')
                        label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize all texts and align the labels with them for embedder.
    def tokenize_and_align_labels_embedder(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            max_length=data_args.max_seq_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )

        offsets = [get_offsets_from_offsets_mapping(offset_mapping) for offset_mapping in tokenized_inputs['offset_mapping']]
        mask = [[True]*(len(offset)) for offset in offsets]

        del tokenized_inputs['offset_mapping']
        tokenized_inputs['offsets'] = offsets
        tokenized_inputs['mask'] = mask

        labels = [ [label_to_id[label] for label in labels_sentence] for labels_sentence in examples[label_column_name]]
        #Trucante labels
        labels = [ label[:len(offset)-2] for label, offset in zip(labels, offsets)]
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    dataset_preprocess_function = tokenize_and_align_labels
    if model_args.use_crf:
        dataset_preprocess_function = tokenize_and_align_labels_embedder

    def add_id_to_data(examples):
        examples['id'] = list(range(len(examples['tokens'])))
        return examples
    
    def next_label_continues(label, next_label):
        if label.startswith('B-') and (next_label.startswith('I-') or next_label.startswith('E-') or next_label.startswith('L-')):
            return True
        if label.startswith('I-') and (next_label.startswith('I-') or next_label.startswith('E-') or next_label.startswith('L-')):
            return True
        return False

    for split in raw_datasets:
        print("Cleaning dataset with", split, len(raw_datasets[split]))
        raw_datasets[split] = raw_datasets[split].filter(lambda example: len(example[text_column_name]) > 0)
        print("Cleaning dataset result:", split, len(raw_datasets[split]))

        if 'id' not in raw_datasets[split].column_names:
            raw_datasets[split] = raw_datasets[split].map(add_id_to_data, batched=True)

        def break_docs_to_max_length(examples, end_last_tag=end_last_tag, single_unique_tag=single_unique_tag):
            not_split_entites = data_args.break_docs_not_split_entities
            fix_splitted_entites = data_args.break_docs_fix_splitted_entites
            max_length = data_args.max_seq_length if data_args.max_seq_length is not None else tokenizer.model_max_length

            new_docs = {
                'id': [],
                'orig_ids': [],
                text_column_name: [],
                label_column_name: [],
                'breaked': []
            }
            
            tokens = examples[text_column_name]
            #labels = 
            temp_label2id = {l: i for i, l in enumerate(label_list)}
            labels = [[label_list[label] for label in label_seq] for label_seq in examples[label_column_name]]	

            tokenized_inputs = tokenizer(
                tokens,
                padding=False,
                truncation=False,
                is_split_into_words=True,
                return_offsets_mapping=True,
                add_special_tokens = False
            )

            for i, orig_tokens, labels, word_ids in zip(examples['id'], tokens, labels, [tokenized_inputs.word_ids(i) for i in range(len(tokens))]):
                id_count = 0
                if len(word_ids) > max_length:
                    token_id = 0
                    while token_id < len(word_ids)-1:

                        next_word_ids = word_ids[token_id:]
                        sentence_word_ids = next_word_ids[:max_length]

                        last_word_id = sentence_word_ids[-1]
                        last_index_of_last_word_id = len(next_word_ids) - 1 - next_word_ids[::-1].index(last_word_id)
                        if last_index_of_last_word_id > len(sentence_word_ids)-1:
                            first_index_of_last_word_id = sentence_word_ids.index(last_word_id)
                            if first_index_of_last_word_id > 0:
                                sentence_word_ids = sentence_word_ids[:first_index_of_last_word_id]
                            else:
                                sentence_word_ids = next_word_ids[:last_index_of_last_word_id+1]

                        if not_split_entites:
                            sentence_labels = labels[min(sentence_word_ids):max(sentence_word_ids)+1]
                            last_sentence_label = labels[max(sentence_word_ids)]
                            if len(labels) > max(sentence_word_ids)+1 and next_label_continues(last_sentence_label, labels[max(sentence_word_ids)+1]):
                                last_sentence_label_b = 'B-' + last_sentence_label[2:]
                                begging_of_entity_index = len(sentence_labels) - 1 - sentence_labels[::-1].index(last_sentence_label_b)
                                max_word_id = begging_of_entity_index + min(sentence_word_ids)
                                if max_word_id > min(sentence_word_ids):
                                    sentence_word_ids = next_word_ids[:next_word_ids.index(max_word_id)]

                        if fix_splitted_entites:
                            last_sentence_label = labels[max(sentence_word_ids)]
                            if last_sentence_label.startswith('I-') and is_iobes:
                                labels[max(sentence_word_ids)] = f'{end_last_tag}-' + last_sentence_label[2:]
                            elif last_sentence_label.startswith('B-') and is_iobes:
                                labels[max(sentence_word_ids)] = f'{single_unique_tag}-' + last_sentence_label[2:]

                            if len(labels) > max(sentence_word_ids)+1:
                                next_label = labels[max(sentence_word_ids)+1]
                                if next_label.startswith('I-'):
                                    labels[max(sentence_word_ids)+1] = 'B-' + next_label[2:]
                                elif next_label.startswith('L-'): #BIOLU
                                    labels[max(sentence_word_ids)+1] = 'U-' + next_label[2:]
                                elif next_label.startswith('E-'): #IOBES
                                    labels[max(sentence_word_ids)+1] = 'S-' + next_label[2:]
                                
                        token_id += len(sentence_word_ids)

                        new_docs['id'].append(f"{i}_{id_count}")
                        new_docs['orig_ids'].append(i)
                        new_docs['tokens'].append(orig_tokens[min(sentence_word_ids):max(sentence_word_ids)+1])
                        new_docs['ner_tags'].append(labels[min(sentence_word_ids):max(sentence_word_ids)+1])
                        new_docs['breaked'].append(True)
                        id_count += 1
                else:
                    new_docs['id'].append(f"{i}_{id_count}")
                    new_docs['orig_ids'].append(i)
                    new_docs['tokens'].append(orig_tokens)
                    new_docs['ner_tags'].append(labels)
                    new_docs['breaked'].append(False)
                    id_count += 1
            
            new_docs[label_column_name] = [[temp_label2id[label] for label in label_seq] for label_seq in new_docs[label_column_name]]
            return new_docs
        
        if data_args.break_docs_to_max_length:
            raw_datasets[split] = raw_datasets[split].map(
                break_docs_to_max_length,
                batched=True,
                desc="Breaking documents",
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[c for c in raw_datasets[split].column_names if c not in (text_column_name, label_column_name, 'id')]
            )


    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                dataset_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            
        logger.info("** Sample Train dataset **")
        logger.info(str(train_dataset[0]))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                dataset_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                dataset_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        logger.info("** Sample Predict dataset **")
        logger.info(str(predict_dataset[0]))

    # Data collator
    if model_args.use_crf:
        data_collator = DataCollatorForTokenClassificationWithCRF(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
        
    callbacks = []
    if model_args.early_stopping_patience is not None:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience = model_args.early_stopping_patience,
            early_stopping_threshold = model_args.early_stopping_threshold
        )
        callbacks.append(early_stopping_callback)

    # Metrics
    metric = evaluate.load("seqeval")
    import numpy as np
    def get_preds_labels(predictions, labels):
        # if CRF reorder labels and preds with the output original order
        if model_args.use_crf:
            preds = predictions[-1]
            #t_labels = labels
            
            original_order = predictions[-2]
            infered_batch_size = max(original_order)+1
            
            new_preds = []
            new_labels = []
            for i in range(0, len(labels), infered_batch_size):
                original_order_batch = original_order[i:i+infered_batch_size]
                reorder_index_batch = np.argsort(original_order_batch)
                
                new_preds.extend(preds[i:i+infered_batch_size][reorder_index_batch])
                new_labels.extend(labels[i:i+infered_batch_size][reorder_index_batch])
                
            return new_preds, new_labels
        else:
            return np.argmax(predictions, axis=2), labels
            
    def convert_label_to_iob2(label):
        if label[0] == 'L' or label[0] == 'E':
            label = 'I' + label[1:]
        if label[0] == 'U' or label[0] == 'S':
            label = 'B' + label[1:]
        return label

    def compute_metrics(p):
        predictions, labels = p
        predictions, labels = get_preds_labels(predictions, labels)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        scheme = 'IOB2'
        if is_iobes:
            scheme = 'IOBES' if end_last_tag == 'E' else 'BILOU'

        true_predictions_iob2 = [[convert_label_to_iob2(label) for label in label_seq] for label_seq in true_predictions]
        true_labels_iob2 = [[convert_label_to_iob2(label) for label in label_seq] for label_seq in true_labels]

        #results = metric.compute(predictions=true_predictions, references=true_labels)
        #results_strict = metric.compute(predictions=true_predictions, references=true_labels, scheme=scheme, mode='strict')
        
        results= metric.compute(predictions=true_predictions_iob2, references=true_labels_iob2, scheme='IOB2', mode=None)
        #results_iob2_strict = metric.compute(predictions=true_predictions_iob2, references=true_labels_iob2, scheme='IOB2', mode='strict')
        
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for rtype, res in [('default', results_iob2)]:
                for key, value in res.items():
                    if isinstance(value, dict):
                        for n, v in value.items():
                            res_name = f"{rtype}_{key}_{n}" if rtype != 'default' else f"{key}_{n}"
                            final_results[res_name] = v
                    else:
                        res_name = f"{rtype}_{key}" if rtype != 'default' else key
                        final_results[res_name] = value
                return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"], 
            }
        
    #test: TrainingArguments = training_args
    #test.label_names = ['predicts']
    #training_args.label_names = ['predicts']
    # Initialize our Trainer
    trainer = CRFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    import wandb
    wandb.config.update(model_args, allow_val_change=True)
    wandb.config.update(data_args, allow_val_change=True)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        output, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions, labels = get_preds_labels(output, labels)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_predictions_iob2 = [[convert_label_to_iob2(label) for label in label_seq] for label_seq in true_predictions]
        true_labels_iob2 = [[convert_label_to_iob2(label) for label in label_seq] for label_seq in true_labels]
        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        delimiter = " "
        
        if trainer.is_world_process_zero():
            output_predictions_file = os.path.join(training_args.output_dir, "predictions.txt")
            with open(output_predictions_file, "w", encoding='utf8') as writer:
                for data, true_seq, preds_seq in zip(predict_dataset, true_labels_iob2, true_predictions_iob2):
                    for token, label, pred in zip(data[text_column_name], true_seq, preds_seq):
                        writer.write(token + delimiter + label + delimiter + pred + "\n")
                    if 'breaked' in data and data['breaked']:
                        continue
                    writer.write("\n")
            if data_args.convert_to_iobes:
                output_predictions_file = os.path.join(training_args.output_dir, "predictions_iobes.txt")
                with open(output_predictions_file, "w", encoding='utf8') as writer:
                    for data, true_seq, preds_seq in zip(predict_dataset, true_labels, true_predictions):
                        for token, label, pred in zip(data[text_column_name], true_seq, preds_seq):
                            writer.write(token + delimiter + label + delimiter + pred + "\n")
                        if 'breaked' in data and data['breaked']:
                            continue
                        writer.write("\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "token-classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
