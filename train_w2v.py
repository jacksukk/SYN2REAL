from datasets import load_dataset, concatenate_datasets
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC, EarlyStoppingCallback, Wav2Vec2FeatureExtractor
import torch
from evaluate import load
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer, HfArgumentParser
from datasets import Audio
from transformers import Wav2Vec2CTCTokenizer
import random
metric = evaluate.load("wer", experiment_id=str(random.random()))
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).upper()
    return batch
# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task='transcribe')
SLURP_DOMAIN = {
'cooking',
'audio',
'transport',
'news',
'music',
'lists',
'weather',
'calendar',
'qa',
'general',
'datetime',
'recommendation',
'play',
'iot',
'social',
'takeaway',
'email',
'alarm',
}
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch, processor):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    pred_str = [s.lower() for s in pred_str]
    label_str = [s.lower() for s in label_str]
    print('pred: ', pred_str)
    print('label: ', label_str)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, default=None)
    parser.add_argument('--syn', type=str, default=None)
    parser.add_argument('--mix', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="facebook/wav2vec2-conformer-rope-large-960h-ft")
    parser.add_argument('--configs', type=str, default="configs/w2v.yaml")
    args = parser.parse_args()
    
    print('loading model')
    patience = 20
    steps = 4000
    args.domains = args.domains.split(';')
    args.domains = [d.strip() for d in args.domains if d.strip() != '']
    # tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    # feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    # processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    model = Wav2Vec2ConformerForCTC.from_pretrained(
        "facebook/wav2vec2-conformer-rope-large-960h-ft",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        attention_dropout=0.1,
        hidden_dropout=0.1,
        mask_time_prob=0.05,
        ignore_mismatched_sizes=True,
    )
    model.freeze_feature_encoder()
    # model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_path, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id,).to("cuda")

    print('loading data')
    print('train_domains:', args.domains)
    

    data_files = {"train":"data/slurp/hg_face_data/data/train/*", "devel":"data/slurp/hg_face_data/data/devel/*"}
    # dataset = load_dataset("marcel-gohsen/slurp", use_auth_token=False, cache_dir="/tmp")
    dataset = load_dataset("audiofolder", data_files=data_files, cache_dir="/tmp")
    dataset = dataset.remove_columns(['split'])
    run_name = f'w2v_slurp'
    training_args = HfArgumentParser(TrainingArguments).parse_yaml_file(args.configs)[0]
    if args.domains:
        
        if len(args.domains) == 1:
            dataset = dataset.filter(lambda example: example["scenario"] in args.domains, load_from_cache_file=False)
            run_name = run_name + "_" + args.domains[0]
        else:
            only = list(SLURP_DOMAIN.difference(set(args.domains)))[0]
            domain_dict = {d:1 for d in args.domains}
            dataset['train'] = dataset['train'].filter(lambda example: example["scenario"] in domain_dict, load_from_cache_file=False)
            dataset['devel'] = dataset['devel'].filter(lambda example: example["scenario"] in [only], load_from_cache_file=False)
            run_name = run_name + "_" + only + '_anti'
    
    if args.syn == "True":
        print("use synthetic data!!")
        synthetic_files = {d:f"data/synthetic/{d}/*" for d in args.domains}
        syn_dataset = load_dataset("audiofolder", data_files=synthetic_files, cache_dir="/tmp")
        syn_dataset = concatenate_datasets([syn_dataset[d] for d in args.domains])
        syn_dataset = syn_dataset.cast_column("audio", Audio(sampling_rate=16000))
        if args.mix != "True":
            dataset['train'] = syn_dataset
            run_name += "_synthetic"
        elif args.mix == "True":
            print("use mixed data!!")
            dataset['train'] = dataset['train'].cast_column("audio", Audio(sampling_rate=16000))
            dataset['train'] = concatenate_datasets([dataset['train'], syn_dataset])
            run_name += "_mixed"
            patience = 100
            training_args.max_steps = 70000
    # run_name += "_subset"

    if "small" in args.model_path:
        run_name += "_small"
    elif "medium" in args.model_path:
        run_name += "_medium"
    elif "large" in args.model_path:
        run_name += "_large"
    
    if "outputs/" in args.model_path:
        run_name = args.model_path.split('/')[-1] + "_continue"
        training_args.learning_rate = training_args.learning_rate * 0.1
    print(dataset['train'][0])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(remove_special_characters)
    dataset = dataset.map(prepare_dataset, num_proc=16, load_from_cache_file=False, fn_kwargs={"processor": processor})
    print(dataset['train'][0].keys())
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    training_args.output_dir=f"./outputs/{run_name}"
    # training_args.learning_rate = learning_rate
    training_args.run_name = run_name
    # training_args = Seq2SeqTrainingArguments(
    # do_train=True,
    # overwrite_output_dir=True,
    # do_eval=True,
    # output_dir=f"./outputs/{run_name}",  # change to a repo name of your choice
    # per_device_train_batch_size=16,
    # gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    # learning_rate=learning_rate,
    # run_name=run_name,
    # warmup_steps=500,
    # max_steps=steps,
    # gradient_checkpointing=True,
    # fp16=True,
    # evaluation_strategy="steps",
    # per_device_eval_batch_size=8,
    # predict_with_generate=True,
    # generation_max_length=225,
    # save_steps=50,
    # eval_steps=50,
    # logging_steps=25,
    # report_to=["wandb"],
    # load_best_model_at_end=True,
    # metric_for_best_model="wer",
    # greater_is_better=False,
    # push_to_hub=False,
    # save_total_limit=2,
    # )

    trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["devel"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    trainer.save_model(training_args.output_dir)