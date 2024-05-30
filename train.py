from datasets import load_dataset, concatenate_datasets
from transformers import WhisperForConditionalGeneration, WhisperProcessor, EarlyStoppingCallback
import torch
from evaluate import load
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from datasets import Audio
import random
metric = evaluate.load("wer")

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
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
def prepare_dataset(batch, processor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    batch["text"] = processor.tokenizer._normalize(batch["text"])
    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# def prepare_dataset(batch, processor):
#     audio = batch["audio"]
#     batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
#     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
#     print(processor.tokenizer(batch["text"]).input_ids)
#     print(processor.tokenizer.decode(batch["labels"], skip_special_tokens=False))
#     return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [processor.tokenizer._normalize(s) for s in pred_str]
    label_str = [processor.tokenizer._normalize(s) for s in label_str]
    print('pred: ', pred_str)
    print('label: ', label_str)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, default=None)
    parser.add_argument('--syn', type=str, default=None)
    parser.add_argument('--mix', type=str, default=None)
    parser.add_argument('--model_path', type=str, default="openai/whisper-small")
    parser.add_argument('--configs', type=str, default="configs/whisper-small.yaml")
    parser.add_argument('--numbers', type=int, default=17)
    args = parser.parse_args()
    
    print('loading model')
    patience = 20
    steps = 4000
    args.domains = args.domains.split(';')
    args.domains = [d.strip() for d in args.domains if d.strip() != '']

    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, device_map="auto", cache_dir='/work/b04203058/huggingface_hub')
    # model.config.dropout = 0.1
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print('loading data')
    print('train_domains:', args.domains)
    processor = WhisperProcessor.from_pretrained(args.model_path, task='transcribe', cache_dir='/work/b04203058/huggingface_hub')

    data_files = {"train":"data/slurp/hg_face_data/data/train/*", "devel":"data/slurp/hg_face_data/data/devel/*"}
    # dataset = load_dataset("marcel-gohsen/slurp", use_auth_token=False, cache_dir="/work/b04203058/huggingface_hub")
    dataset = load_dataset("audiofolder", data_files=data_files, cache_dir="/tmp/")
    dataset = dataset.remove_columns(['split'])
    run_name = f'whisper_slurp' + '_dpwd_'
    training_args = HfArgumentParser(Seq2SeqTrainingArguments).parse_yaml_file(args.configs)[0]
    if args.domains:
        
        if len(args.domains) == 1:
            dataset = dataset.filter(lambda example: example["scenario"] in args.domains, load_from_cache_file=False, num_proc=16)
            run_name = run_name + "_" + args.domains[0]
        else:

            only = list(SLURP_DOMAIN.difference(set(args.domains)))[0]
            selected = random.sample(args.domains, args.numbers)
            domain_dict = {d:1 for d in selected}

            dataset['train'] = dataset['train'].filter(lambda example: example["scenario"] in domain_dict, load_from_cache_file=False, num_proc=16)
            dataset['devel'] = dataset['devel'].filter(lambda example: example["scenario"] in [only], load_from_cache_file=False, num_proc=16)
            run_name = run_name + "_" + only + '_anti'+str(args.numbers)
    
    if args.syn == "True":
        print("use synthetic data!!")
        synthetic_files = {d:f"data/synthetic/{d}/*" for d in selected}
        syn_dataset = load_dataset("audiofolder", data_files=synthetic_files, cache_dir="/work/b04203058/huggingface_hub")
        syn_dataset = concatenate_datasets([syn_dataset[d] for d in selected])
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
        # run_name += "_speech_t5"
            
    # run_name += "_subset"
    if "small" in args.model_path:
        run_name += "_small"
    elif "medium" in args.model_path:
        run_name += "_medium"
    elif "large" in args.model_path:
        run_name += "_large"
    elif "tiny" in args.model_path:
        run_name += "_tiny"
    elif "base" in args.model_path:
        run_name += "_base"
    
    if "outputs/" in args.model_path:
        run_name = args.model_path.split('/')[-1] + "_continue"
        training_args.learning_rate = training_args.learning_rate * 0.1
    print(dataset['train'][0])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(prepare_dataset, num_proc=16, load_from_cache_file=False, fn_kwargs={"processor": processor})
    print(dataset['train'][0].keys())
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
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

    trainer = Seq2SeqTrainer(
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