from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Audio

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

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, default=None)
    parser.add_argument('--syn', type=str, default=None)
    args = parser.parse_args()
    
    print('loading model')



    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to("cuda")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print('loading data')
    print('train_domains:', args.domains)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large", language="Hindi", task='transcribe')
    dataset = DatasetDict()

    dataset["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="train+validation", use_auth_token=False)
    dataset["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=False)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    run_name = f'whisper_common_hindi_large'

    # run_name += "_small"
    print(dataset['train'][0])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(prepare_dataset, num_proc=1, fn_kwargs={"processor": processor})
    print(dataset['train'][0].keys())
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=f"./outputs/{run_name}",  # change to a repo name of your choice
    group_by_length=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # increase by 2x for every 2x decrease in batch size
    learning_rate=4.375e-6,
    run_name=run_name,
    warmup_steps=500,
    max_steps=3000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    trainer.save_model(training_args.output_dir)