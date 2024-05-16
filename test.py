from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from argparse import ArgumentParser

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['text'])

    with torch.no_grad():
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    # print(predicted_ids)
    transcription = processor.decode(predicted_ids, skip_special_tokens=False)
    # print(transcription)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    # print(batch["prediction"])
    return batch
# for p1, p2 in zip(model1.paremeters(), model2.parameters()):
#     p2.data -= p1.data
def merge(model_syn_anti, model_anti, model_target_syn, args):
    
    for p1, p2 in zip(model_syn_anti.parameters(), model_anti.parameters()):
        p2.data -= 1 * p1.data
    
    for p1, p2 in zip(model_anti.parameters(), model_target_syn.parameters()):
        p2.data += args.weight *  p1.data

    return model_target_syn
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_syn_anti', type=str, default=None)
    parser.add_argument('--model_anti', type=str, default=None)
    parser.add_argument('--model_target_syn', type=str, default=None)
    parser.add_argument('--domain', type=str, default=None)
    parser.add_argument('--weight', type=float, default=None)
    args = parser.parse_args()
    data_files = { "test":"data/slurp/hg_face_data/data/test/*"}
    dataset = load_dataset("audiofolder", data_files=data_files)
    dataset = dataset.filter(lambda example: example["scenario"] == args.domain)
    

    if args.model_syn_anti == None and args.model_anti == None and args.model_target_syn == None and args.weight == None:
        print("Loading model without merge")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to("cuda")
        processor = WhisperProcessor.from_pretrained(args.model_path)
    else:
        print("Loading model and merge")
        model_syn_anti = WhisperForConditionalGeneration.from_pretrained(args.model_syn_anti)
        processor = WhisperProcessor.from_pretrained(args.model_syn_anti)
        model_anti = WhisperForConditionalGeneration.from_pretrained(args.model_anti)
        model_target_syn = WhisperForConditionalGeneration.from_pretrained(args.model_target_syn)
        model = merge(model_syn_anti, model_anti, model_target_syn, args).to("cuda")
        args.model_path = 'merge'
    model.config.forced_decoder_ids = None
    result = dataset['test'].map(map_to_pred)
    wer = load("wer")
    # print(result["prediction"])
    print(args.model_path, 100 * wer.compute(references=result["text"], predictions=result["prediction"]))

