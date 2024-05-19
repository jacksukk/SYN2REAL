from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC, EarlyStoppingCallback, Wav2Vec2FeatureExtractor
import torch
from evaluate import load
from argparse import ArgumentParser
import random

# def map_to_pred(batch):
#     inputs = processor(batch["audio"]["array"], return_tensors="pt", padding="longest")
#     input_values = inputs.input_values.to("cuda")
#     attention_mask = inputs.attention_mask.to("cuda")
    
#     with torch.no_grad():
#         logits = model(input_values, attention_mask=attention_mask).logits

#     predicted_ids = torch.argmax(logits, dim=-1)
#     transcription = processor.batch_decode(predicted_ids)
#     batch["transcription"] = transcription
#     return batch

def map_to_pred(batch):
    inputs = processor(batch["audio"]["array"], return_tensors="pt", padding="longest", sampling_rate=16000)
    input_values = inputs.input_values.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    batch["prediction"] = [x.lower() for x in transcription][0]
    print("123###", batch["prediction"])
    print(batch["text"])
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
        model = Wav2Vec2ConformerForCTC.from_pretrained(args.model_path).to("cuda")
        processor = Wav2Vec2Processor.from_pretrained(args.model_path)
    else:
        print("Loading model and merge")
        model_syn_anti = Wav2Vec2ConformerForCTC.from_pretrained(args.model_syn_anti)
        processor = Wav2Vec2Processor.from_pretrained(args.model_syn_anti)
        model_anti = Wav2Vec2ConformerForCTC.from_pretrained(args.model_anti)
        model_target_syn = Wav2Vec2ConformerForCTC.from_pretrained(args.model_target_syn)
        model = merge(model_syn_anti, model_anti, model_target_syn, args).to("cuda")
        args.model_path = 'merge'
    model.config.forced_decoder_ids = None
    result = dataset['test'].map(map_to_pred)
    wer = load("wer", experiment_id=str(random.random()))
    # print(result["prediction"])
    print(args.model_path, 100 * wer.compute(references=result["text"], predictions=result["prediction"]))

