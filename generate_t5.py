import pandas as pd 
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from tqdm import tqdm
import torch
import random
from argparse import ArgumentParser
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, default=None)

    args = parser.parse_args()

    table = pd.read_csv('data/slurp/hg_face_data/data/train/metadata.csv')
    domains = args.domains.split(';')
    selected_table = table[table['scenario'].isin(domains)]  
    print(selected_table)  
    texts = set(selected_table['text'].tolist())
    texts = list(texts)
    file_names = [i for i in range(len(texts))]
    pbar = tqdm(total=len(texts) * 5)
    new_file_names = []
    new_texts = []
    new_scenario = []
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    

    for text_prompt, name in zip(texts, file_names):
        for i in range(5):
            speaker_id = random.randint(0, len(embeddings_dataset)-1)
            speaker_embedding = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
            # save audio to disk
            speech = synthesiser(text_prompt, forward_params={"speaker_embeddings": speaker_embedding})

            sf.write(f"data/speech_t5/{domains[0]}/audio_{name}_{i}.wav", speech["audio"], samplerate=speech["sampling_rate"])
            new_file_names.append(f"audio_{name}_{i}.wav")
            new_texts.append(text_prompt)
            new_scenario.append(domains[0])
            pbar.update(1)
    pbar.close()
    new_table = pd.DataFrame({'file_name':new_file_names, 'text':new_texts, 'scenario':new_scenario})
    new_table.to_csv(f"data/speech_t5/{domains[0]}/metadata.csv", index=False)
            # play text in notebook
    