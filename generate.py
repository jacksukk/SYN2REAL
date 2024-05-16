import pandas as pd 
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from tqdm import tqdm
from argparse import ArgumentParser
preload_models()
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
    for text_prompt, name in zip(texts, file_names):
        for i in range(5):
            audio_array = generate_audio(text_prompt, silent=True)

            # save audio to disk
            write_wav(f"data/synthetic/{domains[0]}/audio_{name}_{i}.wav", SAMPLE_RATE, audio_array)
            new_file_names.append(f"audio_{name}_{i}.wav")
            new_texts.append(text_prompt)
            new_scenario.append(domains[0])
            pbar.update(1)
    pbar.close()
    new_table = pd.DataFrame({'file_name':new_file_names, 'text':new_texts, 'scenario':new_scenario})
    new_table.to_csv(f"data/synthetic/{domains[0]}/metadata.csv", index=False)
            # play text in notebook
    