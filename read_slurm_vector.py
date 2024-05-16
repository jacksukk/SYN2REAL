from argparse import ArgumentParser
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--ids', type=str, default=None)
    args = parser.parse_args()
    final_table = []
    for i in range(1, 5):
        with open(f'slurm_logs/test/test-{args.ids}_{i}.out', 'r') as f:
            table = f.readlines()
            # print(table)
            final = table[-1].strip().split(' ')[-1]
            # print(final)
            final_table.append(float(final))
        # print(table)
            print(table[25].strip(), final)
            # domain = table[-1].strip().split(' ')[0].split('_')[2]


