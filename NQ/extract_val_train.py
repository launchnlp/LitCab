import json

splits = ['dev', 'test', 'train']
for split in splits:
    file_name = f'./{split}.json'
    output_file_name = f'./{split}.txt'
    with open(file_name, 'r') as f, open(output_file_name, 'w') as g:
        data = json.load(f)
        for item in data:
            # print(item.keys())
            g.write(item['question'] + '\n' + item['answers'][0] + '\n\n')