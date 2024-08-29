import os
import json
from typing import List


def process_source(file_name: str, source: List[str]):
    with open(file_name, "a") as f:
        for i in source:
            f.write(i)


def process_output(file_name: str, output: List[str]):
    with open(file_name, "a") as f:
        f.write('\n\n')
        for i in output:
            f.write(f"# {i}")


def process_ipynb(file_name: str):
    with open(f"{file_name}.ipynb") as ff:
        a = json.loads(ff.read())
        source = []
        outputs = []

        for i in a['cells']:
            if not ('source' in i and 'outputs' in i):
                continue

            source += (i['source']+['\n'])
            outputs += (i['outputs'][0]['text']+['\n'])

        process_source("first_day.py", source)
        process_output("first_day.py", outputs)


for i in os.listdir('./'):
    file_name, extension = i.split('.')
    if not extension == 'ipynb':
        continue
    print(file_name)
    process_ipynb(file_name)