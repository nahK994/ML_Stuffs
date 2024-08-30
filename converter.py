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
    ipynb_file = f"{file_name}.ipynb"
    py_file = f"{file_name}.py"

    with open(ipynb_file) as ff:
        source = []
        outputs = []
        a = json.loads(ff.read())
        for i in a['cells']:
            if not ('source' in i and 'outputs' in i):
                continue

            source += [] if len(i['source']) == 0 else (i['source']+['\n'])
            outputs += [] if len(i['outputs']) == 0 else (i['outputs'][0]['text']+['\n'])

        if os.path.exists(ipynb_file):
            os.remove(ipynb_file)
        if os.path.exists(py_file):
            os.remove(py_file)

        process_source(py_file, source)
        process_output(py_file, outputs)


for i in os.listdir('./'):
    file_name, extension = i.split('.')
    if not extension == 'ipynb':
        continue
    print(file_name)
    process_ipynb(file_name)
