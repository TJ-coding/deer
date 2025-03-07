# DEcoder-Embedding based Relational KGC/Probe
* This project is still under development, there will likely be disruptive changes.

## Installation
`pip install deer-kgc`

## Encoding Queries
```python
from deer_kgc import convert_tsv_to_prompt

convert_tsv_to_prompt(
    input_file="data/queries.tsv",
    output_file="data/queries_prompt.txt",
    prompt_template="What is the relation between {entity1} and {entity2}?",
    entity1_col="entity1",
    entity2_col="entity2",
)


```

## Encoding Tail-Entities
```python

```
