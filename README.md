---
license: cc-by-nc-4.0
tags:
- merge
- mergekit
- lazymergekit
- LDCC/LDCC-SOLAR-10.7B
- upstage/SOLAR-10.7B-Instruct-v1.0
---

# SOLAR-10.7B-slerp

SOLAR-10.7B-slerp is a merge of the following models using [mergekit](https://github.com/cg123/mergekit):
* [LDCC/LDCC-SOLAR-10.7B](https://huggingface.co/LDCC/LDCC-SOLAR-10.7B)
* [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)

## 🧩 Configuration

```yaml
slices:
  - sources:
      - model: LDCC/LDCC-SOLAR-10.7B
        layer_range: [0, 48]
      - model: upstage/SOLAR-10.7B-Instruct-v1.0
        layer_range: [0, 48]
merge_method: slerp
base_model: upstage/SOLAR-10.7B-Instruct-v1.0
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
tokenizer_source: union
dtype: float16

```