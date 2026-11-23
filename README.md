# Grounding Methods for Neural-Symbolic AI

[![IJCAI 2025](https://img.shields.io/badge/IJCAI-2025-blue)](https://doi.org/10.24963/ijcai.2025/535)
[![arXiv](https://img.shields.io/badge/arXiv-2507.08216-b31b1b)](https://arxiv.org/abs/2507.08216)
[![PDF](https://img.shields.io/badge/PDF-open%20access-green)](https://www.ijcai.org/proceedings/2025/0535.pdf)

Official code for the IJCAI 2025 paper **"Grounding Methods for Neural-Symbolic AI"**
by Rodrigo Castellano Ontiveros, Francesco Giannini, Marco Gori, Giuseppe Marra and Michelangelo Diligenti.

📄 Paper: [IJCAI proceedings](https://doi.org/10.24963/ijcai.2025/535) · [arXiv](https://arxiv.org/abs/2507.08216) · [PDF](https://www.ijcai.org/proceedings/2025/0535.pdf)

## TL;DR

Neural-symbolic (NeSy) models that reason with First-Order Logic must *ground* their rules,
i.e. decide which substitutions of entities to consider. Exhaustive grounding does not scale,
while heuristic grounding has no guarantees. We propose a **parametrized family of grounding
methods that generalizes Backward Chaining**: it recovers commonly used grounders as special
cases and lets you control the trade-off between expressiveness and scalability.
Experiments on knowledge-graph and relational benchmarks (Countries, Kinship, WN18RR, FB15k-237) show that **the choice of grounding
method is often as important as the choice of the NeSy model itself**.

## Installation

```bash
conda env create -f env.yml
conda activate grounders
```

## Training

To train the models presented in the paper, go to the `experiments` folder and run `runner.py`. Example:

```bash
python experiments/runner.py --d wn18rr --m r2n
```

## Data

The datasets (Countries S1–S3, Kinship, WN18RR, FB15k-237 and the ablation sets) are in `experiments/data`.

## Citation

If you use this code or build on these grounding methods, please cite:

```bibtex
@inproceedings{castellanoontiveros2025grounding,
  title     = {Grounding Methods for Neural-Symbolic {AI}},
  author    = {Castellano Ontiveros, Rodrigo and Giannini, Francesco and Gori, Marco and Marra, Giuseppe and Diligenti, Michelangelo},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  pages     = {4806--4814},
  year      = {2025},
  doi       = {10.24963/ijcai.2025/535},
  url       = {https://doi.org/10.24963/ijcai.2025/535}
}
```

## Authors

Rodrigo Castellano Ontiveros and Michelangelo Diligenti.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
