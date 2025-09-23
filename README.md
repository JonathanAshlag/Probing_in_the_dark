This repository is the official implementation of [PROBING IN THE DARK:
MAXIMUM STATE ENTROPY IN POMDPS]. subbmited to ICLR 2026.
--
## Requirements

```bash
conda env create -f env.yml
```
## Experiments
For each probe environment, there is a pretraining file named accordingly where entropy_type denotes the type of pretraining objective.
- **`latent`** – Our algorithm.
- **`state`** –  Privilleged information maximum state entropy.
- **`obs`** –  Maximum observation entropy.

Additionaly There is a finetuning file where --task determines the task and init_type controls whether to load a pretrained model; if so, provide its path via the path argument..

Bash files for pretraining and finetuning with different configurations are provided for your convenience.
