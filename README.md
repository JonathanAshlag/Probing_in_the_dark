This repository is the official implementation of [PROBING IN THE DARK:
MAXIMUM STATE ENTROPY IN POMDPS]. subbmited to ICLR 2026.

explain how to create env,

for each of the probe environments there is a pretrain file where entropy_type denotes the type of pretraining objective and a fintuning file where task determines the task, pretrain determines whether to load a pretraind model , if so then ass its path at the path argument, bash file for pretraining and finetuning with differenat configurations are here for your comfort.
