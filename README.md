# Neuromodulatory-Control-Networks
Large Language Models (LLMs) based on the Transformer architecture have achieved remarkable success, yet their core processing mechanisms remain largely static after training. While powerful, this static nature limits their ability to dynamically adapt their processing strategy based on nuanced contextual cues, task demands, or desired operational modes (e.g., shifting between exploration and exploitation). We propose Neuromodulatory Control Networks (NCNs), a novel architectural modification inspired by the neuromodulatory systems in the vertebrate brain (e.g., those utilizing dopamine, acetylcholine, norepinephrine). NCNs are small, parallel networks that receive contextual input, summarizing the global state, task information, or external control signals, and compute dynamic "modulatory signals". These signals are broadcast to the main LLM to influence its computational properties during a forward pass, analogous to how neuromodulators alter neuronal gain, plasticity, and network states. Instead of merely routing information, NCNs aim to change how information is processed throughout the base model by modulating key components like attention mechanisms (e.g., temperature), layer gains, and activation functions. This paper introduces the NCN architecture, details its components and potential mechanisms, discusses its conceptual advantages for enhancing LLM adaptability, controllability, and efficiency, and provides an open-source PyTorch implementation to facilitate community exploration and future empirical validation.

# Example Commands to use the NCN architecture
## Example of Starting a Training Run
python train.py --d_model 384 --n_layer 6 --n_head 6 --ncn_heads 4 --num_epochs 30 --batch_size 4 --block_size 1024 --lr 3e-4 --total_tokens 313228872 --use_amp --checkpoint_frequency 2000 --log_frequency 50 --num_workers auto

## Example of Resuming a Checkpoint
python train.py --resume_checkpoint ncn_checkpoints/checkpoint_step_76000.pt --num_epochs 30 --batch_size 8 --total_tokens 313228872 --use_amp --checkpoint_frequency 2000 --log_frequency 50

## What is the --total_tokens Argument?
Use the --total_tokens argument after counting tokens in your training set once so you don't have to waste time recounting tokens every time you start a new training run or resume training a checkpoint.

# Creation of the Validation Set
## First Run
When you first run the NCN architecture using the train.py script, it will create a tokenized validation set with the default name tokenized_val_data.pt. This is time consuming.

## Starting a New Run or Resuming a Run using the same Training Set
As creating a new tokenized_val_data.pt file is time and resource expensive, if you start a new run or resume an old training run with the same training set, you may keep the tokenized_val_data.pt file and reuse it, saving significant time.
