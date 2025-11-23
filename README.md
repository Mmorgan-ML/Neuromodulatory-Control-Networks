# Neuromodulatory-Control-Networks
Large Language Models (LLMs) based on the Transformer architecture have achieved remarkable success, yet their core processing mechanisms remain largely static after training. While powerful, this static nature limits their ability to dynamically adapt their processing strategy based on nuanced contextual cues, task demands, or desired operational modes (e.g., shifting between exploration and exploitation). We propose Neuromodulatory Control Networks (NCNs), a novel architectural modification inspired by the neuromodulatory systems in the vertebrate brain (e.g., those utilizing dopamine, acetylcholine, norepinephrine). NCNs are small, parallel networks that receive contextual input, summarizing the global state, task information, or external control signals, and compute dynamic "modulatory signals". These signals are distributed as layer-specific control vectors to the main LLM to influence its computational properties during a forward pass, analogous to how neuromodulators alter neuronal gain, plasticity, and network states across different cortical depths. Instead of merely routing information, NCNs aim to change *how* information is processed throughout the base model by modulating key components like attention mechanisms (e.g., via precision scaling), layer gains, and activation functions. Crucially, the architecture allows the model to *implicitly learn* to self-regulate these parameters via backpropagation, effectively becoming its own "tuning expert." We further introduce formal stability mechanisms, including homeostatic regularization, to prevent control manifold collapse. This paper introduces the NCN architecture, details its components and implicit learning mechanism, discusses its conceptual advantages and potential failure modes (such as contextual stereotyping), and provides an open-source PyTorch implementation to facilitate community exploration and future empirical validation. 

# What does it do? How does it work?
The Neuromodulatory Control Network architecture operates by running a compact neural network in parallel with the main LLM. When the system processes an input sequence, the NCN generates a latent representation, consisting of a sequence of 768-dimensional vectors, that captures the specific "texture" of the input. During training, the network uses end-to-end gradient modulation to dynamically adjust the attention temperature, layer gain, and feed-forward gating, implicitly learning which parameter states minimize loss for different contexts. For example, if a user asks a standard math question, the NCN detects the context and lowers the temperature to encourage fact recall, whereas asking the model to write a poem results in the NCN increasing temperature to foster creativity. We recently updated the architecture to make these representations "phasic" rather than "tonic," meaning the network now distinguishes between sequences that share the same words but in different orders. While a tonic representation might generate identical embeddings for "The dog chased the cat" and "The cat chased the dog," the phasic approach produces distinct values for each. This prevents the system from overfitting on keywords, ensuring that while rote calculation triggers a low-temperature state, the NCN can still apply high creativity to complex prompts like "Create a new mathematical conjecture about black holes" or "Unify Knot Theory and Number Theory" despite the mathematical vocabulary.

# Example Commands to use the NCN architecture
## Example of Starting a Training Run
python train.py --d_model 384 --n_layer 6 --n_head 6 --ncn_heads 4 --num_epochs 30 --batch_size 4 --gradient_accumulation_steps 16 --block_size 1024 --lr 3e-4 --total_tokens 313228872 --use_amp --checkpoint_frequency 1000 --log_frequency 50 --num_workers auto

## Example of Resuming a Checkpoint
python train.py --resume_checkpoint ncn_checkpoints/checkpoint_step_76000.pt --num_epochs 30 --batch_size 4 --gradient_accumulation_steps 16 --total_tokens 313228872 --use_amp --checkpoint_frequency 1000 --log_frequency 50

## What is the --total_tokens Argument?
Use the --total_tokens argument after counting tokens in your training set once so you don't have to waste time recounting tokens every time you start a new training run or resume training a checkpoint.

# Creation of the Validation Set
## First Run
When you first run the NCN architecture using the train.py script, it will create a tokenized validation set using the files in the training_data directory with the default name tokenized_val_data.pt. This is time consuming.

## Starting a New Run or Resuming a Run using the same Training Set
As creating a new tokenized_val_data.pt file is time and resource expensive, if you start a new run or resume an old training run with the same training set, you may keep the tokenized_val_data.pt file and reuse it, saving significant time.

# Current Progress / Work
Tests are currently being performed training a 31M parameter model on 4.4GB of .txt files. We are very compute constrained, so this will take quite a while. We ask the community to help in validating the architecture and the role the NCN plays in modulating the main LLM. Here are some pictures of the perplexity falling over the course of training.

This is an older version of the architecture without multi-head attention. This architecture, training on a GTX 1650, was averaging 1760 tokens/s. After one epoch of training, it had a Final interval Loss: 2.4735 and PPL: 11.86, along with Avg Validation Loss: 2.2904 | Perplexity: 9.8792. This version has since become deprecated.
<img width="1200" height="700" alt="Figure_2 500 step" src="https://github.com/user-attachments/assets/2fdbc15f-2e93-45ad-9c41-cfc35b4d1bcc" />

This is the current architecture undergoing its first training run. Not only is it now averaging 1970 tokens/s, but according to analysis of perplexity drops from the log file, it seems to be better at converging. Additional training is required, along with validation perplexity scores, to confirm its feasibility. This run has finished its first epoch. Final interval Loss: 2.5449, PPL: 12.74 and Validation Avg Loss: 2.1264 | Perplexity: 8.3846 
<img width="3600" height="2100" alt="smoothed_perplexity_plot" src="https://github.com/user-attachments/assets/918426e7-a95e-4594-970a-51477144cc9e" />


## Future Work
While we're currently training models using the architecture, we must consider future work. Some future work ideas are laid out in the accompanying paper, but there are more near term goals. After fully converged models are made (either this 31M parameter model, or perhaps a larger model if 31M proves too small), a generation / inference script will be written to produce text using the model. After this, a method will be devised to examine the attention / temperature, layer gains, and FF gating of the model as it undergoes inference, to see if the NCN is successfully modulating the outputs of the main LLM.




