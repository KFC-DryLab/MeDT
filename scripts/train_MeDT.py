import argparse
import os

import torch
from datasets import mimic_dataset
from models.GPT import GPTConfig
from models.MeDT import MeDT
from trainer.trainer_MeDT import Trainer, TrainerConfig

################################################################################################################
# Constants
#
################################################################################################################
CONTEXT_LENGTH = 20
RTG_SCALE = 1
VOCAB_SIZE = 25
WARMPUP_TOKENS = 512*20
sub_blocks = {'BC':CONTEXT_LENGTH*2, 'DT':CONTEXT_LENGTH*3, 'MeDT':CONTEXT_LENGTH*10, 
                'PromptDT':CONTEXT_LENGTH*2+1, 'PromptDTATG':CONTEXT_LENGTH*2+8}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################################################
# Helper Functions
################################################################################################################

def get_subset(dataset, percentage):
    """Get a subset of the dataset based on the given percentage."""
    subset_size = int(len(dataset) * percentage)
    return torch.utils.data.Subset(dataset, range(subset_size))

################################################################################################################
# train
#
# This is where learning happens. More specifically, the policy network is trained to predict actions  
# with teacher forcing using cross-entropy loss.
#
#
################################################################################################################
def train(args, percentage):

    # Experiment name 
    exp_name = f'{args.model_type}v1_ID{args.id}_nLayer{args.n_layer}_nHead{args.n_head}_nEmb{args.n_embd}_LR{args.learningrate}_BS{args.batchsize}_SEED{args.seed}_P{int(percentage * 100)}'
    
    # Load dataset
    train_path = os.path.join(args.datadir, 'train_Phys45.pickle')
    eval_path = os.path.join(args.datadir, 'val_Phys45.pickle')
    test_path = os.path.join(args.datadir, 'test_Phys45.pickle')

    train_dataset = mimic_dataset.MimicTrajectoryDataset(train_path, CONTEXT_LENGTH, RTG_SCALE)
    eval_dataset = mimic_dataset.MimicTrajectoryDataset(eval_path, CONTEXT_LENGTH, RTG_SCALE)
    test_dataset = mimic_dataset.MimicTrajectoryDataset(test_path, CONTEXT_LENGTH, RTG_SCALE)
    
    train_dataset = get_subset(train_dataset, percentage)

    # Build model
    block_size = sub_blocks[args.model_type]
    mconf = GPTConfig(VOCAB_SIZE, block_size, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, 
                    model_type=args.model_type, max_timestep=CONTEXT_LENGTH)
    model = MeDT(mconf)

    # Train model
    tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batchsize, 
                      learning_rate=args.learningrate,lr_decay=args.lr_decay, warmup_tokens=WARMPUP_TOKENS, 
                      final_tokens=2*len(train_dataset)*sub_blocks[args.model_type], num_workers=args.num_workers, 
                      seed=args.seed, model_type=args.model_type, max_timestep=CONTEXT_LENGTH)
    trainer = Trainer(model, train_dataset, eval_dataset, test_dataset, exp_name, tconf)
    trainer.train(args=args)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--epochs", "-e", type=int, default=150)
    parser.add_argument("--batchsize", "-b", type=int, default=64)
    parser.add_argument("--learningrate", "-lr", type=float, default=6e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type=str, default="./cgrd_medDT/logs")
    parser.add_argument("--id", type=str, default="0")
    parser.add_argument("--loadfile", "-l", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--num_workers", "-nw", type=int, default=0)
    parser.add_argument("--lr_decay", "-ld", action="store_true")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--model_type", type=str, default='MeDT')
    args = parser.parse_args()
    
    percentages = [0.5, 0.75, 1.0]
    for percentage in percentages:
        train(args, percentage)