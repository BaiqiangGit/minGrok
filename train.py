
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #1: import Libs ')
import sys
sys.path.append('D:\conda\minGrok\Lib\site-packages')
import dataclasses
from model import *
from tokenizer import SimpleTokenizer, loaded_stoi, loaded_merges


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #2: prepare tokenizer')

tokenizer = SimpleTokenizer(loaded_stoi, loaded_merges)
print("vocab length: ", tokenizer.vocab_len)

# Encoding text
encoded_text = tokenizer.encode("JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?")
print("Encoded:", encoded_text)

# Decoding back
decoded_text = tokenizer.decode(encoded_text)
print("Decoded:", decoded_text)


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #3: prepare config')

@dataclasses.dataclass
class Config:
    # v was defined earlier when we loaded TinyShakespeare. In Grok it's 131,072
    vocab_size: int = tokenizer.vocab_len

    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 256 # 256 # in Grok it's 8,192

    # The number of layers in the model.
    num_layers: int = 8 # In Grok it's 64

    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 8 # 4 # In Grok it's 48

    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 1 # In Grok it's 8

    # The hidden size of the model, AKA the embedding dimension. Each token embedding vector will be this long
    hidden_size: int = 128 # 96 # In Grok it's 6,144

    # How much wider should the inner dimension of the experts be than the model's embedding dimension?
    embedding_multiplier_scale: int = 2 # In Grok it's roughly 5.33

    # how many experts?
    tot_num_experts: int = 4 # 4 # in Grok it's 8

    # how many active experts per token?
    chosen_num_experts: int = 2 # in Grok it's also 2

    # what amount of noise should be injected into the router during training?
    noise_std = 0.1 # the value for Grok has not been shared

    # When we create a loss to encourage all experts to be used, how should that loss be weighted?
    lambadada = 10 # Grok's value has not been shared
    # excuse my silly naming

    # The number of head dimensions
    head_dim: int = 24 # In Grok it's 128

    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-5 # this is to promote numerical stability & prevent dividing by 0

    # the scaling factor that determines the frequencies for the rotary positional encodings
    rope_theta = 100.0 # Grok and most models use 10,000
    # smaller models should use a smaller theta, but I'm just guessing here. 1000 might work too

    # whether to use a linear layer after normalization
    use_scale: bool = True # same in Grok

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # the dropout rate to use during training
    dropout = 0.05
    
config = Config()
print(config)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #4: prepare dataset')

# load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text and how many there are
chars = sorted(list(set(text)))
print('unique chars:', chars)
print('chars length:', len(chars))

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be our training dataset, the rest for validation
train_data = data[:n]
val_data = data[n:]


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #5: prepare batch/eval func ')
# data loading for training which generates a small batch of data of inputs x and targets y
def get_batch(split, batch_size):
    # whether we grab from our training or validation dataset
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.max_position_embeddings, (batch_size,))
    x = torch.stack([data[i:i+config.max_position_embeddings] for i in ix])
    y = torch.stack([data[i+1:i+config.max_position_embeddings+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters = 10): # to periodically estimate loss during the training loop
    out = {}
    model.eval() # sets model to eval mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    input_str = "JULIET:\nO Romeo, Romeo! wherefore art thou" # the classic line
    max_useable_output_len = config.max_position_embeddings - len(input_str)
    output = model.generate(input_str, output_len = max_useable_output_len)
    print(output)
    model.train() # just resets to training mode
    return out


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #6: init model and optim')


# instantiate a new model
model = minGrok(config, tokenizer).to(config.device)


# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e3, 'K parameters')

# create a PyTorch optimizer
# this is not what they used, but this learning rate & weight decay work for our tiny minGemma
learning_rate = 3e-4 # 1e-5 # used 3e-4 for the first 5000 iters
weight_decay = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #7: training model')

# how long we want to train for
max_iters = 10000

# how often we want to check & see how our loss is doing
eval_interval = 50

# batch size to use
batch_size = 32 # 128

import time as time

start_time = time.time()


# Enable anomaly detection. uncomment these lines if you need to do extensive debugging
#torch.autograd.set_detect_anomaly(True)

for iter in range(max_iters):

    # sample a batch of data
    xb, yb = get_batch('train', batch_size)

    # train
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, batch_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")

        input_str = "JULIET:\nO Romeo, Romeo! wherefore art thou" # the classic line
        max_useable_output_len = config.max_position_embeddings - len(input_str)
        output = model.generate(input_str, output_len = max_useable_output_len)
        print(output)

# Disable anomaly detection after the training loop
#torch.autograd.set_detect_anomaly(False)

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Step #8: save model')

# save the model currently held in memory
# the filename specifies the model's class, hyperparameters, and date/time it was saved
import os

# Ensure the directory exists
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Create a shorter, more concise filename
filename = (f'{model.__class__.__name__}'
            f'-v{config.vocab_size}'
            f'-max_t{config.max_position_embeddings}'
            f'-layers{config.num_layers}'
            f'-heads{config.num_attention_heads}'
            f'-kv_heads{config.num_key_value_heads}'
            f'-hidden{config.hidden_size}'
            f'-embedding_multiplier_scale{config.embedding_multiplier_scale}'
            f'-head_dim{config.head_dim}'
            f'-theta{config.rope_theta}'
            f'-lr{learning_rate}'
            f'-decay{weight_decay}'
            f'-tot_num_experts{config.tot_num_experts}'
            f'-chosen_num_experts{config.chosen_num_experts}'
            f'-use_scale{config.use_scale}'
            f'-batch{batch_size}'
            f'-train_iter{max_iters}'
            f'--{time.strftime("%Y-%m-%d_%H-%M-%S")}.pth')

# Save the model
model_path = os.path.join(model_dir, filename)
torch.save(model.state_dict(), model_path)
