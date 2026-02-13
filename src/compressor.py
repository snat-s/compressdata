"""
   The structure is mainly based from the works of Yu Mao, et. al.
   https://github.com/mynotwo/A-Fast-Transformer-based-General-Purpose-LosslessCompressor
"""
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import struct
import shutil
# Conditional wandb - only enable if WANDB_API_KEY is set
USE_WANDB = os.environ.get("WANDB_API_KEY") is not None
if USE_WANDB:
    import wandb

import arithmeticcoding
from transformer_model import SLiMPerformer
from model import GPT, GPTConfig
from rwkv_v7 import RWKVv7LM, RWKVv7Config
from gpt_modern import ModernGPT, ModernGPTConfig
from muon_optimizer import configure_optimizers
# from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
# from mamba_ssm.models.config_mamba import MambaConfig
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print(device)

# Defaults (overridable via CLI args)
BATCH_SIZE = 256
SEQ_LENGTH = 8
ENCODE = True
DECODE = False
VOCAB_SIZE = 256
LOG_TRAINING = 10000
SEED = 42
MODEL_TYPE = "modern_gpt"
VOCAB_DIM = 256
HIDDEN_DIM = 256
N_LAYERS = 2
N_HEADS = 8
FEATURE_TYPE = 'sqr'
COMPUTE_TYPE = 'iter'
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.0
USE_MUON = True
MUON_LR = 0.02
MUON_MOMENTUM = 0.95
GRAD_CLIP = 1.0
LR_WARMUP_STEPS = 1000
LR_DECAY_START = 10000
LR_DECAY_POWER = 0.5
WANDB_RUN_NAME = None
WANDB_GROUP = None


def parse_args():
    p = argparse.ArgumentParser(description="Neural network lossless compression")
    # Data
    p.add_argument('--file_path', type=str, default='src/data/alice.txt')
    p.add_argument('--encode_only', action='store_true', help='Only encode (no decode)')
    p.add_argument('--decode_only', action='store_true', help='Only decode (no encode)')
    # Model architecture
    p.add_argument('--model_type', type=str, default='modern_gpt',
                   choices=['gpt', 'rwkv_v7', 'slim_performer', 'modern_gpt'])
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=8)
    # Training
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--seq_length', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4, help='AdamW learning rate')
    p.add_argument('--muon_lr', type=float, default=0.02, help='Muon learning rate')
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--grad_clip', type=float, default=1.0, help='Max gradient norm (0=disable)')
    p.add_argument('--muon_momentum', type=float, default=0.95)
    p.add_argument('--no_muon', action='store_true', help='Disable Muon, use Adam only')
    # LR schedule
    p.add_argument('--warmup_steps', type=int, default=1000)
    p.add_argument('--decay_start', type=int, default=10000)
    p.add_argument('--decay_power', type=float, default=0.5)
    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log_interval', type=int, default=10000)
    # Wandb
    p.add_argument('--wandb_run_name', type=str, default=None, help='Override wandb run name')
    p.add_argument('--wandb_group', type=str, default=None, help='Wandb group for sweep phases')
    return p.parse_args()


def get_lr(step, base_lr, warmup_steps=LR_WARMUP_STEPS,
           decay_start=LR_DECAY_START, decay_power=LR_DECAY_POWER):
    """NNCP-style LR schedule: linear warmup -> constant -> power-law decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    elif step < decay_start:
        return base_lr
    else:
        return base_lr * (decay_start / step) ** decay_power


def build_model(model_type, seq_length):
    """Factory function to create model based on MODEL_TYPE."""
    if model_type == "gpt":
        return GPT(GPTConfig(
            block_size=seq_length, vocab_size=VOCAB_SIZE,
            n_layer=N_LAYERS, n_head=N_HEADS,
            n_embd=VOCAB_DIM, dropout=0.0, bias=True
        ))
    elif model_type == "rwkv_v7":
        return RWKVv7LM(RWKVv7Config(
            block_size=seq_length, vocab_size=VOCAB_SIZE,
            n_layer=N_LAYERS, n_head=N_HEADS,
            n_embd=VOCAB_DIM, dropout=0.0, bias=True
        ))
    elif model_type == "slim_performer":
        return SLiMPerformer(
            VOCAB_SIZE, VOCAB_DIM, HIDDEN_DIM,
            N_LAYERS, FFN_DIM, N_HEADS, FEATURE_TYPE, COMPUTE_TYPE
        )
    elif model_type == "modern_gpt":
        return ModernGPT(ModernGPTConfig(
            block_size=seq_length, vocab_size=VOCAB_SIZE,
            n_layer=N_LAYERS, n_head=N_HEADS,
            n_embd=VOCAB_DIM, dropout=0.0, bias=False,
            use_flash=True  # Enable Flash Attention
        ), use_swiglu=True)  # Use SwiGLU activation
    else:
        raise ValueError(f"Unknown model type: {model_type}")

TMP_DIR = "tmp"
FILE_PATH = "src/data/alice.txt"
COMPRESSED_FILE = None  # Set in main() from args


def init_wandb():
    """Initialize wandb if available, using current global config."""
    if not USE_WANDB:
        return
    dataset_name = os.path.basename(FILE_PATH).replace('.txt', '').replace('.xml', '')
    optimizer_tag = "muon" if USE_MUON else "adam"
    run_name = WANDB_RUN_NAME or f"{MODEL_TYPE}_{N_LAYERS}L_{VOCAB_DIM}d_{N_HEADS}h_{optimizer_tag}_{dataset_name}"
    init_kwargs = dict(
        project="compressdata",
        name=run_name,
    )
    if WANDB_GROUP:
        init_kwargs['group'] = WANDB_GROUP
    wandb.init(
        **init_kwargs,
        config={
            "learning_rate": LEARNING_RATE,
            "muon_lr": MUON_LR,
            "architecture": MODEL_TYPE,
            "dataset": FILE_PATH,
            "n_layers": N_LAYERS,
            "n_embd": VOCAB_DIM,
            "n_heads": N_HEADS,
            "batch_size": BATCH_SIZE,
            "seq_length": SEQ_LENGTH,
            "grad_clip": GRAD_CLIP,
            "muon_momentum": MUON_MOMENTUM,
            "warmup_steps": LR_WARMUP_STEPS,
            "decay_start": LR_DECAY_START,
            "decay_power": LR_DECAY_POWER,
            "weight_decay": WEIGHT_DECAY,
        }
    )


def var_int_encode(byte_str_len, f):
    # print(byte_str_len, end=" ")

    while byte_str_len > 0:
        this_byte = byte_str_len & 127
        byte_str_len >>= 7

        if byte_str_len == 0:
            f.write(struct.pack('B', this_byte))
        else:
            f.write(struct.pack('B', this_byte | 128))


def var_int_decode(f):
    data = f.read(1)
    byte_str_len = 0
    shift = 1

    while True:
        this_byte = struct.unpack('B', data)[0]
        # print(this_byte, end=" ")

        byte_str_len += (this_byte & 127) * shift

        if this_byte & 128 == 0:
            break

        shift <<= 7
        data = f.read(1)

    return byte_str_len


def decode_token(token): return str(chr(token))


def decode_tokens(tokens): return ''.join(list(map(decode_token, tokens)))


def decode(temp_dir, compressed_file, len_series, last):

    iter_num = (len_series - SEQ_LENGTH) // BATCH_SIZE
    ind = np.array(range(BATCH_SIZE))*iter_num
    print(iter_num - SEQ_LENGTH)
    series_2d = np.zeros((BATCH_SIZE, iter_num), dtype=np.uint8).astype('int')

    f = [open(os.path.join(temp_dir, compressed_file)+'.'+str(i), 'rb')
         for i in range(BATCH_SIZE)]
    bitin = [arithmeticcoding.BitInputStream(f[i]) for i in range(BATCH_SIZE)]
    dec = [arithmeticcoding.ArithmeticDecoder(
        32, bitin[i]) for i in range(BATCH_SIZE)]

    prob = np.ones(VOCAB_SIZE)/VOCAB_SIZE
    cumul = np.zeros(VOCAB_SIZE+1, dtype=np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)

    # Decode first K symbols in each stream with uniform probabilities
    for i in range(BATCH_SIZE):
        for j in range(min(SEQ_LENGTH, iter_num)):
            series_2d[i, j] = dec[i].read(cumul, VOCAB_SIZE)

    cumul_batch = np.zeros((BATCH_SIZE, VOCAB_SIZE+1), dtype=np.uint64)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = build_model(MODEL_TYPE, SEQ_LENGTH)
    model = model.to(device)
    print(model)

    # Use Muon optimizer for modern_gpt if enabled, otherwise use Adam
    if MODEL_TYPE == "modern_gpt" and USE_MUON:
        optimizer = configure_optimizers(
            model, muon_lr=MUON_LR, adamw_lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, momentum=MUON_MOMENTUM,
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using Muon optimizer (muon_lr={MUON_LR}, adamw_lr={LEARNING_RATE}, momentum={MUON_MOMENTUM})")
    else:
        optimizer = torch.optim.Adam(model.parameters(
        ), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(.9, .999))

    # if torch.__version__ > "1.0.0":
    #     print("Compiling model")
    #     model = torch.compile(model)
    # training_start = time.time()
    model.train()
    for train_index in trange(iter_num-SEQ_LENGTH):
        # Get current batch
        train_batch = torch.LongTensor(
            series_2d[:, train_index:train_index + SEQ_LENGTH])#.cuda()
        train_batch = train_batch.to(device)

        # Forward pass to get logits for prediction
        logits, _ = model.forward(train_batch)
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()

        cumul_batch[:, 1:] = np.cumsum(prob*10000000 + 1, axis=1)

        # Decode the next byte using predictions from BEFORE weight update
        for i in range(BATCH_SIZE):
            series_2d[i, train_index +
                      SEQ_LENGTH] = dec[i].read(cumul_batch[i, :], VOCAB_SIZE)

        # Now train on this batch (compute loss and update weights)
        # Full-sequence loss: use all positions, not just the last
        label = torch.from_numpy(
            series_2d[:, train_index+1:train_index+SEQ_LENGTH+1]).to(device)
        train_loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            label.reshape(-1),
            reduction='mean')

        if USE_WANDB:
            wandb.log({"loss": train_loss.item()})
        train_loss.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if train_index % LOG_TRAINING == 0:
            print(train_index, ":", train_loss.item()/np.log(2))

    out = open('decompressed_file', 'wb')
    # Write streams sequentially - each stream contains a contiguous chunk
    # Convert to uint8 before writing to ensure proper byte representation
    for i in range(len(series_2d)):
        out.write(series_2d[i].astype(np.uint8).tobytes())
    out.close()

    for i in range(BATCH_SIZE):
        bitin[i].close()
        f[i].close()

    if last:
        out = open('decompressed_file', 'ab')  # Reopen in binary append mode for last part
        series = np.zeros(last, dtype=np.uint8).astype('int')
        f = open(os.path.join(temp_dir, compressed_file)+'.last', 'rb')
        bitin = arithmeticcoding.BitInputStream(f)
        dec = arithmeticcoding.ArithmeticDecoder(32, bitin)
        prob = np.ones(VOCAB_SIZE)/VOCAB_SIZE
        cumul = np.zeros(VOCAB_SIZE+1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        for j in range(last):
            series[j] = dec.read(cumul, VOCAB_SIZE)

        print("Last decode part don't need inference.")
        out.write(series.astype(np.uint8).tobytes())
        print(decode_tokens(series))
        out.close()
        bitin.close()
        f.close()

        if USE_WANDB:
            wandb.finish()
        return


def encode(temp_dir, compressed_file, series, train_data, last_train_data):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    f = [open(os.path.join(temp_dir, compressed_file+'.'+str(i)), 'wb')
         for i in range(BATCH_SIZE)]
    bitout = [arithmeticcoding.BitOutputStream(
        f[i]) for i in range(BATCH_SIZE)]
    enc = [arithmeticcoding.ArithmeticEncoder(
        32, bitout[i]) for i in range(BATCH_SIZE)]

    # Start the probabilities at the same spot
    prob = np.ones(VOCAB_SIZE)/VOCAB_SIZE
    cumul = np.zeros(VOCAB_SIZE+1, np.uint64)
    cumul[1:] = np.cumsum(prob*10000000 + 1)

    iter_num = len(train_data)//BATCH_SIZE
    ind = np.array(range(BATCH_SIZE))*iter_num
    iter_num -= SEQ_LENGTH

    for i in range(BATCH_SIZE):
        for j in range(SEQ_LENGTH):
            enc[i].write(cumul, series[ind[i]+j])

    cumul_batch = np.zeros((BATCH_SIZE, VOCAB_SIZE+1), dtype=np.uint64)
    model = build_model(MODEL_TYPE, SEQ_LENGTH)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"The model has {total_params} parameters")
    # print(model)

    # if torch.__version__ > "1.0.0":
    #     print("Compiling model")
    #     model = torch.compile(model)

    # Use Muon optimizer for modern_gpt if enabled, otherwise use Adam
    if MODEL_TYPE == "modern_gpt" and USE_MUON:
        optim = configure_optimizers(
            model, muon_lr=MUON_LR, adamw_lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY, momentum=MUON_MOMENTUM,
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using Muon optimizer (muon_lr={MUON_LR}, adamw_lr={LEARNING_RATE}, momentum={MUON_MOMENTUM})")
    else:
        optim = torch.optim.Adam(
            model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    print("Number of iterations", iter_num)
    for train_index in trange(iter_num):
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]
        train_batch = torch.from_numpy(train_batch).long()#.cuda().long()
        train_batch = train_batch.to(device)
        train_loss, logits = model.full_loss(train_batch, with_grad=True)
        if USE_WANDB:
            wandb.log({"loss": train_loss.item()})
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optim.step()
        optim.zero_grad(set_to_none=True)

        logits = logits.transpose(1, 2)
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        cumul_batch[:, 1:] = np.cumsum(prob*10000000 + 1, axis=1)

        for i in range(BATCH_SIZE):
            enc[i].write(cumul_batch[i, :], y[i])

        ind += 1
        # Log compressed size every 50 steps for real-time tracking
        if train_index % 50 == 0:
            # Flush all file buffers so os.path.getsize reflects actual data
            for fi in f:
                fi.flush()
            size = 0
            for cf in os.listdir(temp_dir):
                size += os.path.getsize(temp_dir+"/"+cf)
            bytes_processed = (train_index + 1 + SEQ_LENGTH) * BATCH_SIZE
            running_bpb = (size * 8) / bytes_processed if bytes_processed > 0 else 0
            projected_ratio = bytes_processed / size if size > 0 else 0
            if USE_WANDB:
                wandb.log({
                    "size_bytes": size,
                    "size_mb": size / (1024 * 1024),
                    "running_bpb": running_bpb,
                    "projected_ratio": projected_ratio,
                })
        if train_index % LOG_TRAINING == 0:
            size = 0
            for cf in os.listdir(temp_dir):
                size += os.path.getsize(temp_dir+"/"+cf)
            print(train_index, ":", train_loss.item() /
                  np.log(2), "size:", size/(1024*1024))

    for i in range(BATCH_SIZE):
        enc[i].finish()
        bitout[i].close()
        f[i].close()

    if last_train_data is not None:

        print("last series")
        f = open(os.path.join(temp_dir, compressed_file)+'.last', 'wb')
        bitout = arithmeticcoding.BitOutputStream(f)
        enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

        prob = np.ones(VOCAB_SIZE)/VOCAB_SIZE
        cumul = np.zeros(VOCAB_SIZE+1, dtype=np.uint64)
        cumul[1:] = np.cumsum(prob*10000000 + 1)

        for j in range(len(last_train_data)):
            enc.write(cumul, last_train_data[j])

        print("Last encode part don't need inference.")

        enc.finish()
        bitout.close()
        f.close()


def main():
    global BATCH_SIZE, SEQ_LENGTH, ENCODE, DECODE, SEED, LOG_TRAINING
    global MODEL_TYPE, VOCAB_DIM, HIDDEN_DIM, N_LAYERS, N_HEADS
    global LEARNING_RATE, WEIGHT_DECAY, USE_MUON, MUON_LR, MUON_MOMENTUM
    global GRAD_CLIP, LR_WARMUP_STEPS, LR_DECAY_START, LR_DECAY_POWER
    global FILE_PATH, COMPRESSED_FILE, WANDB_RUN_NAME, WANDB_GROUP

    args = parse_args()

    # Override globals from CLI args
    BATCH_SIZE = args.batch_size
    SEQ_LENGTH = args.seq_length
    SEED = args.seed
    LOG_TRAINING = args.log_interval
    MODEL_TYPE = args.model_type
    VOCAB_DIM = args.n_embd
    HIDDEN_DIM = args.n_embd
    N_LAYERS = args.n_layers
    N_HEADS = args.n_heads
    LEARNING_RATE = args.lr
    WEIGHT_DECAY = args.weight_decay
    USE_MUON = not args.no_muon
    MUON_LR = args.muon_lr
    MUON_MOMENTUM = args.muon_momentum
    GRAD_CLIP = args.grad_clip
    LR_WARMUP_STEPS = args.warmup_steps
    LR_DECAY_START = args.decay_start
    LR_DECAY_POWER = args.decay_power
    FILE_PATH = args.file_path
    WANDB_RUN_NAME = args.wandb_run_name
    WANDB_GROUP = args.wandb_group

    if args.encode_only:
        ENCODE, DECODE = True, False
    elif args.decode_only:
        ENCODE, DECODE = False, True
    else:
        ENCODE, DECODE = True, False  # default: encode only

    # Derived config
    dataset_name = os.path.basename(FILE_PATH).split('.')[0]
    COMPRESSED_FILE = f"{dataset_name}_{N_LAYERS}L_{VOCAB_DIM}d_{N_HEADS}h_b{BATCH_SIZE}"
    SEQ_LENGTH = SEQ_LENGTH * (HIDDEN_DIM // VOCAB_DIM)

    print(f"Config: {MODEL_TYPE} {N_LAYERS}L {VOCAB_DIM}d {N_HEADS}h | "
          f"batch={BATCH_SIZE} seq={SEQ_LENGTH} | "
          f"lr={LEARNING_RATE} muon_lr={MUON_LR} clip={GRAD_CLIP}")

    init_wandb()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    temp_dir = TMP_DIR
    file_path = FILE_PATH
    compressed_file = COMPRESSED_FILE

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
        print(f"Created directory at {temp_dir}")

    def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
        nrows = ((a.size - L) // S) + 1
        n = a.strides[0]
        return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

    with open(file_path, "rb") as f:
        series = np.frombuffer(f.read(), dtype=np.uint8)

    training_data = strided_app(series, SEQ_LENGTH+1, 1)
    total_length = len(training_data)

    if ENCODE and total_length % BATCH_SIZE == 0:
        encode(temp_dir, compressed_file, series, training_data, None)
    elif ENCODE:
        last_lines = total_length // BATCH_SIZE * BATCH_SIZE
        encode(temp_dir, compressed_file,
               series[:last_lines + SEQ_LENGTH], training_data[:last_lines],
               series[last_lines:])

    if ENCODE:
        # combine compressed results
        f = open(compressed_file+'.compressed', 'wb')

        for i in range(BATCH_SIZE):
            f_in = open(os.path.join(
                temp_dir, compressed_file+'.'+str(i)), 'rb')
            byte_str = f_in.read()
            byte_str_len = len(byte_str)
            var_int_encode(byte_str_len, f)
            f.write(byte_str)
            f_in.close()

        if total_length % BATCH_SIZE != 0:
            f_in = open(os.path.join(temp_dir, compressed_file)+'.last', 'rb')
            byte_str = f_in.read()
            var_int_encode(len(byte_str), f)
            f.write(byte_str)
            f_in.close()
        f.close()

        total = 0

        for ff in os.listdir(temp_dir):
            total += os.path.getsize(os.path.join(temp_dir, ff))

        print(total/(1024*1024))
        shutil.rmtree(temp_dir)
        if USE_WANDB:
            wandb.finish()

    if DECODE:

        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
            print(f"Created directory at {temp_dir}")
        f = open(compressed_file+'.compressed', 'rb')
        len_series = len(series)

        for i in range(BATCH_SIZE):
            f_out = open(os.path.join(
                temp_dir, compressed_file)+'.'+str(i), 'wb')
            byte_str_len = var_int_decode(f)
            byte_str = f.read(byte_str_len)
            f_out.write(byte_str)
            f_out.close()

        f_out = open(os.path.join(temp_dir, compressed_file)+'.last', 'wb')
        byte_str_len = var_int_decode(f)
        byte_str = f.read(byte_str_len)
        f_out.write(byte_str)
        f_out.close()
        f.close()

        len_series = len(series)

        if (len_series-SEQ_LENGTH) % BATCH_SIZE == 0:
            decode(temp_dir, compressed_file, len_series, 0)
        else:
            last_lines = (len_series-SEQ_LENGTH) % BATCH_SIZE + SEQ_LENGTH
            decode(temp_dir, compressed_file, len_series, last_lines)

        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
