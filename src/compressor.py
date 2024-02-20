"""
   The structure is mainly based from the works of Yu Mao, et. al.
   https://github.com/mynotwo/A-Fast-Transformer-based-General-Purpose-LosslessCompressor
"""
import numpy as np
import os
import torch
import torch.nn.functional as F
import struct
import shutil
import wandb

import arithmeticcoding
from transformer_model import SLiMPerformer
from model import GPT, GPTConfig
from mamba_ssm import Mamba
from tqdm import trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
SHOULD_SAVE = True
SEQ_LENGTH = 8
ENCODE = True
DECODE = False
VOCAB_SIZE = 256
LOG_TRAINING = 1000
SEED = 42
TMP_DIR = "tmp"
FILE_PATH = "data/enwik8"
COMPRESSED_FILE = "enwik8_1"

# MODEL CONFIG
VOCAB_DIM = 64
HIDDEN_DIM = 256
N_LAYERS = 1
FFN_DIM = 4096
N_HEADS = 8
FEATURE_TYPE = 'sqr'
COMPUTE_TYPE = 'iter'
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
# END MODEL CONFIG


# WANDB config
wandb.init(
    # set the wandb project where this run will be logged
    project="compressdata",

    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "RWKV",
        "dataset": FILE_PATH,
        "epochs": 1,
        "n_layers": N_LAYERS,
        "ENCODE": ENCODE,
        "DECODE": DECODE,
    }
)
# END WANDB config


def var_int_encode(byte_str_len, f):
    #print(byte_str_len, end=" ")

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
        #print(this_byte, end=" ")

        byte_str_len += (this_byte & 127) * shift

        if this_byte & 128 == 0:
            break

        shift <<= 7
        data = f.read(1)

    return byte_str_len


def decode_token(token): return str(chr(max(32, token)))


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

    model = GPT(GPTConfig(block_size=SEQ_LENGTH, vocab_size=VOCAB_SIZE,
                          n_layer=N_LAYERS, n_head=N_HEADS,
                          n_embd=VOCAB_DIM, dropout=0.0, bias=True))
    model = model.to(device)
    #model = SLiMPerformer(VOCAB_SIZE, VOCAB_DIM, HIDDEN_DIM,
    #                     N_LAYERS, FFN_DIM, N_HEADS, FEATURE_TYPE,
    #                     COMPUTE_TYPE).cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(
    ), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(.9, .999))

    # training_start = time.time()
    model.train()
    for train_index in trange(iter_num-SEQ_LENGTH):
        train_batch = torch.LongTensor(
            series_2d[:, train_index:train_index + SEQ_LENGTH]).cuda()
        logits, _ = model.forward(train_batch)
        #logits = model.forward(train_batch)
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()

        cumul_batch[:, 1:] = np.cumsum(prob*10000000 + 1, axis=1)

        # Decode with Arithmetic Encoder
        for i in range(BATCH_SIZE):
            series_2d[i, train_index +
                      SEQ_LENGTH] = dec[i].read(cumul_batch[i, :], VOCAB_SIZE)

        logits = logits.transpose(1, 2)
        label = torch.from_numpy(
            series_2d[:, train_index+1:train_index+SEQ_LENGTH+1]).cuda()
        train_loss = torch.nn.functional.cross_entropy(
            logits[:, :, -1], label[:, -1], reduction='mean')

        wandb.log({"loss": train_loss})
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if train_index % LOG_TRAINING == 0:
            print(train_index, ":", train_loss.item()/np.log(2))

    out = open('decompressed_file', 'w')
    for i in range(len(series_2d)):
        out.write(decode_tokens(series_2d[i]))

    for i in range(BATCH_SIZE):
        bitin[i].close()
        f[i].close()

    if last:
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
        out.write(decode_tokens(series))
        print(decode_tokens(series))
        bitin.close()
        f.close()

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
    model = GPT(GPTConfig(block_size=SEQ_LENGTH, vocab_size=VOCAB_SIZE,
                          n_layer=N_LAYERS, n_head=N_HEADS,
                          n_embd=VOCAB_DIM, dropout=0.0, bias=True))

    #model = SLiMPerformer(VOCAB_SIZE, VOCAB_DIM, HIDDEN_DIM,
    #                      N_LAYERS, FFN_DIM, N_HEADS, FEATURE_TYPE,
    #                      COMPUTE_TYPE)

    total_params = sum(p.numel() for p in model.parameters())
    model = model.to(device)

    print(f"The model has {total_params} parameters")
    # print(model)

    if torch.__version__ > "1.0.0":
        print("Compiling model")
        model = torch.compile(model)

    optim = torch.optim.Adam(
        model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    model.train()
    print("Number of iterations",iter_num)
    for train_index in trange(iter_num):
        train_batch = train_data[ind, :]
        y = train_batch[:, -1]
        train_batch = torch.from_numpy(train_batch).cuda().long()

        train_loss, logits = model.full_loss(train_batch, with_grad=True)
        wandb.log({"loss": train_loss})
        optim.step()
        optim.zero_grad(set_to_none=True)

        logits = logits.transpose(1, 2)
        prob = logits[:, -1, :]
        prob = F.softmax(prob, dim=1).detach().cpu().numpy()
        cumul_batch[:, 1:] = np.cumsum(prob*10000000 + 1, axis=1)

        for i in range(BATCH_SIZE):
            enc[i].write(cumul_batch[i, :], y[i])

        ind += 1
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

    global SEQ_LENGTH
    SEQ_LENGTH = SEQ_LENGTH*(HIDDEN_DIM // VOCAB_DIM)

    with open(file_path, "rb") as f:
        series = np.frombuffer(f.read(), dtype=np.uint8)

    training_data = strided_app(series, SEQ_LENGTH+1, 1)
    total_length = len(training_data)

    if ENCODE and total_length % BATCH_SIZE == 0:
        encode(temp_dir, compressed_file, series, training_data, None)
    elif ENCODE:
        # encode(temp_dir, compressed_file, series, training_data, None)
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
