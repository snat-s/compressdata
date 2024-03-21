# compressdata

Check references at docs/references.

I am using RWKV for compression. It is pretty good, the layers are stackable.

| N Layers                                     | kb/min         | # Params  | gpu allocation |
|----------------------------------------------|----------------|-----------|----------------|
| RWKV 1                                       | "1202,36"      | 0.06M     | 1.09G          |
| RWKV 2                                       | "714,29"       | 0.11M     | 1.42G          |
| RWKV 4                                       | "380,08"       | 0.22M     | 1.84G          |
| RWKV 8                                       | "202,10"       | 0.42M     | 3.43G          |
| "Tensorflow-compress (Mao, Y. et. al, 2022)" | "30,72~163,8 " | 258.3M    | 10.64G         |
| "NNCP (Mao, Y. et. al, 2022)"                | "28,6~122,9"   | 151M-224M | 7.75G          |
| "Dzip (Mao, Y. et. al, 2022)"                | "399,4"        | 1M        | 6.39G          |
| "TRACE (Mao, Y. et. al, 2022)"               | "952,3"        | 2.40M     | 2.02G          |
| "TRACE+BP controller (Mao, Y. et. al, 2022)" | "1228,8"       | 2.40M     | 2.02G          |

TRACE+BP is not open source, and they use a better graphics card, I have to test because I think
in my computer it would be a lot slower.

| Name of document         | Ratio | File size | Type        |
|--------------------------|-------|-----------|-------------|
| enwik8                   | 1     | 100000000 | Original    |
| enwik8_1_1024.compressed | 3,43  | 29120871  | RWKV        |
| enwik8_2_1024.compressed | 3,71  | 26958823  | RWKV        |
| enwik8_4_1024.compressed | 3,97  | 25215016  | RWKV        |
| enwik8_8_1024.compressed | 4,15  | 24114950  | RWKV        |
| enwik8.zip               | 2,74  | 36445475  | Traditional |
| gzip -9 enwik8.gz        | 2,74  | 36445248  | Traditional |
| 7z enwik                 | 3,83  | 26080305  | Traditional |

