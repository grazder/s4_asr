# S4 ASR

WandB Project - [grazder/s4_asr](https://wandb.ai/grazder/s4_asr)

## 0. Installation

Download NeMo dependencies:
```bash
cd Docker
docker build -f Dockerfile -t pytorch22.11 .
bash start.sh
```

Download LibriSpeech
```bash
python get_librispeech_data.py \
  --data_root /work/s4_asr/LibriSpeech_NeMo \
  --data_sets ALL

cat train*.json > train.json
```

## 1. Conformer Baseline

`Config`: [conformer_ctc_char.yaml](configs/conformer_ctc_char.yaml)

`WandB Run`: [Conformer-CTC-Char](https://wandb.ai/grazder/s4_asr/runs/2023-02-27_23-32-00?workspace=user-grazder)

`GPU`: 2x RTX 2080 Ti 11 GB

Start run

```bash
python speech_to_text_ctc.py --config-name=conformer_ctc_char  --config-path /work/s4_asr/configs/
```

`Result`: 6.5 WER per 50 epoch

## 2. S4 instead of MHA Conformer

Installation

```
pip install opt_einsum
pip install pykeops cmake
cd state-spaces/extensions/cauchy/
python setup.py install
cd ../../../
```

`Config`: [s4_instead_attention_ctc_char.yaml](configs/s4_instead_attention_ctc_char.yaml)

`WandB Run`: [S4-Instead-of-Attention-CTC-Char](https://wandb.ai/grazder/s4_asr/runs/2023-03-02_23-01-43)

`GPU`: 2x RTX 2080 Ti 11 GB

```bash
python speech_to_text_ctc.py --config-name=s4_instead_attention_ctc_char  --config-path /work/s4_asr/configs/
```

`Result`: 13.5 WER per 38 epoch

## 3. H3 instead of MHA Conformer

```bash
python speech_to_text_ctc.py --config-name=h3_instead_attention_ctc_char  --config-path /work/s4_asr/configs/
```



