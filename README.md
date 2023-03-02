# S4 ASR

WandB Project - [grazder/s4_asr](https://wandb.ai/grazder/s4_asr)

## 0. Installation

Download NeMo dependencies:
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']==1.15.0
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

## 2. S4 Encoder from Sashimi

