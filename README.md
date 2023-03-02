```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython
pip install nemo_toolkit['all']
```


```bash
python get_librispeech_data.py \
  --data_root /work/s4_asr/LibriSpeech_NeMo \
  --data_sets ALL

cat train*.json > train.json
```
 

```bash
python speech_to_text_ctc.py --config-name=conformer_ctc_char  --config-path /work/s4_asr/configs/
```