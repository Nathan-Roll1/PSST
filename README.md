## PSST! Prosodic Speech Segmentation With Transformers
[[Colab example]](https://github.com/Nathan-Roll1/PSST/blob/main/Transcription_Example.ipynb)
[[Paper]](https://arxiv.org/abs/2302.01984)

PSST can be acessed throigh the transformers module:
```cli
pip install transformers
```

Load the pretrained checkpoint:
```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")
model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en")
```

Load sample audio file:

```python
import librosa
y, sr = librosa.load('gettysburg.wav')
audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
```
Define transcript generation function:
```python
def generate_transcription(audio, gpu=False):
  """Generate a transcription from audio using a pre-trained model

  Args:
    audio: The audio to be transcribed
    gpu: Whether to use GPU or not. Defaults to False.

  Returns:
    transcription: The transcribed text
  """
  # Preprocess audio and return tensors
  inputs = processor(audio, return_tensors="pt", sampling_rate=16000)

  # Assign inputs to GPU or CPU based on argument
  if gpu:
    input_features = inputs.input_features.cuda()
  else:
    input_features = inputs.input_features

  # Generate transcribed ids
  generated_ids = model.generate(inputs=input_features, max_length=250)

  # Decode generated ids and replace special tokens
  transcription = processor.batch_decode(
      generated_ids, skip_special_tokens=True, output_word_offsets=True)[0].replace('!!!!!', '<|IU_Boundary|>')
  
  return transcription
```

Generate transcription:
```python
generate_transcription(audio, gpu=True)
```
