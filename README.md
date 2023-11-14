## PSST! Prosodic Speech Segmentation With Transformers
[[Paper]](https://arxiv.org/abs/2302.01984)
[[Colab Tutorial]](https://colab.research.google.com/github/Nathan-Roll1/PSST/blob/main/Transcription_Example.ipynb)

New: [[*Quantized* Tutorial]](https://colab.research.google.com/github/Nathan-Roll1/PSST/blob/main/Tutorials/PSST_Q_Inference.ipynb)

Easy to use, prosodically-informed text-to-speech! 
- Integrated with intonation unit ~ intonational phrase ~ prosodic unit
- Boundaries are transcribed with the `!!!!!` token.
- Finetuned from Whisper (medium.en)

Generate transcriptions using PSST:
```python
!pip install transformers librosa
```
Next, import the necessary libraries and functions from the installed modules.
```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import librosa
```
Define a function `init_model_processor` to initialize the model and processor, which will be used to generate transcriptions from audio inputs.
```python
def init_model_processor(pretrained_name="NathanRoll/psst-medium-en", gpu=False):
    """Initializes the model and processor with the pre-trained weights.
    
    Returns:
      model, processor
    """
    processor = AutoProcessor.from_pretrained(pretrained_name)
    device = "cuda:0" if gpu else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(pretrained_name).to(device)
    
    return model, processor
```

The function `generate_transcription` utilizes the initialized model and processor to create a textual transcription of provided audio data.
```python
def generate_transcription(audio, model, processor, gpu=False):
    """Generate a transcription from audio using a pre-trained model."""
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    input_features = inputs.input_features.to("cuda:0") if gpu else inputs.input_features

    generated_ids = model.generate(input_features, max_length=250)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace('!!!!!', '|')
```

Next, use `librosa` to load and resample the audio file.
```python
y, sr = librosa.load('gettysburg.wav') # Your audio file here
audio = librosa.resample(y, orig_sr=sr, target_sr=16000)
```

Finally, initialize the model and processor, then generate and display the transcription of the resampled audio.
```python
# Initialize model and processor
model, processor = init_model_processor(gpu=False)

# Generate Transcription
transcript = generate_transcription(audio, model, processor, gpu=False)
print(transcript)
```

**Output:**
```
Four score and seven years ago <|IU_Boundary|> our fathers brought forth on this continent <|IU_Boundary|> a new nation <|IU_Boundary|> conceived in liberty <|IU_Boundary|> and dedicated to the proposition <|IU_Boundary|> that all men are created equal <|IU_Boundary|> Now we are engaged in a great civil war <|IU_Boundary|> testing whether that nation <|IU_Boundary|> or any nation so conceived and so dedicated <|IU_Boundary|> can long endure
```

You may cite this work as: 
```
@inproceedings{roll2023psst,
  title={PSST! Prosodic Speech Segmentation with Transformers},
  author={Roll, Nathan and Graham, Calbert and Todd, Simon},
  journal={Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL)},
  year={Forthcoming}
 }
```
