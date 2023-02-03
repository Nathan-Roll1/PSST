## PSST! Prosodic Speech Segmentation With Transformers

### Getting Started
Start by downloading the github module:
```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")
model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en")
```

Import ProsodPy
```python
import PSST
```

