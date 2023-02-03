import os

import pkg_resources

for r in pkg_resources.parse_requirements(
    open(os.path.join(os.path.dirname(__file__), "PSST/requirements.txt"))
)

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def init_model_processor(gpu=False):
  processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")
  
  if gpu:
    model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en").to("cuda:0")
  else:
    model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en")

  return model, processor