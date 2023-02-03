import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", transformers])

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def init_model_processor(gpu=False):
  processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")
  
  if gpu:
    model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en").to("cuda:0")
  else:
    model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en")

  return model, processor