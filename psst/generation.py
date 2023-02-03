def init_model_processor(AutoProcessor, AutoModelForSpeechSeq2Seq, gpu=False):
  processor = AutoProcessor.from_pretrained("NathanRoll/psst-medium-en")
  
  if gpu:
    model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en").to("cuda:0")
  else:
    model = AutoModelForSpeechSeq2Seq.from_pretrained("NathanRoll/psst-medium-en")

  return model, processor
