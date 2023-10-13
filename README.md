## PSST! Prosodic Speech Segmentation With Transformers
[[Paper]](https://arxiv.org/abs/2302.01984)
[[Tutorial]](https://colab.research.google.com/github/Nathan-Roll1/PSST/blob/main/Transcription_Example.ipynb)
[[Quantized]](https://colab.research.google.com/github/Nathan-Roll1/PSST/blob/main/Tutorials/PSST_Q_Inference.ipynb)

Easy to use, prosodically-informed text-to-speech! 
- Integrated with intonation unit ~ intonational phrase ~ prosodic unit
- Boundaries are transcribed with the '!!!!!' token.
- Finetuned from Whisper (medium.en)

PSST can be acessed through the transformers module:
```python
!pip install transformers
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
