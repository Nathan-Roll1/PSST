## PSST! Prosodic Speech Segmentation With Transformers
[[Colab example]](https://github.com/Nathan-Roll1/PSST/blob/main/Transcription_Example.ipynb)
[[Paper]](https://arxiv.org/abs/2302.01984)

PSST can be acessed through the transformers module:
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

Samples:




SBCSAE04_7:34_7:45-sample.wav

MODEL

um

and so I go in there and I'm like

well can you tell me whether the

the form is on file

Cause I I n

I realize it takes two to three weeks to process

but just tell me whether it's on file

Because if not

I want her to have another one

now



ACTUAL

um

and so I go in there and I'm like

well can you tell me whether the

the form is on file

cause I I n

I realize it takes two to three weeks to process

but just tell me whether it's on file

Because if not

I want her to have another one

now





SBCSAE03_14:42_14:52-sample.wav

MODEL

I have a neat book

I don't know if you've seen

What's Cebuano

It's in the Philippines too

It's on the island of Cebu

Oh

Of course



ACTUAL

I have a neat book

I don't know if you've seen

What's Cebuano

It's in the Philippines too

It's on the island of Cebu

Oh

Of course





SBCSAE03_8:13_8:14-sample.wav

MODEL

Um



ACTUAL

Um





SBCSAE04_18:15_18:20-sample.wav

MODEL

They should've learned that in the second grade

In the in the second grade I learned my times tables



ACTUAL

They should have learned that in the second grade

In the in the second grade I learned my times tables





SBCSAE02_11:01_11:07-sample.wav

MODEL

Cause there are a_lot_of women out there

who apparently don't believe in using condoms

Mm

I'm just amazed



ACTUAL

don't believe in using condoms

Hm

I'm just amazed





SBCSAE04_8:33_8:50-sample.wav

MODEL

I'm the only teacher who's not experienced

who's not certified

who just started teaching

All these other teachers are old hands

I mean they've all been at it for at

Well Chris is the least experienced besides me

but still he's

you know

he's had his certification

and he's had a year and stuff

he's real good at it



ACTUAL

I'm the only teacher who's not experienced

who's not certified

who just started teaching

All these other teachers are old hands

I mean they've all been at it for at

Well Chris is the least experienced besides me

but still he's

you know

he's had his certification

and he's had a year and stuff

he's real good at it





SBCSAE01_10:29_10:37-sample.wav

MODEL

Especially

you know

twelve bucks

for a trim

I mean that's twelve bucks

every time I can go out and trim my own horse's hooves



ACTUAL

Especially

uh_you know

twelve bucks

for a trim

I mean that's twelve bucks

every time I can go out

and trim my own horse's hooves





SBCSAE01_6:43_6:50-sample.wav

MODEL

you know

it's just

uh it's

it sounds easy

but it's really hard to do

and then

I would never do it

cause I never trust myself to do it



ACTUAL

you know

It's just

uh it s it sounds easy

but it's really hard to do

and then

I would never do it

cause I'd never trust myself to do it





SBCSAE03_0:28_0:31-sample.wav

MODEL

Well let's see



ACTUAL

Well let's see





SBCSAE03_0:22_0:24-sample.wav

MODEL

I know

It's kinda smelly

isn't it

Mm



ACTUAL

I know

It's kind of smelly

isn't it

Mhm





SBCSAE04_12:60_13:00-sample.wav

MODEL

Oh



ACTUAL

Oh





SBCSAE03_21:02_21:04-sample.wav

MODEL

Oops



ACTUAL

Oops





SBCSAE04_13:26_13:28-sample.wav

MODEL

and we all heard it

and we all



ACTUAL

And we all heard it

and we all





SBCSAE03_19:45_19:50-sample.wav

MODEL

Presplit

Yeah

those are left over

they're probably

Yeah

there might be mold



ACTUAL

Presplit

Yeah

those are left over

They're probably

Yeah

there might be mold





SBCSAE03_2:18_2:23-sample.wav

MODEL

cause Nancy's in the column

Where's the lettuce washer

You know

the salad spinner thing

X



ACTUAL

cause Pete's using the colander

Where's the lettuce washer

You know

the salad spinner thing

xxx





SBCSAE02_15:43_15:44-sample.wav

MODEL

and I'm just going



ACTUAL

Are they going





SBCSAE03_21:01_21:01-sample.wav

MODEL

beans



ACTUAL

beans





SBCSAE01_20:38_20:41-sample.wav

MODEL

I didn't even know she was pregnant

three months ago



ACTUAL

I didn't even know she was pregnant

three months ago





SBCSAE03_14:24_14:30-sample.wav

MODEL

Nepal

and

India too

when

November



ACTUAL

Nepal

And

India too

when

November





SBCSAE02_14:01_14:10-sample.wav

MODEL

cause I burned em for the past three months

I didn't think anything of it

But then

this guy played songs for a whole hour

And it was like

eighty percent of those songs out

that band

his son

that very night

Mhm



ACTUAL

cause I've heard em for the past three months

I didn't think anything of it

but then

this guy played songs for a whole hour

and it was like

eighty per cent of those songs I'd

that band had sung that very night

Mhm





SBCSAE04_8:15_8:16-sample.wav

MODEL

uh

uh



ACTUAL

uh

uh





SBCSAE04_9:28_9:30-sample.wav

MODEL

technicality that they have to meet

Yeah



ACTUAL

technicality that they have to meet

Yeah





SBCSAE04_19:09_19:14-sample.wav

MODEL

We just

We lucked out

man

But we got Missis Lindberg

who was like

the first granola woman I ever met



ACTUAL

We just

We lucked out

man

But we got Missis Lindberg

who was like

the first granola woman I ever met





SBCSAE03_7:26_7:28-sample.wav

MODEL

You know

they eat it

when they're up there



ACTUAL

You know

they eat it

when they're up there





SBCSAE04_15:56_15:58-sample.wav

MODEL

I have

five thirdgraders



ACTUAL

five thirdgraders





SBCSAE01_15:50_15:52-sample.wav

MODEL

Whatever they do wrong

we do get mad at em



ACTUAL

whatever they do wrong

we do get mad at em





SBCSAE01_18:48_18:49-sample.wav

MODEL

look

right here



ACTUAL

Look

Right here





SBCSAE03_14:01_14:02-sample.wav

MODEL

Right

They're out of there



ACTUAL

Right

They're out of there





SBCSAE03_7:33_7:42-sample.wav

MODEL

Isn't that great

It's nice for them

They have some recreation

But no salmon in your stockings this year hunh



ACTUAL

Isn't that great

It's nice for them

They have some recreation with it

But no salmon in your stockings this year

Hunh





SBCSAE04_15:22_15:29-sample.wav

MODEL

well

what you do with those thirdgraders

you know

is you just like

take them

and put them

you know

with one of the smarter fourthgraders



ACTUAL

well

what you do with those thirdgraders

you know

is you just like

take them

and put them

you know

with one of the smarter fourthgraders





SBCSAE04_7:57_7:58-sample.wav

MODEL

that

you know



ACTUAL

that

you know





SBCSAE04_12:55_12:58-sample.wav

MODEL

and you practice

Finally that man was fired

man



ACTUAL

And you practice

Finally that man was fired

man





SBCSAE04_1:35_1:39-sample.wav

MODEL

yo estoy vayando poner

uh las letras alla

E



ACTUAL

yo estoy vayando poner

uh las letras alla

E





SBCSAE01_23:22_23:28-sample.wav

MODEL

Mhm

One side of my heart was just hard

from the mus muscle

it was just hard



ACTUAL

Mhm

One side of her heart was just hard

from

the muscle

the muscle was just hard





SBCSAE03_17:56_17:57-sample.wav

MODEL

and showers



ACTUAL

and showers





SBCSAE02_15:43_15:44-sample.wav

MODEL

and I'm just going



ACTUAL

Are they going





SBCSAE02_0:38_0:40-sample.wav

MODEL

Probably

yeah

Talk about how Gregory Hiney said



ACTUAL

probably

yeah

talking about how Gregory Hines said





SBCSAE01_2:58_3:11-sample.wav

MODEL

well you can trim em too short

And make em

you know

and they're just a little bit

for f the first couple of days

You know

I mean they're just sore

That's not bad

but sometimes you can get it really bad

You can really make a horse really bad



ACTUAL

Well you can trim em too short

uh And make em

you know

and they're just a little bit

for f

the first couple of days

you know

I mean they're just sore

That's not bad

but sometimes you can get it really bad

You can really make a horse really bad





SBCSAE03_21:38_21:41-sample.wav

MODEL

So should I just finish it all off



ACTUAL

So should I just finish it all off





SBCSAE03_22:27_22:28-sample.wav

MODEL

and watch us pull in

and she goes like this



ACTUAL

and watch us pull in

and she goes like this





SBCSAE03_16:57_17:13-sample.wav

MODEL

That looks like enough salad

Oh I get it

Bill went out to put the

You put the water on the plants

Is that what was going on

What

You were watering with that

Is that what you were doing

Yeah

Yeah

That looks good



ACTUAL

That looks like enough salad

Oh I get it

Mar went out to put the

You put the water on the plants

Is that what was going on

What

You were watering with that

Is that what you were doing

Yeah

Yeah

that's good





SBCSAE03_8:55_9:05-sample.wav

MODEL

Yeah

I don't know

I mean

I I don't know if our drought here will ever break

I wonder if this is just isn't

This is it



ACTUAL

Yeah

I don't know

I mean

I I don't know if our drought here will ever break

I wonder if this is just isn't

This is it





SBCSAE01_19:49_20:00-sample.wav

MODEL

Next thing you knew it was just overcast

well the smoke all blew in

Just like there was a fire

Right around the clothes

And it was just dense

You couldn't even hardly see

Very far away

Okay then

it

the



ACTUAL

next thing you knew

it was just overcast

well the smoke all blew in

Just like there was a fire

right around close

And it was just dense

You couldn't even hardly see

very far away

Okay then

it

that





SBCSAE02_16:53_16:55-sample.wav

MODEL

Do they still teach at Bahia on Sunday

Oh yeah



ACTUAL

Do they still teach at Bahia on Sunday

night

Oh yeah





SBCSAE03_1:27_1:29-sample.wav

MODEL

Cause

well

we won't use it

if you don't cook it



ACTUAL

Cause

well

we won't use it

if you don't cook it





SBCSAE02_7:34_7:36-sample.wav

MODEL

oh

the Brazilian troop that was here



ACTUAL

oh the Brazilian troop that was here





SBCSAE03_3:31_3:38-sample.wav

MODEL

um

We're having like salad and fish

Unhunh

and gre green beans

Yeah

We can make um

garlic bread or something



ACTUAL

um

we're having like salad and fish

unhunh

and gre green beans

Yeah

We can make um

garlic bread or something





SBCSAE01_2:26_2:35-sample.wav

MODEL

and

you know

of the hoof

and

what everything was called there

and then

he went over on how to trim it

and where he

th there's a white line that you go by

and a horse has this little white line



ACTUAL

and

uh you know

of the hoof

and

what everything was called there

and then

he went over on how to trim it

and where you

th there's a white line

that you go by

and a horse has this little white line





SBCSAE01_0:48_0:54-sample.wav

MODEL

I don't know how to say it

the

you know

they do it for a living

you know

most people that you would get to trim your horse do it

all the time



ACTUAL

I don't know how to say it

But you know

they do it for a living

you know

all the time





SBCSAE02_11:47_11:51-sample.wav

MODEL

You know

and I'm thinking

I I can't believe that

Cause in this city



ACTUAL

you know

and I'm thinking

I I can't believe that

cause in this city
