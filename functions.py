def display_results(n, save=False):
  '''
  This function displays the audio file and the transcript of the generated model as well as the actual transcript of a given index "n" in a dataframe.
  :param n: int - the index of the dataframe to display the results for
  '''

  # get the nth row of the dataframe
  r = df.iloc[n]

  # use the IPython library to display the audio file
  IPython.display.display(IPython.display.Audio(r['audio'], rate=16000))

  # optionally save audio chunk in question
  if save:
    fpath = f'/content/drive/Shareddrives/Ling Thesis/audio samples/{get_meta(n)}-sample.wav'
    write(fpath, 16000, (r['audio']*32767).astype(np.int16))
    print(f'Written to {fpath}')


  # print the model generated transcript with tab spacing and newlines to separate phrases
  print('MODEL\t', df.iloc[n]['model_generated'].replace(' <|IU_Boundary|> ','\n\t'), end='\n\n')

  # print the actual transcript with tab spacing and newlines to separate phrases
  print('ACTUAL\t', df.iloc[n]['actual_transcript'].replace(' <|IU_Boundary|> ','\n\t'))




def calculate_results(n: int, df):
    '''
    This function takes an integer n and returns a tuple of 4 integers.
    :param n: the index of the audio sample in the dataframe
    :param df: dataframe with columns `actual_transcript`, `model_generated`, and `audio`
    '''
    d = {} # dictionary to store timestamps
    r = df.iloc[n] # select the nth row from the dataframe
    write('/content/test.wav', 16000, r['audio']) # write the audio to a file

    for s in ['actual_transcript','model_generated']:
        # Align the transcript with the audio and extract timestamps
        aligned_output = charsiu.align(audio='/content/test.wav',text=r[s].replace('  ',' '))[1]
        aligned_words = np.array([(x[2].replace('_',' '), x[0]) for x in aligned_output if x[2] != '[SIL]'])
        actual_words = np.array(r[s].split(' '))
        for_a = [x for x in actual_words] # copy to sep memory obj

        buffered_timestamps = []

        # loop through the words in the transcript
        for i in range(len(actual_words)):
            # check if the first word in the transcript matches the first word in the aligned transcript
            if actual_words[0].lower() == aligned_words[0][0].lower():
                aligned_words = aligned_words[1:]
                actual_words = actual_words[1:]
            else:
                if actual_words[0] == '<|IU_Boundary|>':
                    actual_words = actual_words[1:]
                    buffered_timestamps.append(np.float16(aligned_words[0][1]))
                else:
                    actual_words = actual_words[1:]
        d[s] = np.array(buffered_timestamps)

    # Snap nearby bounds (within 0.02 seconds)
    for trscrpt_bound in d['actual_transcript']:
        bool_arr = (d['model_generated']<(trscrpt_bound+0.03)) & (d['model_generated']>(trscrpt_bound-0.03))
        d['model_generated'][bool_arr] = trscrpt_bound

    intersect = d['actual_transcript'][np.in1d(d['actual_transcript'], d['model_generated'])]

    fn = [x for x in d['actual_transcript'] if x not in intersect]
    fp = [x for x in d['model_generated'] if x not in intersect]

    tn = len(for_a) - len(intersect) - len(fn) - len(fp)

    return len(intersect), tn, len(fp), len(fn)



def yield_metrics(r: dict):
  '''
  This function returns a tuple of the accuracy, precision, recall, and f1 score.
  :param r: a dictionary containing the counts of true positives (tp), 
  true negatives (tn), false positives (fp), and false negatives (fn)
  '''

  #Calculating accuracy by dividing the sum of true positives and true negatives by the total number of observations
  accuracy = np.divide((r['tp']+r['tn']),np.sum(list(r.values())))

  #Calculating precision by dividing the number of true positives by the sum of true positives and false positives
  precision = np.divide(r['tp'],(r['tp']+r['fp']))

  #Calculating recall by dividing the number of true positives by the sum of true positives and false negatives
  recall = np.divide(r['tp'],(r['tp']+r['fn']))

  #Calculating f1 score by taking the harmonic mean of precision and recall
  f1 = np.divide(2*(precision*recall),(precision + recall))

  #Printing out the values for accuracy, precision, recall, and f1 score
  print('')
  for metric in ['accuracy','precision','recall','f1']:
    print(f'{metric}: {round(eval(metric),3)}')

  #Returning a tuple of the accuracy, precision, recall, and f1 score
  return accuracy, precision, recall, f1



def generate(n: int, ds_test, filter=None, m='default'):
    """
    Generates a transcription for a given audio file, with an optional filter applied.
    If the model is not set to 'whisper', it will use the default `model` to generate the transcription.
    :param n: index of the audio file in the test dataset
    :param filter: tuple of filter parameters (cutoff, type)
    :param m: model to use for transcription (default: 'default')
    """

    # Retrieve audio data from test dataset
    audio = ds_test[n]["audio"]["array"]
    
    # Apply filter to audio data, if one is provided
    if filter:
        b, a = signal.butter(4, filter[0] / (16000 / 2.), filter[1])
        audio = signal.filtfilt(b, a, audio)
    
    # Generate transcription using processor or whisper model
    if m != 'whisper':
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
        input_features = inputs.input_features.cuda()
        generated_ids = model.generate(inputs=input_features, max_length=250)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True, output_word_offsets=True)[0]
    else:
        r = whisper_model.transcribe(torch.from_numpy(audio).float(), language='en', max_initial_timestamp=None)
        transcription = ' !!!!!'.join([x['text'].lower().replace('.','').replace(',','').replace('?','').replace('!','').replace('  ',' ') for x in r['segments']])
    
    # Remove leading space from transcription, if present
    try:
      if transcription[0] == ' ':
          transcription = transcription[1:] 
    except:
      pass
    
    # Replace '!!!!!' token
    transcription = transcription.replace('!!!!!','<|IU_Boundary|>')
    actual = ds_test[n]['sentence'].replace('!!!!!','<|IU_Boundary|>')
    
    # Store generated transcription, actual transcription, and audio data in dictionary
    globals()['outs'][ds_test[n]['audio']['path']] = {'model_generated':transcription, 
                                        'actual_transcript':actual,
                                        'audio':audio}


def gen_results(ds, filter=None):
  '''
  Generate results for the given test dataset, with an optional filter.

  :param ds: The test dataset
  :param filter: Optional filter to apply to the dataset
  :return: Tuple containing the accuracy, precision, recall, and F1 score
  '''
  # Globalize outs dict for generate function
  globals()['outs'] = {}

  # Use tqdm to display progress bar while iterating over the test dataset
  for i in tqdm(range(len(ds)), position=0, leave=True):
    generate(i, ds, filter=filter)

  df = pd.DataFrame(globals()['outs']).T

  # Initialize dictionary to store results
  r = {
      'tp':0,
      'tn':0,
      'fp':0,
      'fn':0
  }

  # Iterate over the dataframe and calculate results
  for i in trange(len(df)):
    try:
      tp, tn, fp, fn = calculate_results(i, df)
      for metric in r.keys():
        r[metric] += eval(metric)
    except:
      pass

  # Calculate and return accuracy, precision, recall, and F1 score
  accuracy, precision, recall, f1 = yield_metrics(r)
  return accuracy, precision, recall, f1