from aux_functions import toBraille, encode_char
from aux_values import alphabet_to_dots, dots_to_alphabet, dots_to_unicode, unicode_to_dots
import numpy as np
import math
import re

# DONE: add new endpoint to api
# DONE: add load in rfc in api.py
# DONE: add any new dependencies
# DONE: add new files to backend repo i.e. the fivegram with prob and rfc classifier

def predict_text_rfc(
    text,
    rfc_model,
    lstm_model,
    fivegram_prob_model,
    all_chars,
    type_chars,
    encoded,
    lstm_input_seq_len=5,
    temperature=1.0
):
    """
    Make a prediction for the next character on a single sequence string, and choose the model based on inputting both the LSTM and fivegram outputs into a Random Forest Classifier to decide which model outputs to use.
    
    If no fivegram prediction is available to ensemble, then default to the LSTM output.

    Return an array of arrays, where each subarray is a pair of predicted character and probability, e.g.:

    * "Hello worl" -> [['d', 1.0], ['w', 0.0]]
    * "What is t" -> [['h', 0.784], ['o', 0.11], ['r', 0.035], ['a', 0.033], ['e', 0.013]]
    """

    text = clean_text(text)

    lstm_predictions = sample_with_temperature(predict_with_lstm(text, lstm_model, all_chars, type_chars, encoded, lstm_input_seq_len), temperature=temperature)
    fivegram_predictions = predict_with_fivegram(text, fivegram_prob_model=fivegram_prob_model)
    
    """
    The shape of the input to the RFC is a NumPy array of length 20 subarrays, where every 2 consecutive values represent:
    - the probability of the guess
    - 0 for if the guess was made by the LSTM model, 1 for if it was made by the fivegram model, and -1 if uncertain (default to LSTM)
    """

    if fivegram_predictions is None:
        return lstm_predictions

    else:
        X = []
        for char, prob in lstm_predictions:
            X.append(prob)
            X.append(0)
        for char, prob in fivegram_predictions:
            X.append(prob)
            X.append(1)
        X = np.array([X])

        rfc_pred = rfc_model.predict(X)

        if rfc_pred[0] < 1:
            return lstm_predictions
        else:
            return strip_fivegram_predictions_of_placeholders(fivegram_predictions)
        




def strip_fivegram_predictions_of_placeholders(fivegram_predictions):
    """
    Fivegram predictions may have ['*', 0.0] predictions in them to fill up to 5 guesses for the RFC model to correctly be able to give predictions.
    
    This function removes all pairs to get the output ready to send back to the front-end.
    """
    
    r = []
    for char, prob in fivegram_predictions:
        if char != '*':
            r.append([char, prob])
    return r






def clean_text(text):
    """
    This function is the same as can be found in aux_functions.py, but removes support for numeric characters as this is not supported in the fivegram model, so it won't be in the RFC.
    """

    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.lower()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text





def predict_with_fivegram(text, fivegram_prob_model, placeholder_prediction=['*', 0.0]):
    """
    Make a prediction for the next character on a single sequence string, using a fivegram model.

    Return an array of arrays, where each subarray is a pair of predicted character and probability.
    """

    if len(text) < 4:
        padding = "*" * (4 - len(text))
        X = padding + text
    else:
        X = text[-4:]

    if X in fivegram_prob_model:

        y = fivegram_prob_model[X]

        if len(y) < 5:
            for _ in range(5 - len(y)):
                y.append(placeholder_prediction)

        y = y[:5] # taking top 5 predictions
        result = []
        
        # some of the fivegram output subarrays have 'fivegram' appended to them many times e.g. see output for ' tha'
        # this code asserts that the subarrays of the result are pairs
        for pair in y:
            newpair = pair[:2]
            result.append(newpair)

        return result

    else:
        return None
    





def predict_with_lstm(
    text,
    model,
    all_chars,
    type_chars,
    encoded,
    input_seq_len
):
    """
    Make a prediction for the next character on a single sequence string, using a single LSTM.

    Return an array of arrays, where each subarray is a pair of predicted character and probability.
    """

    # Get the correct length of input sequence, this is important when training on LSTMs of different input sequence length to capture different contexts
    if len(text) > input_seq_len:
        text = text[-1*input_seq_len:]
    elif len(text) == input_seq_len:
        pass
    else:
        padding = " " * (input_seq_len - len(text))
        text = padding + text

    x = np.array([encode_char(str(dots_to_unicode[alphabet_to_dots[c]]), encoded) for c in text])
    x = x.reshape((1, len(x), 1))

    pred = model.predict(x, verbose=0)
    pred = pred.reshape(type_chars, )

    argsortI = np.argsort(pred)

    # Initialize an empty list to store character-probability pairs
    predictions_list = []

    # Loop through the top 5 predictions
    for i in range(5):
        ind = argsortI[-i-1]
        unicode_of_pred_braille = int(all_chars[ind])

        # Convert Unicode to corresponding character using provided mappings
        if unicode_of_pred_braille in unicode_to_dots and unicode_to_dots[unicode_of_pred_braille] in dots_to_alphabet:
            pred_char = dots_to_alphabet[unicode_to_dots[unicode_of_pred_braille]]
        else:
            pred_char = " "

        # Append character and probability to the list
        predictions_list.append([pred_char, pred[ind]])

    # Return the list of character-probability pairs
    return predictions_list






def sample_with_temperature(predicted, temperature=1.0):
    """
    Take a given list of prediction pairs (of character-probability subarrays), and add temperature to the sample.

    A temperature of:

    1: will not have an effect on the data
    >1: will make the output more diverse but may reduce accuracy
    <1: will make the output more focused, but can reduce diversity
    """

    sum_before = sum([a[1] for a in predicted])
    temp = []
    result = []
    sum_after = 0

    for char, prob in predicted:
        prob = math.log(prob) / temperature
        exp_prob = math.exp(prob)

        temp.append([char, exp_prob])
        sum_after += exp_prob

    for char, prob in temp:
        prob /= sum_after
        prob *= sum_before
        result.append([char, prob])

    return sorted(result, key=lambda x: x[1], reverse=True)