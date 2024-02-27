import requests

api_url = "http://0.0.0.0:8080"


def test_translate_endpoint():
    """
    Test the /translate endpoint.
    """

    translate_payload = {
        "text": "Hello, worl",
        "tableList": ["en-us-g1.ctb"]
    }
    response = requests.post(f"{api_url}/translate", json=translate_payload)
    if response.status_code == 200:
        print(response.json())
        braille = response.json()["braille"]
        print("Braille:", braille)
        return braille
    else:
        print("Translate request failed with status code:", response.status_code)
        quit(0)


def test_backtranslate_endpoint(braille):
    """
    Test the /backtranslate endpoint.
    """

    backtranslate_payload = {
        "braille": braille,
        "tableList": ["en-us-g1.ctb"]
    }
    response = requests.post(f"{api_url}/backtranslate", json=backtranslate_payload)
    if response.status_code == 200:
        text = response.json()["text"]
        print("Back translated text:", text)
        return text
    else:
        print("Backtranslate request failed with status code:", response.status_code)
        quit(0)



def test_fivegram_endpoint(text):
    """
    Test the /fivegram endpoint.
    """

    fivegram_payload = {
        'text': text
    }
    response = requests.post(f"{api_url}/fivegram", json=fivegram_payload)
    if response.status_code == 200:
        pred = response.json()["pred"]
        print("Fivegram prediction:", pred)
        return pred
    else:
        print("Fivegram request failed with status code:", response.status_code)
        quit(0)


def test_lstm_endpoint(text):
    """
    Test the /lstm endpoint.
    """

    lstm_payload = {
        'text': text
    }
    response = requests.post(f"{api_url}/lstm", json=lstm_payload)
    if response.status_code == 200:
        pred = response.json()["pred"]
        print("Fivegram prediction:", pred)
        return pred
    else:
        print("LSTM request failed with status code:", response.status_code)
        quit(0)



def test_rfc_endpoint(text):
    """
    Test the /randomforest endpoint.
    """

    rfc_payload = {
        'text': text
    }
    response = requests.post(f"{api_url}/randomforest", json=rfc_payload)
    if response.status_code == 200:
        pred = response.json()["pred"]
        print("RFC prediction:", pred)
        return pred
    else:
        print("RFC request failed with status code:", response.status_code)
        quit(0)





if __name__ == "__main__":
    braille = test_translate_endpoint()
    text = test_backtranslate_endpoint(braille)

    text = "Hello worl"

    test_fivegram_endpoint(text)
    test_lstm_endpoint(text)
    test_rfc_endpoint(text)
