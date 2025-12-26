from .phonemize import Phonemizer

phonemizer = Phonemizer()

def japanese_cleaners(text):
    return phonemizer(text)