
import re
import pyopenjtalk

def openjtalk_cleaner(text):
    # This is the robust cleaner function using the placeholder strategy.

    # Define placeholders for special tokens
    replacements = {
        '<cough>': '__COUGH__',
        '{cough}': '__COUGH__',
        '[': '__LBRACKET__',
        ']': '__RBRACKET__',
    }

    # Replace special tokens with placeholders before phonemization
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Phonemize the whole text
    phonemes = pyopenjtalk.g2p(text)

    # Replace pyopenjtalk's pause symbol 'pau' with 'sp'
    phonemes = phonemes.replace('pau', 'sp')

    # Restore special tokens from placeholders, adding spaces
    phonemes = phonemes.replace('__COUGH__', ' <cough> ')
    phonemes = phonemes.replace('__LBRACKET__', ' [ ')
    phonemes = phonemes.replace('__RBRACKET__', ' ] ')

    # Final cleanup of spaces
    final_text = " ".join(phonemes.split())
    
    return final_text
