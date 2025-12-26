"""
Defines the set of symbols used in text input to the model.
"""
_pad        = '_'
_punctuation = ';:,.!?¡¿—…«»“” []'

# The tables below are copied from text_JP/phonemize.py
# to dynamically generate the symbol set without modifying the original file.
table_jpn = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "k a", "キ": "k i", "ク": "k u", "ケ": "k e", "コ": "k o",
    "ガ": "g a", "ギ": "g i", "グ": "g u", "ゲ": "g e", "ゴ": "g o",
    "サ": "s a", "シ": "s i", "ス": "s u", "セ": "s e", "ソ": "s o",
    "ザ": "z a", "ジ": "z i", "ズ": "z u", "ゼ": "z e", "ゾ": "z o",
    "タ": "t a", "チ": "t i", "ツ": "t u", "テ": "t e", "ト": "t o",
    "ダ": "d a", "ヂ": "z i", "ヅ": "z u", "デ": "d e", "ド": "d o",
    "ナ": "n a", "ニ": "n i", "ヌ": "n u", "ネ": "n e", "ノ": "n o",
    "ハ": "h a", "ヒ": "h i", "フ": "h u", "ヘ": "h e", "ホ": "h o",
    "パ": "p a", "ピ": "p i", "プ": "p u", "ペ": "p e", "ポ": "p o",
    "バ": "b a", "ビ": "b i", "ブ": "b u", "ベ": "b e", "ボ": "b o",
    "マ": "m a", "ミ": "m i", "ム": "m u", "メ": "m e", "モ": "m o",
    "ヤ": "y a", "ユ": "y u", "ヨ": "y o",
    "ラ": "r a", "リ": "r i", "ル": "r u", "レ": "r e", "ロ": "r o",
    "ワ": "w a", "ン": "N", "ッ": "Q", "＃": "#", "ヲ": "o"
}

table2_jpn = {
    "キャ": "ky a", "キュ": "ky u", "キョ": "ky o",
    "ギャ": "gy a", "ギュ": "gy u", "ギョ": "gy o",
    "シャ": "sy a", "シュ": "sy u", "シェ": "sy e", "ショ": "sy o",
    "ジャ": "zy a", "ジュ": "zy u", "ジェ": "zy e", "ジョ": "zy o",
    "チャ": "ch a", "チュ": "ch u", "チェ": "ch e", "チョ": "ch o",
    "ニャ": "ny a", "ニュ": "ny u", "ニョ": "ny o",
    "ヒャ": "hy a", "ヒュ": "hy u", "ヒョ": "hy o",
    "ピャ": "py a", "ピュ": "py u", "ピョ": "py o",
    "ビャ": "by a", "ビュ": "by u", "ビョ": "by o",
    "ミャ": "my a", "ミュ": "my u", "ミョ": "my o",
    "リャ": "ry a", "リュ": "ry u", "リョ": "ry o",
    "ティ": "t i", "ディ": "d i",
    "トゥ": "t u", "ドゥ": "d u",
    "ツァ": "ts a", "ツェ": "ts e", "ツォ": "ts o",
    "スィ": "s i", "ズィ": "z i",
    "ファ": "f a", "フィ": "f i", "フェ": "f e", "フォ": "f o",
    "ウィ": "w i", "ウェ": "w e"
}

_phonemes = set()
for v in table_jpn.values():
    _phonemes.update(v.split(' '))
for v in table2_jpn.values():
    _phonemes.update(v.split(' '))

# Add special phonemes that are not in the tables but used in the cleaner
_phonemes.add('sp')
#_phonemes.add(':')

_phonemes.update(['a:', 'i:', 'u:', 'e:', 'o:'])

# Export all symbols:
symbols = [_pad] + list(_punctuation) + sorted(list(_phonemes))

# Special symbol ids
SPACE_ID = symbols.index(" ")