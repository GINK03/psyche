"""
MeCabのインスタンスを返すライブラリ
"""

from sys import platform
import MeCab

if platform in ("linux", "linux2"):
    WAKACHI_PARSER = MeCab.Tagger(
        "-Owakati -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/"
    )
elif platform == "darwin":
    WAKACHI_PARSER = MeCab.Tagger(
        "-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")

if platform in ("linux", "linux2"):
    CHASEN_PARSER = MeCab.Tagger(
        "-Ochasen -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd/"
    )
elif platform == "darwin":
    CHASEN_PARSER = MeCab.Tagger(
        "-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
