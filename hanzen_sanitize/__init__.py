"""
半角カタカナを全角に
全角英数字を半角に変換
"""
import mojimoji

def hanzen_sanitize(x: str) -> str:
    x = mojimoji.zen_to_han(
        x, kana=False, digit=True, ascii=True)  # 全角英数字を半角に
    x = mojimoji.han_to_zen(
        x, kana=True, digit=False, ascii=False)  # 半角カナを全角に
    return x
