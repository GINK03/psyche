""" GCPのNLPサービスを利用してポジネガ判定を行う """
from typing import List, Dict
from google.cloud import language_v1


def sentiment_analytics(text_content: str) -> List[Dict[str, object]]:
    """
    テキストをgoogle language APIを利用して判定する
    `GOOGLE_APPLICATION_CREDENTIALS`を参照してGCPのサービスを認証する
    Args:
        - text_content:
            - ポジネガ判定を行いたいテキスト
    Returns:
        - List[Dict[str, object]]:
            - 判定を行った結果(短文ごとに別れて出力される)
    """
    client = language_v1.LanguageServiceClient()
    type_ = language_v1.Document.Type.PLAIN_TEXT

    language = "ja"
    document = {"content": text_content, "type_": type_, "language": language}

    encoding_type = language_v1.EncodingType.UTF8
    response = client.analyze_sentiment(request={
        'document': document,
        'encoding_type': encoding_type
    })

    rets = []
    for sentence in response.sentences:
        ret = {
            "content": sentence.text.content,
            "score": sentence.sentiment.score
        }
        rets.append(ret)
    return rets


if __name__ == "__main__":
    """ テスト """
    sentiment_analytics("""可能性という言葉を無限定に使ってはいけない。""")
