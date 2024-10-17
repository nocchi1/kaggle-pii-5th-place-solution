from typing import List


def visualize_pii_text(tokens: list[str], spaces: list[int], labels: list[str]):
    text = ""
    for token, space, label in zip(tokens, spaces, labels):
        # piiを赤色で表示
        if label != "O":
            token = f"\033[31m{token}\033[0m"
        text += token
        if space:
            text += " "
    print(text)


def visualize_pii_pred_text(tokens: list[str], spaces: list[int], labels: list[str], preds: list[str]):
    text = ""
    for token, space, label, pred in zip(tokens, spaces, labels, preds):
        if label == pred and label != "O":
            # TPを赤色で表示
            token = f"\033[31m{token}\033[0m"
        elif label == "O" and pred != "O":
            # FPを黄色で表示
            token = f"\033[33m{token}\033[0m"
        elif label != pred and label != "O":
            # FNを緑色で表示(一部FPも含む)
            token = f"\033[32m{token}\033[0m"
        text += token
        if space:
            text += " "
    print(text)
