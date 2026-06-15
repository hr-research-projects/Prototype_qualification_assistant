import emoji

def remove_emojis(text: str) -> str:
    """
    Removes emojis and other non-standard Unicode characters from a string.
    """
    return emoji.replace_emoji(text, replace='')