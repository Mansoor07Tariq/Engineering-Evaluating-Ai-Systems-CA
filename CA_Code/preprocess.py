import pandas as pd
import re
from Config import *

#Load and combine both datasets
def get_input_data() -> pd.DataFrame:
    """
Load AppGallery.csv and Purchasing.csv, rename label columns,
    and merge into a single DataFrame.
    """
    df1 = pd.read_csv("data//AppGallery.csv", skipinitialspace=True)
    df1.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df2 = pd.read_csv("data//Purchasing.csv", skipinitialspace=True)
    df2.rename(columns={'Type 1': 'y1', 'Type 2': 'y2', 'Type 3': 'y3', 'Type 4': 'y4'}, inplace=True)

    df = pd.concat([df1, df2], ignore_index=True)

    #Convert columns to string
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].astype(str)
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].astype(str)

    #Set target column
    df["y"] = df[Config.CLASS_COL]

    #Remove empty or NaN labels
    df = df[(df["y"] != '') & (~df["y"].isna())]

    return df


#Remove repeated or template messages
def de_duplication(data: pd.DataFrame):
    """
Clean interaction content by removing duplicate or template text blocks.
    """
    data["ic_deduplicated"] = ""

    templates = sum(list({
        "english": [r"Aspiegel.*?Customer Support.*?", r"Aspiegel.*?SE.*?Dublin"],
        "german": [r"Aspiegel.*?Kundenservice.*?", r"SE.*?Dublin, Irland"],
        "french": [r"équipe.*?Aspiegel.*?", r"SE.*?Dublin, en Irlande"],
        "spanish": [r"Aspiegel.*?Servicio al Cliente.*?", r"SE.*?Dublín, Irlanda"],
        "italian": [r"Aspiegel.*?team.*?", r"SE.*?Dublino, Irlanda"],
        "portguese": [r"Aspiegel.*?Support team.*?", r"SE.*?Dublin, Irlanda"],
    }.values()), [])
    cu_pattern = "|".join(f"({i})" for i in templates)

    split_pattern = "|".join([
        r"From\s?:\s?xxxxx@xxxx.com.*?Subject\s?:",
        r"On.*?wrote:",
        r"Re\s?:|RE\s?:",
        r"\*\*\*\*\*\(PERSON\) Support issue submit",
        r"\s?\*\*\*\*\*\(PHONE\)*$"
    ])

    for t_id in data["Ticket id"].unique():
        ticket_df = data[data["Ticket id"] == t_id]
        seen = set()
        cleaned = []

        for text in ticket_df[Config.INTERACTION_CONTENT]:
            parts = [re.sub(split_pattern, "", i.strip()) for i in re.split(split_pattern, text) if i]
            parts = [re.sub(cu_pattern, "", p).strip() for p in parts]

            filtered = []
            for p in parts:
                if p and p not in seen:
                    seen.add(p)
                    filtered.append(p + "\n")

            cleaned.append(" ".join(filtered))

        data.loc[data["Ticket id"] == t_id, "ic_deduplicated"] = cleaned

    data[Config.INTERACTION_CONTENT] = data['ic_deduplicated']
    data.drop(columns=['ic_deduplicated'], inplace=True)

    return data


#Remove extra noise like greetings, emails, dates
def noise_remover(df: pd.DataFrame):
    """
Clean summary and interaction content by removing noise and extra formatting.
    """
    summary_noise = r"(sv\s*:|wg\s*:|ynt\s*:|fw(d)?\s*:|r\s*:|re\s*:|\[|\]|null|nan|support.pt 自动回复:)"
    df[Config.TICKET_SUMMARY] = (
        df[Config.TICKET_SUMMARY].str.lower()
        .replace(summary_noise, " ", regex=True)
        .replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].str.lower()

    content_noise = [
        r"(from :|subject :|sent :)",
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"\d{2}(:|.)\d{2}", r"xxxxx@xxxx\.com", r"\*{5}\([a-z]+\)",
        r"dear (customer|user)?", r"hello", r"hi", r"thank you.*?",
        r"we hope.*?", r"in this matter", r"original message",
        r"sent from my huawei.*?", r"\d+", r"[^0-9a-zA-Z]+", r"(\s|^).(\s|$)"
    ]

    for noise in content_noise:
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(noise, " ", regex=True)

    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].replace(r'\s+', ' ', regex=True).str.strip()

    #Remove groups with very few records
    if Config.GROUPED in df.columns:
        valid = df[Config.GROUPED].value_counts()
        df = df[df[Config.GROUPED].isin(valid[valid > 10].index)]

    return df

#optional: translation utility
def translate_to_en(texts: list[str]):
    """
    Detect language and translate to English using M2M100 model.
    """
    import stanza
    from stanza.pipeline.core import DownloadMethod
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

    nlp = stanza.Pipeline(lang="multilingual", processors="langid", download_method=DownloadMethod.REUSE_RESOURCES)
    model_name = "facebook/m2m100_418M"
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name)

    translated = []
    for text in texts:
        if not text:
            translated.append(text)
            continue

        lang = nlp(text).lang
        lang = {"fro": "fr", "la": "it", "nn": "no", "kmr": "tr"}.get(lang, lang)

        if lang == "en":
            translated.append(text)
        else:
            tokenizer.src_lang = lang
            tokens = tokenizer(text, return_tensors="pt")
            output = model.generate(**tokens, forced_bos_token_id=tokenizer.get_lang_id("en"))
            translated.append(tokenizer.decode(output[0], skip_special_tokens=True))

    return translated
