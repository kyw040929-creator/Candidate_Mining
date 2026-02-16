#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 13:23:30 2026

@author: alicia
"""
# =============================================================================
# SECTION 1: DATA LOADING & CLEANING
# Description: Load raw CSVs, normalize columns and dtypes, handle missing data,
#              merge TikTok + Instagram, create weekly buckets, and run minimal text cleaning.
# =============================================================================

# IMPORTS
import pandas as pd
import re
from collections import Counter, defaultdict
import numpy as np
from transformers import pipeline
import unicodedata

# --- Read TikTok CSV and drop unused columns
tiktok = pd.read_csv("SaltBread_Tiktok.csv", encoding="utf-8", encoding_errors="replace")

cols_to_drop = [
    "CREATETIMEISO",
    "DATE_SCRAPED",
    "CID",
    "UID",
    "UNIQUEID"
]

tiktok = tiktok.drop(columns=cols_to_drop)
print(tiktok.columns)

# --- Read Instagram CSV and inspect replies column, then drop unused columns
instagram = pd.read_csv("SaltBread_Instagram.csv", encoding="utf-8", encoding_errors="replace")

instagram["REPLIES"].value_counts().head()
(instagram["REPLIES"] != "[]" ).sum()
instagram.loc[instagram["REPLIES"] != "[]", "REPLIES"].head()

cols_to_drop = [
    "ERROR",
    "ERRORDESCRIPTION",
    "REQUESTERRORMESSAGES",
    "OWNERPROFILEPICURL",
    "URL",
    "MEDIA",
    "OWNER",
    "ID",
    "REPLIES"
]

instagram = instagram.drop(columns=cols_to_drop)
print(instagram.columns)


# --- Rename columns for a consistent schema and add platform flag
tiktok = tiktok.rename(columns={
    "TEXT": "comment_text",
    "CREATETIME": "comment_time",
    "DIGGCOUNT": "likes",
    "REPLYCOMMENTTOTAL": "reply_count",
    "REPLIESTOID": "parent_comment_id",
    "LIKEDBYAUTHOR": "liked_by_author",
    "PINNEDBYAUTHOR": "pinned_by_author",
    "MENTIONS": "mentions",
    "INPUT": "post_url"
})
tiktok["platform"] = "tiktok"
print(tiktok.columns)

instagram = instagram.rename(columns={
    "TEXT": "comment_text",
    "TIMESTAMP": "comment_time",
    "LIKESCOUNT": "likes",
    "REPLIESCOUNT": "reply_count",
    "POSTURL": "post_url"
})
instagram["platform"] = "instagram"
print(instagram.columns)


# --- Normalize times to datetimes and clean numeric dtypes
tiktok["comment_time"] = pd.to_datetime(
    tiktok["comment_time"],
    unit="s",
    utc=True,
    errors="coerce"
)

instagram["comment_time"] = pd.to_datetime(
    instagram["comment_time"],
    utc=True,
    errors="coerce"
)

print(tiktok["comment_time"].dtype)
print(instagram["comment_time"].dtype)

tiktok["likes"] = pd.to_numeric(tiktok["likes"], errors="coerce")
tiktok["reply_count"] = pd.to_numeric(tiktok["reply_count"], errors="coerce")

instagram["likes"] = pd.to_numeric(instagram["likes"], errors="coerce")
instagram["reply_count"] = pd.to_numeric(instagram["reply_count"], errors="coerce")

print(tiktok.dtypes)
print(instagram.dtypes)


# --- Handle missing core fields, fill engagement counts, and concat platforms
tiktok = tiktok.dropna(subset=["comment_text", "comment_time"])
instagram = instagram.dropna(subset=["comment_text", "comment_time"])

tiktok["likes"] = tiktok["likes"].fillna(0)
tiktok["reply_count"] = tiktok["reply_count"].fillna(0)

instagram["likes"] = instagram["likes"].fillna(0)
instagram["reply_count"] = instagram["reply_count"].fillna(0)

print(tiktok[["likes", "reply_count"]].isna().sum())
print(instagram[["likes", "reply_count"]].isna().sum())

all_comments = pd.concat([tiktok, instagram], ignore_index=True)

print(all_comments.shape)
print(all_comments["platform"].value_counts())
print(all_comments.dtypes)
print(all_comments.head())
print(all_comments.columns)


# --- Create weekly buckets and limit to last 10 weeks per platform
all_comments = all_comments.dropna(subset=["comment_time"])

all_comments["week_start"] = (
    all_comments["comment_time"]
    .dt.to_period("W-MON")
    .dt.start_time
)

all_comments["week_start"].isna().sum()

print(all_comments[["comment_time", "week_start"]].head())
print(all_comments["week_start"].value_counts().sort_index().head())

all_comments["week_start"].nunique()

latest_week = all_comments["week_start"].max()

last10_weeks_by_platform = (
    all_comments[["platform", "week_start"]]
      .dropna()
      .drop_duplicates()
      .sort_values(["platform", "week_start"])
      .groupby("platform", group_keys=False)
      .tail(10)
)

last_10_weeks = all_comments.merge(
    last10_weeks_by_platform,
    on=["platform", "week_start"],
    how="inner"
).copy()

print(last10_weeks_by_platform)


# --- Minimal text cleaning helpers and apply to comments
def basic_clean(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)         # remove urls
    s = re.sub(r"@\w+", " ", s)                    # remove @mentions
    s = re.sub(r"\s+", " ", s).strip()             # collapse whitespace
    return s

def remove_emojis_and_punct(s):
    s = re.sub(r"[^\w\s]", " ", s)    # remove punctuation and emojis
    s = re.sub(r"_+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

last_10_weeks["text_step1"] = last_10_weeks["comment_text"].apply(basic_clean)
last_10_weeks["text_step2"] = last_10_weeks["text_step1"].apply(remove_emojis_and_punct)

# drop empty comments after cleaning (if any)
last_10_weeks = last_10_weeks.loc[last_10_weeks["text_step2"].str.strip().astype(bool)].copy()
print("Rows after cleaning:", last_10_weeks.shape[0])


# =============================================================================
# SECTION 2: TOP-10 FLAVOR EXTRACTION
# Description: Mine candidate flavor phrases, export for labeling, load labels,
#              and extract labeled flavors per comment, then compute weekly top-10.
# =============================================================================

# --- Lightweight normalization utilities for flavor extraction
def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"@\w+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_phrase(p: str) -> str:
    p = str(p).lower().strip()
    p = re.sub(r"(.)\1{2,}", r"\1", p)
    p = re.sub(r"[_\-]+", " ", p)
    glue = {
        "creamcheese": "cream cheese",
        "garlicbutter": "garlic butter",
        "saltbread": "salt bread",
        "buttergarlic": "butter garlic",
        "matchabread": "matcha bread",
    }
    p = glue.get(p, p)
    p = re.sub(r"\s+", " ", p).strip()
    return p

# --- Tokenization and token filters for flavors
def tokenize(s: str):
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.findall(r"\b\w+\b", s.lower())

STOP = {
    "i","me","my","you","your","we","our","it","its","they","them","this","that","these","those",
    "a","an","the","and","or","but","to","of","in","on","for","with","as","at","by","from",
    "is","are","was","were","be","been","am","im","s","t","re","ve","ll","m",
    "so","very","just","can","cant","do","did","does","what","when","where","why","how",
    "thank","thanks","pls","please",
    "good","great","best","nice","amazing","delicious","yummy","tasty","love","like","want","need",
    "try","tried","eat","eating","ate","buy","bought","order","ordered","go","went","going",
    "wow","omg","lol","lmao","haha","hahaha",
    "kak","ini","ya","yg","aku","mau","nya","di","ga","gak","bgt","banget","ada",
    "dong","nih","sih","kok","aja","lagi","udah","enak","mantap","pengen","beli",
}

TOPIC = {"salt", "bread", "saltbread", "roti", "소금빵", "빵", "pan"}

BAN_TOKENS = {
    "price","pricing","expensive","cheap","worth","waste","overrated",
    "harga","mahal","murah",
    "open","close","hour","hours","jam","buka","tutup",
    "address","where","location","nyc","newyork","tokyo","seoul","korea","japan",
    "map","maps","google","line","queue","antri","wait","waiting",
    "sold","out","soldout","restock","stock",
    "recipe","resep","ingredients","ingredient","bake","baking","oven",
    "order","delivery","shipping",
    "tiktok","ig","instagram","reels","comment","comments",
}

def clean_tokens(txt: str):
    toks = tokenize(txt)
    out = []
    for t in toks:
        if t in STOP or t in TOPIC or t in BAN_TOKENS:
            continue
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        if re.fullmatch(r"[a-z]*\d+[a-z]*", t):
            continue
        out.append(t)
    return out

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# --- Anchors and gates to keep flavor candidates focused
ANCHORS = {
    "flavor","taste","filling","cream","cheese","butter","chocolate","matcha","garlic","corn",
    "truffle","pistachio","vanilla","strawberry","blueberry","honey","caramel","cocoa","mocha",
    "custard","jam","syrup","salted","brown","sweet","spicy",
    "keju","coklat","mentega",
    "버터","치즈","초코","말차","갈릭",
    "バター","チーズ","チョコ","抹茶","ガーリック",
}

CONTEXT_HINTS = {"with","filled","filling","topping","inside","in","맛","맛있","맛있는","맛있어","isi","rasa"}

GENERIC_NONFLAVOR_UNI = {
    "good","great","best","nice","amazing","delicious","yummy","tasty",
    "love","like","want","need","try","tried","trying",
    "fresh","crispy","soft","hard","sweet","salty","spicy","creamy","buttery",
    "wow","omg","lol","lmao","haha","hahaha",
    "please","pls","thanks","thank",
}

def passes_anchor_gate(
    phrase,
    tokens_in_comment,
    phrase_tokens,
    corpus_token_freq=None,
    corpus_anchor_cooccur=None,
    min_count_uni=3,
    cooccur_min_count=2,
    cooccur_min_prop=0.10
):
    ptoks = set(phrase_tokens)

    # Multiword phrases: require anchor presence or proximity to anchor
    if len(phrase_tokens) >= 2:
        if ptoks & ANCHORS:
            return True

        n = len(phrase_tokens)
        for i in range(len(tokens_in_comment) - n + 1):
            if tokens_in_comment[i:i+n] == phrase_tokens:
                left = tokens_in_comment[max(0, i-3):i]
                right = tokens_in_comment[i+n:i+n+3]
                if (set(left) | set(right)) & ANCHORS:
                    return True
        return False

    # Unigrams: use anchor co-occurrence heuristics and length checks
    token = phrase_tokens[0]

    if token in ANCHORS:
        return True

    if token in GENERIC_NONFLAVOR_UNI:
        return False

    for i in range(len(tokens_in_comment)):
        if tokens_in_comment[i] == token:
            left = tokens_in_comment[max(0, i-4):i]
            right = tokens_in_comment[i+1:i+5]
            if (set(left) | set(right)) & (ANCHORS | CONTEXT_HINTS):
                return True

    if corpus_token_freq is not None and corpus_anchor_cooccur is not None:
        tf = corpus_token_freq.get(token, 0)
        ac = corpus_anchor_cooccur.get(token, 0)
        if tf >= min_count_uni:
            if ac >= cooccur_min_count:
                return True
            if tf > 0 and (ac / tf) >= cooccur_min_prop:
                return True

    if re.fullmatch(r"[a-z]+", token) and 3 <= len(token) <= 15 and token not in GENERIC_NONFLAVOR_UNI:
        return True

    return False

# --- Phrase-level filtering for flavor candidates
PHRASE_BAN = {
    "so good", "very good", "best one", "want try", "need try",
    "in nyc", "in tokyo", "in seoul",
}

def looks_like_bad_phrase(p: str) -> bool:
    p = normalize_phrase(p)
    toks = p.split()

    if p in PHRASE_BAN:
        return True

    if len(toks) == 1:
        t = toks[0]
        if t in STOP or t in TOPIC or t in BAN_TOKENS or t in GENERIC_NONFLAVOR_UNI:
            return True
        return False

    if any(t in BAN_TOKENS for t in toks):
        return True

    if toks[0] in {"want","need","try","love","like","good","best"}:
        return True

    return False


# --- Candidate builder: generate flavor candidate list with counts & examples
def build_flavor_candidates_low_noise(
    df,
    text_col="text_step2",
    min_count_uni=3,
    min_count_bi=3,
    max_examples=8,
    unigram_cooccur_min_count=2,
    unigram_cooccur_min_prop=0.10,
    top_unigrams=800
):
    # compute corpus token frequency and anchor co-occurrence
    corpus_token_freq = Counter()
    corpus_anchor_cooccur = Counter()
    total_comments = 0

    for _, row in df.iterrows():
        txt = normalize_text(row.get(text_col, ""))
        toks = set(clean_tokens(txt))
        if not toks:
            continue
        total_comments += 1
        corpus_token_freq.update(toks)
        if toks & ANCHORS:
            for t in toks:
                corpus_anchor_cooccur[t] += 1

    if total_comments == 0:
        return pd.DataFrame([])

    counts = {
        "unigram": Counter(),
        "bigram": Counter(),
    }
    plat_counts = defaultdict(lambda: Counter())
    examples = defaultdict(list)

    for _, row in df.iterrows():
        txt = row.get(text_col, "")
        plat = row.get("platform", "unknown")
        raw_comment = row.get("comment_text", "")

        txt = normalize_text(txt)
        toks = clean_tokens(txt)
        if not toks:
            continue

        u_set = set(normalize_phrase(t) for t in toks)
        b_set = set(normalize_phrase(p) for p in ngrams(toks, 2))

        def add_phrase(p: str, phrase_type: str, phrase_tokens: list):
            p = normalize_phrase(p)
            if looks_like_bad_phrase(p):
                return
            if not passes_anchor_gate(
                p, toks, phrase_tokens,
                corpus_token_freq, corpus_anchor_cooccur,
                min_count_uni=min_count_uni,
                cooccur_min_count=unigram_cooccur_min_count,
                cooccur_min_prop=unigram_cooccur_min_prop
            ):
                return

            counts[phrase_type][p] += 1
            plat_counts[p][plat] += 1
            if len(examples[p]) < max_examples:
                examples[p].append(raw_comment)

        for p in u_set:
            add_phrase(p, "unigram", p.split())

        for p in b_set:
            add_phrase(p, "bigram", p.split())

    rows = []
    for phrase_type, counter in counts.items():
        min_count = min_count_uni if phrase_type == "unigram" else min_count_bi

        for phrase, cnt in counter.items():
            if cnt < min_count:
                continue
            rows.append({
                "phrase": phrase,
                "phrase_type": phrase_type,
                "count_total": cnt,
                "count_tiktok": plat_counts[phrase].get("tiktok", 0),
                "count_instagram": plat_counts[phrase].get("instagram", 0),
                "ex1": (examples[phrase][0] if len(examples[phrase]) > 0 else ""),
                "ex2": (examples[phrase][1] if len(examples[phrase]) > 1 else ""),
                "ex3": (examples[phrase][2] if len(examples[phrase]) > 2 else ""),
                "flavor_label": ""   # to be populated after manual labeling
            })

    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand

    uni = cand[cand["phrase_type"] == "unigram"].sort_values("count_total", ascending=False).head(top_unigrams)
    bi = cand[cand["phrase_type"] == "bigram"]
    cand = pd.concat([uni, bi], ignore_index=True)

    cand["phrase_len"] = cand["phrase"].str.split().apply(len)
    cand["is_unigram"] = cand["phrase_len"] == 1
    cand = cand.sort_values(["is_unigram", "count_total", "phrase_len"], ascending=[False, False, False]).drop(columns=["phrase_len"])

    return cand


# --- Generate flavor candidates and save for labeling
flavor_candidates = build_flavor_candidates_low_noise(
    last_10_weeks,
    text_col="text_step2",
    min_count_uni=3,
    min_count_bi=3,
    max_examples=8,
    unigram_cooccur_min_count=2,
    unigram_cooccur_min_prop=0.10,
    top_unigrams=800
)

flavor_candidates.to_csv("flavor_candidates_to_label1.csv", index=False, encoding="utf-8")
print("Saved: flavor_candidates_to_label.csv")
print("TOTAL:", len(flavor_candidates))
print(flavor_candidates["phrase_type"].value_counts())
print(flavor_candidates.head(50))


# --- Load labeled flavor CSV and map phrases to canonical flavor labels
labels = pd.read_csv("flavor_candidates_to_label.csv", encoding="utf-8")

labels["flavor_label"] = labels["flavor_label"].fillna("").astype(str).str.strip()
labels = labels[(labels["flavor_label"] != "") & (labels["flavor_label"].str.lower() != "ignore")].copy()

labels["phrase_norm"] = labels["phrase"].astype(str).str.lower().str.strip()
labels["label_norm"]  = labels["flavor_label"].astype(str).str.lower().str.strip()

phrase_to_label = dict(zip(labels["phrase_norm"], labels["label_norm"]))
print("Mapped phrases:", len(phrase_to_label))
print(labels["label_norm"].value_counts().head(20))


# --- Compile regex from labeled phrases (longest first) and extract flavors per comment
phrases_sorted = sorted(phrase_to_label.keys(), key=lambda x: (len(x.split()), len(x)), reverse=True)

pattern = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in phrases_sorted) + r")\b",
    flags=re.IGNORECASE
)

def extract_flavors(text):
    s = str(text).lower()
    hits = pattern.findall(s)
    labs = []
    for h in hits:
        key = str(h).lower().strip()
        if key in phrase_to_label:
            labs.append(phrase_to_label[key])
    return sorted(set(labs))

last_10_weeks["flavors"] = last_10_weeks["text_step2"].apply(extract_flavors)


# --- Compute weekly top-10 flavors per platform and overall
flavor_long = last_10_weeks.explode("flavors").dropna(subset=["flavors"]).copy()

weekly_flavor = (
    flavor_long
    .groupby(["platform", "week_start", "flavors"])
    .size()
    .reset_index(name="comment_mentions")
)

weekly_flavor["rank"] = (
    weekly_flavor
    .groupby(["platform", "week_start"])["comment_mentions"]
    .rank(method="dense", ascending=False)
)

top10_weekly_flavor = weekly_flavor[weekly_flavor["rank"] <= 10].sort_values(
    ["platform", "week_start", "rank", "flavors"]
)

print(top10_weekly_flavor.head(50))

weekly_flavor_overall = (
    flavor_long
    .groupby(["week_start", "flavors"])
    .size()
    .reset_index(name="comment_mentions")
)

weekly_flavor_overall["rank"] = (
    weekly_flavor_overall
    .groupby("week_start")["comment_mentions"]
    .rank(method="dense", ascending=False)
)

top10_weekly_overall = weekly_flavor_overall[weekly_flavor_overall["rank"] <= 10].sort_values(
    ["week_start", "rank", "flavors"]
)

print(top10_weekly_overall)


# =============================================================================
# SECTION 3: TOP-5 THEME EXTRACTION
# Description: Mine candidate theme phrases, export for labeling, load labels,
#              extract themes per comment, and compute weekly top-5 theme trends.
# =============================================================================

# --- Tokenization and light stopwords for theme extraction
def tokenize(s: str):
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.findall(r"\b\w+\b", s)

STOP_THEME = {
    "i","me","my","you","your","we","our","it","its","they","them","this","that","these","those",
    "a","an","the","and","or","but","to","of","in","on","for","with","as","at","by","from",
    "is","are","was","were","be","been","am","im","s","t","re","ve","ll","m",
    "so","very","just","can","cant","do","did","does","what","when","where","why","how",
    "pls","please","thank","thanks",
}

TOPIC = {"salt", "bread", "saltbread", "roti", "소금빵", "빵", "pan"}

def clean_tokens_theme(txt: str):
    toks = tokenize(txt)
    out = []
    for t in toks:
        if t in STOP_THEME:
            continue
        if t in TOPIC:
            continue
        if t.isdigit():
            continue
        if len(t) < 2:
            continue
        if re.fullmatch(r"[a-z]*\d+[a-z]*", t):
            continue
        out.append(t)
    return out

def ngrams(tokens, n):
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def normalize_phrase(p: str) -> str:
    p = str(p).lower().strip()
    p = re.sub(r"(.)\1{2,}", r"\1", p)
    p = re.sub(r"[_\-]+", " ", p)
    p = re.sub(r"\s+", " ", p).strip()
    return p

# --- Theme anchors and phrase bans to keep candidates relevant
THEME_ANCHORS = {
    "price","pricing","expensive","cheap","overpriced","worth","value",
    "harga","mahal","murah",
    "queue","line","wait","waiting","antri","ngantri","rame","crowd",
    "sold","out","soldout","habis","restock","stock","available","availability",
    "where","location","address","near","nearby","distance","branch","store","shop",
    "maps","google","map",
    "open","close","opened","closed","buka","tutup","jam","hours",
    "order","ordered","preorder","delivery","shipping","pickup","pick","grab","gojek",
    "recipe","resep","how","make","made","bake","baking","oven","ingredients","ingredient",
    "tepung","ragi","butter","mentega",
    "taste","tastes","flavor","enak","gaenak","asin","manis","gurih",
    "texture","soft","crispy","crunchy","dry","fluffy","buttery","fresh","stale",
    "service","staff","employee","rude","friendly","customer","cs","owner",
    "viral","hype","trending","fyp","tiktok","ig","instagram",
}

PHRASE_BAN = {
    "so good", "very good", "love it", "want try", "need try",
    "good job", "thank you", "thanks you",
}

def looks_like_bad_theme_phrase(p: str) -> bool:
    p = normalize_phrase(p)
    if p in PHRASE_BAN:
        return True
    toks = p.split()
    if len(toks) == 1 and toks[0] not in THEME_ANCHORS:
        return True
    if toks and toks[0] in {"want","need","like","love","try","pls","please"}:
        return True
    return False

# --- Candidate miner: build theme candidate list with counts & examples
def build_theme_candidates(
    df,
    text_col="text_step2",
    min_count_bi=4,
    min_count_tri=3,
    max_examples=3,
    context_window=4,
):
    bi = Counter()
    tri = Counter()

    examples = defaultdict(list)
    plat_counts = defaultdict(lambda: Counter())
    week_counts = defaultdict(lambda: Counter())

    for _, row in df.iterrows():
        txt = row.get(text_col, "")
        raw_comment = row.get("comment_text", "")
        plat = row.get("platform", "unknown")
        wk = row.get("week_start", pd.NaT)

        toks = clean_tokens_theme(txt)
        if not toks:
            continue

        anchor_pos = {i for i, t in enumerate(toks) if t in THEME_ANCHORS}

        b_set = set(normalize_phrase(p) for p in ngrams(toks, 2))
        t_set = set(normalize_phrase(p) for p in ngrams(toks, 3))

        def near_anchor(phrase_tokens):
            n = len(phrase_tokens)
            for i in range(len(toks) - n + 1):
                if toks[i:i+n] == phrase_tokens:
                    span = set(range(i, i+n))
                    for pos in span:
                        if any(abs(pos - a) <= context_window for a in anchor_pos):
                            return True
            return False

        def add_phrase(p: str, counter: Counter, phrase_type: str):
            p = normalize_phrase(p)
            if looks_like_bad_theme_phrase(p):
                return
            ptoks = p.split()

            if not (set(ptoks) & THEME_ANCHORS) and not near_anchor(ptoks):
                return

            counter[p] += 1
            plat_counts[p][plat] += 1
            if pd.notna(wk):
                week_counts[p][wk] += 1
            if len(examples[p]) < max_examples:
                examples[p].append(raw_comment)

        for p in b_set:
            add_phrase(p, bi, "bigram")
        for p in t_set:
            add_phrase(p, tri, "trigram")

    rows = []

    def emit(counter, phrase_type, min_count):
        for phrase, cnt in counter.items():
            if cnt < min_count:
                continue
            rows.append({
                "phrase": phrase,
                "phrase_type": phrase_type,
                "count_total": cnt,
                "count_tiktok": plat_counts[phrase].get("tiktok", 0),
                "count_instagram": plat_counts[phrase].get("instagram", 0),
                "weeks_seen": len(week_counts[phrase]),
                "ex1": (examples[phrase][0] if len(examples[phrase]) > 0 else ""),
                "ex2": (examples[phrase][1] if len(examples[phrase]) > 1 else ""),
                "ex3": (examples[phrase][2] if len(examples[phrase]) > 2 else ""),
                "theme_label": ""  # to be populated after manual labeling
            })

    emit(bi, "bigram", min_count_bi)
    emit(tri, "trigram", min_count_tri)

    cand = pd.DataFrame(rows)

    if not cand.empty:
        cand = cand.sort_values(
            ["weeks_seen", "count_total", "phrase_type"],
            ascending=[False, False, True]
        )

    return cand

# --- Generate theme candidates and save for labeling
theme_candidates = build_theme_candidates(
    last_10_weeks,
    text_col="text_step2",
    min_count_bi=4,
    min_count_tri=3,
    max_examples=3,
    context_window=4
)

theme_candidates.to_csv("theme_candidates_to_label.csv2", index=False, encoding="utf-8")
print("Saved: theme_candidates_to_label.csv")
print(theme_candidates.head(50))


# --- Load labeled theme CSV and build mapping from phrase -> theme
theme_df = pd.read_csv("theme_candidates_to_label.csv", encoding="utf-8")

theme_df["theme_label"] = theme_df["theme_label"].fillna("").astype(str).str.strip()
theme_df["phrase_norm"] = theme_df["phrase"].fillna("").astype(str).str.lower().str.strip()

theme_keep = theme_df[
    (theme_df["theme_label"] != "") &
    (theme_df["theme_label"].str.lower() != "ignore") &
    (theme_df["phrase_norm"] != "")
].copy()

theme_keep["theme_label"] = theme_keep["theme_label"].str.lower().str.strip()

phrase_to_theme = dict(zip(theme_keep["phrase_norm"], theme_keep["theme_label"]))

print("Total labeled theme phrases kept:", len(phrase_to_theme))
print("Top themes by #phrases mapped:\n", theme_keep["theme_label"].value_counts().head(20))


# --- Compile regex from labeled theme phrases (longest first) and extract themes per comment
phrases_sorted = sorted(
    phrase_to_theme.keys(),
    key=lambda x: (len(x.split()), len(x)),
    reverse=True
)

pattern_theme = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in phrases_sorted) + r")\b",
    flags=re.IGNORECASE
)

def extract_themes(text):
    s = str(text).lower()
    hits = pattern_theme.findall(s)
    labs = []
    for h in hits:
        key = str(h).lower().strip()
        if key in phrase_to_theme:
            labs.append(phrase_to_theme[key])
    return sorted(set(labs))

last_10_weeks["themes"] = last_10_weeks["text_step2"].apply(extract_themes)

themes_long = (
    last_10_weeks
    .explode("themes")
    .dropna(subset=["themes"])
    .copy()
)

print(themes_long)

# --- Compute weekly theme trends and top-5 per platform-week and overall
weekly_theme = (
    themes_long
    .groupby(["platform", "week_start", "themes"])
    .size()
    .reset_index(name="comment_mentions")
)

print(weekly_theme)

weekly_theme["rank"] = (
    weekly_theme
    .groupby(["platform", "week_start"])["comment_mentions"]
    .rank(method="dense", ascending=False)
)

top5_weekly_theme = (
    weekly_theme[weekly_theme["rank"] <= 5]
    .sort_values(["platform", "week_start", "rank", "themes"])
)

print(top5_weekly_theme.head(50))

weekly_theme_overall = (
    themes_long
    .groupby(["week_start", "themes"])
    .size()
    .reset_index(name="comment_mentions")
)

weekly_theme_overall["rank"] = (
    weekly_theme_overall
    .groupby("week_start")["comment_mentions"]
    .rank(method="dense", ascending=False)
)

top5_weekly_theme_overall = (
    weekly_theme_overall[weekly_theme_overall["rank"] <= 5]
    .sort_values(["week_start", "rank", "themes"])
)


# =============================================================================
# SECTION 4: MOST POPULAR POSTS & COMMENTS (includes sentiment scoring)
# Description: Score sentiment for comments, rank posts by engagement, and list top comments.
# =============================================================================

# --- Load multilingual sentiment pipeline and define scoring helper
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

def get_sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return np.nan
    try:
        result = sentiment_pipe(text[:512])[0]  # truncate long comments
        # label looks like: "1 star", "2 stars", ..., "5 stars"
        stars = int(result["label"][0])
        return stars
    except Exception:
        return np.nan

# --- Apply sentiment scoring to comments
last_10_weeks["sentiment_score"] = last_10_weeks["comment_text"].apply(get_sentiment_score)

last_10_weeks[["comment_text", "sentiment_score"]].head(10)

# --- Weekly sentiment aggregates
weekly_sentiment = (
    last_10_weeks
    .groupby(["platform", "week_start"])
    .agg(
        avg_sentiment=("sentiment_score", "mean"),
        comment_count=("sentiment_score", "count")
    )
    .reset_index()
)

print(weekly_sentiment.head())

weekly_sentiment_overall = (
    last_10_weeks
    .groupby("week_start")
    .agg(
        avg_sentiment=("sentiment_score", "mean"),
        comment_count=("sentiment_score", "count")
    )
    .reset_index()
)

print(weekly_sentiment_overall)

weekly_sentiment = weekly_sentiment.sort_values(["platform", "week_start"])

weekly_sentiment["sentiment_change"] = (
    weekly_sentiment
    .groupby("platform")["avg_sentiment"]
    .diff()
)

biggest_sentiment_moves = (
    weekly_sentiment
    .sort_values("sentiment_change", key=lambda s: s.abs(), ascending=False)
)

print(biggest_sentiment_moves.head(10))


# --- Prepare columns and content_type flag, then rank posts and comments by engagement
print(list(last_10_weeks.columns))
print(sorted(last_10_weeks.columns))
print(last_10_weeks.dtypes)

last_10_weeks = last_10_weeks.rename(columns={
    "COMMENTURL": "comment_url"
})

last_10_weeks["content_type"] = np.where(
    (last_10_weeks["platform"].str.lower() == "tiktok")
    & last_10_weeks["parent_comment_id"].notna(),
    "reply",
    "comment"
)

KEEP_COLS = [
    "platform",
    "content_type",
    "post_url",
    "comment_url",
    "comment_text",
    "likes",
    "reply_count",
    "sentiment_score",
    "comment_time",
    "week_start",
    "parent_comment_id",
]

KEEP_COLS = [c for c in KEEP_COLS if c in last_10_weeks.columns]

last_10_weeks = last_10_weeks.loc[:, KEEP_COLS].copy()

print("COLUMNS:", list(last_10_weeks.columns))
print("\nPLATFORM × CONTENT TYPE")
print(
    last_10_weeks
    .groupby(["platform", "content_type"])
    .size()
    .reset_index(name="rows")
)

for c in ["likes", "reply_count", "sentiment_score"]:
    if c in last_10_weeks.columns:
        last_10_weeks[c] = pd.to_numeric(last_10_weeks[c], errors="coerce").fillna(0)

last_10_weeks["comment_text"] = last_10_weeks["comment_text"].astype(str)


# --- Rank posts by summed comment likes and engagement
top_posts = (
    last_10_weeks
    .groupby(["platform", "post_url"], dropna=False)
    .agg(
        total_comment_likes=("likes", "sum"),
        comment_count=("comment_text", "count"),
        total_reply_count=("reply_count", "sum"),
        avg_sentiment=("sentiment_score", "mean"),
        reply_rows=("content_type", lambda s: (s == "reply").sum()),
        comment_rows=("content_type", lambda s: (s == "comment").sum()),
    )
    .reset_index()
)

top_posts = top_posts.sort_values(
    ["total_comment_likes", "comment_count", "total_reply_count"],
    ascending=[False, False, False]
)

top_posts_20 = top_posts.head(20)


# --- Rank individual comments by likes and replies, dedupe repeated scrapes
comment_cols = [
    "platform",
    "content_type",
    "likes",
    "reply_count",
    "sentiment_score",
    "post_url",
    "comment_url",
    "comment_text",
]
comment_cols = [c for c in comment_cols if c in last_10_weeks.columns]

top_comments = last_10_weeks.loc[:, comment_cols].copy()

top_comments["comment_text"] = top_comments["comment_text"].astype(str).str.strip()
top_comments = top_comments[top_comments["comment_text"] != ""]

top_comments = top_comments.sort_values("likes", ascending=False)

top_comments = top_comments.drop_duplicates(
    subset=["platform", "post_url", "comment_text"],
    keep="first"
)

top_comments = top_comments.sort_values(
    ["likes", "reply_count"],
    ascending=[False, False]
)

top_comments_50 = top_comments.head(50)


# --- Platform and content-type summaries
platform_summary = (
    last_10_weeks
    .groupby("platform", dropna=False)
    .agg(
        rows=("comment_text", "count"),
        total_likes=("likes", "sum"),
        avg_likes=("likes", "mean"),
        total_reply_count=("reply_count", "sum"),
        avg_sentiment=("sentiment_score", "mean"),
        unique_posts=("post_url", pd.Series.nunique),
    )
    .reset_index()
)

type_summary = (
    last_10_weeks
    .groupby(["platform", "content_type"], dropna=False)
    .agg(
        rows=("comment_text", "count"),
        total_likes=("likes", "sum"),
        avg_likes=("likes", "mean"),
        total_reply_count=("reply_count", "sum"),
        avg_sentiment=("sentiment_score", "mean"),
    )
    .reset_index()
)


# =============================================================================
# SECTION 5: CSV / EXCEL FILE COMPILATION
# Description: Coerce datetimes for Excel, collect prepared DataFrames, and write to an Excel workbook.
# =============================================================================

OUT = "SaltBread_for_pivot_test6.xlsx"

def coerce_week_start(df: pd.DataFrame) -> pd.DataFrame:
    if "week_start" in df.columns:
        s = df["week_start"]

        if pd.api.types.is_period_dtype(s):
            df["week_start"] = s.dt.start_time

        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

        if pd.api.types.is_datetime64tz_dtype(df["week_start"]):
            df["week_start"] = df["week_start"].dt.tz_localize(None)

        df["week_start"] = df["week_start"].dt.normalize()

    return df

def strip_excel_timezones(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_datetime64tz_dtype(df[c]):
            df[c] = df[c].dt.tz_localize(None)
    return df

def get_df(name: str) -> pd.DataFrame:
    if name not in globals():
        raise KeyError(f"{name} not found in memory.")
    df = globals()[name].copy()
    df = coerce_week_start(df)
    df = strip_excel_timezones(df)
    return df

# --- Pull prepared DataFrames from memory (will error if missing)
cleaned = get_df("last_10_weeks")
flavor_per_comment = get_df("flavor_long")
theme_per_comment = get_df("themes_long")

weekly_flavor_top10 = get_df("top10_weekly_flavor")
weekly_theme_top5 = get_df("top5_weekly_theme")
weekly_sentiment = get_df("weekly_sentiment")

top_posts_20 = get_df("top_posts_20")
top_comments_50 = get_df("top_comments_50")
platform_summary = get_df("platform_summary")
type_summary = get_df("type_summary")

# --- Select and order columns for export
cols_keep = ["platform","week_start","comment_text","text_step2","likes","reply_count","post_url","sentiment_score"]
cleaned = cleaned[[c for c in cols_keep if c in cleaned.columns]]

cols_keep = ["platform","week_start","comment_text","flavors","likes","reply_count","post_url"]
flavor_per_comment = flavor_per_comment[[c for c in cols_keep if c in flavor_per_comment.columns]]

cols_keep = ["platform","week_start","comment_text","themes","likes","reply_count","post_url"]
theme_per_comment = theme_per_comment[[c for c in cols_keep if c in theme_per_comment.columns]]

# --- Write consolidated workbook containing cleaned data, flavors, themes, and summaries
with pd.ExcelWriter(OUT, engine="xlsxwriter", datetime_format="yyyy-mm-dd") as w:
    cleaned.to_excel(w, "cleaned_comments", index=False)
    flavor_per_comment.to_excel(w, "flavor_per_comment", index=False)
    theme_per_comment.to_excel(w, "theme_per_comment", index=False)

    weekly_flavor_top10.to_excel(w, "weekly_flavor_top10", index=False)
    weekly_theme_top5.to_excel(w, "weekly_theme_top5", index=False)
    weekly_sentiment.to_excel(w, "weekly_sentiment", index=False)

    top_posts_20.to_excel(w, "top_posts_by_engagement", index=False)
    top_comments_50.to_excel(w, "top_comments_by_likes", index=False)
    platform_summary.to_excel(w, "platform_summary", index=False)
    type_summary.to_excel(w, "content_type_summary", index=False)

print(f"Wrote {OUT}")