"""
normalize.py
Normalization of business names and addresses (version 1).

Library use:
    from normalize import normalize_record
    normalize_record("Davis Family Offie", "88 OLIVE CIR, LEBANON, TN", "US")

Command line:
    python src/normalize.py --self-test
    python src/normalize.py --data-dir <path>/student_resource/dataset \
                            --work-dir /kaggle/working/work [--limit 100000]

Output (one Parquet file per source file, in <work-dir>/norm/):
    entity_id    original id
    country      lowercased country label (open set: france etc. pass through)
    name_norm    cleaned name, abbreviations expanded (pvt -> private, ...)
    name_core    name_norm without legal suffixes / stopwords (llc, private, the ...)
    addr_norm    cleaned address, abbreviations expanded, state mapped to a code
    addr_nums    unique numbers in the address (leading zeros removed), sorted,
                 space separated, postcode excluded
    postcode     India 6-digit PIN, or a 5-digit postcode where one is clear
    state        state code if one could be detected (tn, ca, mh, ...), else ""

Everything here is rule based and uses no external data or services.
"""

import argparse
import os
import re
import time
import unicodedata
from multiprocessing import Pool

import pandas as pd

# ============================================================== dictionaries

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "washington dc": "dc", "florida": "fl",
    "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va",
    "washington": "wa", "west virginia": "wv", "wisconsin": "wi",
    "wyoming": "wy", "puerto rico": "pr",
}

INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as",
    "bihar": "br", "chhattisgarh": "cg", "chattisgarh": "cg", "ct": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "or": "od",
    "punjab": "pb", "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn",
    "tamilnadu": "tn", "telangana": "tg", "ts": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "ua": "uk",
    "west bengal": "wb", "delhi": "dl", "nct of delhi": "dl", "nct delhi": "dl",
    "jammu and kashmir": "jk", "jammu kashmir": "jk", "ladakh": "la",
    "puducherry": "py", "pondicherry": "py", "chandigarh": "ch",
    "andaman and nicobar islands": "an", "andaman and nicobar": "an",
    "dadra and nagar haveli and daman and diu": "dn",
    "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}


def _with_codes(states):
    """Allow both 'tennessee' and 'tn' to map to 'tn'."""
    out = dict(states)
    for code in set(states.values()):
        out.setdefault(code, code)
    return out


def _with_translit(states):
    """Also accept state names without their final 'a' ("maharashtr",
    "telangan"), which is how transliterated Indian-script names come out."""
    out = dict(states)
    for name, code in states.items():
        if name.endswith("a") and len(name) > 4:
            out.setdefault(name[:-1], code)
    return out


STATE_MAPS = {"us": _with_codes(US_STATES),
              "india": _with_codes(_with_translit(INDIA_STATES))}

# Name abbreviations (applied token by token).
NAME_ABBREV = {
    "pvt": "private", "pvte": "private", "prv": "private", "pvt.": "private",
    "ltd": "limited", "ltda": "limited", "lmtd": "limited", "ltd.": "limited",
    "incorporated": "inc", "incorporation": "inc",
    "corp": "corporation", "corpn": "corporation",
    "co": "company", "cos": "company",
    "intl": "international", "natl": "national", "mfg": "manufacturing",
    "svcs": "services", "svc": "service", "mgmt": "management",
    "assoc": "associates", "assocs": "associates", "bros": "brothers",
    "engg": "engineering", "grp": "group", "hldgs": "holdings",
    "inds": "industries", "ent": "enterprises", "entp": "enterprises",
    "tech": "technology", "techs": "technologies", "tradg": "trading",
    "praivet": "private", "praibhet": "private", "prayivet": "private",
    "limitted": "limited", "kampani": "company", "kanpani": "company",
}

# Removed when building name_core.
NAME_STOP = {
    "private", "limited", "inc", "corporation", "company", "llc", "llp",
    "pllc", "plc", "lp", "pc", "pa", "the", "and", "of", "sarl", "sas", "sasu",
    "sa", "eurl", "sci", "snc", "gmbh", "ag", "bv", "nv", "pty",
}

# Address abbreviations used for every country (including unseen ones).
ADDR_ABBREV_COMMON = {
    "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue",
    "aven": "avenue", "blvd": "boulevard", "boul": "boulevard",
    "bd": "boulevard", "dr": "drive", "drv": "drive", "ln": "lane",
    "cir": "circle", "circ": "circle", "crcl": "circle", "ct": "court",
    "pl": "place", "pkwy": "parkway", "pky": "parkway", "hwy": "highway",
    "sq": "square", "ter": "terrace", "terr": "terrace", "trl": "trail",
    "cres": "crescent", "expy": "expressway", "fwy": "freeway",
    "tpke": "turnpike", "hts": "heights", "jct": "junction", "xing": "crossing",
    "ste": "suite", "apt": "apartment", "apts": "apartment",
    "appts": "apartment", "aptmt": "apartment", "aptmts": "apartment",
    "apartments": "apartment", "bldg": "building", "bld": "building",
    "flr": "floor", "fl": "floor", "rm": "room",
}

ADDR_ABBREV_US = {
    "st": "street", "str": "street", "mt": "mount", "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
}

ADDR_ABBREV_INDIA = {
    "st": "street", "str": "street", "opp": "opposite", "opps": "opposite",
    "nr": "near", "mkt": "market", "stn": "station", "clny": "colony",
    "col": "colony", "sec": "sector", "sect": "sector", "ph": "phase",
    "vill": "village", "vil": "village", "dist": "district",
    "distt": "district", "tal": "taluka", "tq": "taluka", "taluk": "taluka",
    "mg": "mahatma gandhi",
    # old / alternate city names -> one spelling
    "bangalore": "bengaluru", "bombay": "mumbai", "madras": "chennai",
    "calcutta": "kolkata", "trivandrum": "thiruvananthapuram",
    "poona": "pune", "baroda": "vadodara", "mysore": "mysuru",
    "gurgaon": "gurugram", "cochin": "kochi", "vizag": "visakhapatnam",
    "benares": "varanasi", "banaras": "varanasi", "calicut": "kozhikode",
    "mangalore": "mangaluru", "belgaum": "belagavi", "cuddapah": "kadapa",
    "allahabad": "prayagraj", "trichy": "tiruchirappalli",
}

# France: no training data, so these are generic rules only.
# (Check with the challenge FAQ that hand-written rules like these are allowed.)
ADDR_ABBREV_FRANCE = {
    "st": "saint", "ste": "sainte", "bd": "boulevard", "bld": "boulevard",
    "ch": "chemin", "chem": "chemin", "rte": "route", "imp": "impasse",
    "all": "allee", "qu": "quai", "fbg": "faubourg", "crs": "cours",
    "pte": "porte", "res": "residence",
}

ADDR_STOP = {"no", "hno", "nos", "number", "num", "na", "nil", "null",
             "none", "unknown", "cedex"}

# Words that mark a component as a street (used for postcode detection).
STREET_WORDS = {
    "street", "road", "avenue", "boulevard", "drive", "lane", "circle", "court",
    "place", "parkway", "highway", "square", "terrace", "trail", "crescent",
    "expressway", "freeway", "turnpike", "way", "rue", "chemin", "route",
    "impasse", "allee", "quai", "cours", "marg", "nagar",
}


def _addr_map(country):
    table = dict(ADDR_ABBREV_COMMON)
    if country == "us":
        table.update(ADDR_ABBREV_US)
    elif country == "india":
        table.update(ADDR_ABBREV_INDIA)
    elif country == "france":
        table.update(ADDR_ABBREV_FRANCE)
    return table


ADDR_MAPS = {c: _addr_map(c) for c in ("us", "india", "france", "_other")}

# ================================================================== regexes

_WS = re.compile(r"\s+")
# Keeps letters, digits and Indic vowel signs (U+0900-U+0DFF); everything else
# (punctuation, symbols, underscores) becomes a space.
_NON_ALNUM = re.compile(r"[^\w\s\u0900-\u0DFF]|_", flags=re.UNICODE)
_DOTTED_LETTER = re.compile(r"\b([a-z])\.")
_DOMAIN = re.compile(r"\b(?:https?://)?(?:www\.)?([a-z0-9-]+)\."
                     r"(?:co\.in|com|in|net|org|co|biz|info|fr|us)\b")
_MS_PREFIX = re.compile(r"\bm\s*/\s*s\b")
# Record ids pasted into names: "(ID: 81383)", "id#4521", "id 77"
_NAME_ID = re.compile(r"\bid\s*[:#.\-]?\s*\d+")
# Numbers of 5+ digits in a name are phone numbers / reference codes
# ("Element Marketing - 8731858442", "Partners #71910"), not part of the name.
NAME_MAX_DIGITS = 4
_CARE_OF = re.compile(r"\b[cs]\s*/\s*o\b")
_NA = re.compile(r"\bn\s*/\s*a\b")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
_LETTER_DIGIT = re.compile(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)")
_COMPONENT_SPLIT = re.compile(r"[,;|]")


# ================================================================ helpers

def strip_accents(text):
    """Remove accents from Latin letters only (e -> e for 'é'). Other scripts
    (Telugu, Devanagari, ...) are left untouched so their words stay intact."""
    out = []
    for ch in text:
        if ord(ch) < 0x250:
            out.extend(c for c in unicodedata.normalize("NFKD", ch)
                       if not unicodedata.combining(c))
        else:
            out.append(ch)
    return "".join(out)


# ============================================================ Indian scripts
# Names and addresses sometimes appear in Devanagari, Gujarati, Odia, Telugu,
# Bengali, Tamil, Kannada, Malayalam or Gurmukhi. We romanise them using only
# the Unicode character NAMES that ship with Python (unicodedata), e.g.
# "DEVANAGARI LETTER PA" -> "pa". No external data or library is used.

_INDIC = re.compile(r"[\u0900-\u0DFF]")
_INDIC_VOWELS = {"A": "a", "AA": "a", "I": "i", "II": "i", "U": "u", "UU": "u",
                 "E": "e", "EE": "e", "AI": "ai", "O": "o", "OO": "o", "AU": "au",
                 "VOCALIC R": "ri", "VOCALIC RR": "ri", "VOCALIC L": "li",
                 "SHORT E": "e", "SHORT O": "o", "CANDRA E": "e", "CANDRA O": "o"}
_INDIC_CONS = {"TTA": "t", "TTHA": "th", "DDA": "d", "DDHA": "dh", "NNA": "n",
               "SSA": "sh", "SHA": "sh", "LLA": "l", "RRA": "r", "NNNA": "n",
               "LLLA": "l", "NGA": "n", "NYA": "n", "CA": "ch", "CHA": "chh"}
_SCHWA_END = re.compile(r"\b(\w{2,}?[^aeiou\s\d])a\b")


def transliterate_indic(text):
    """Romanise Indian-script letters; other characters pass through."""
    if not _INDIC.search(text):
        return text
    out, prev_cons = [], False
    for ch in text:
        if not ("\u0900" <= ch <= "\u0DFF"):
            out.append(ch)
            prev_cons = False
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        part = name.split(" ", 1)[1] if " " in name else name
        if part.startswith("LETTER "):
            letter = part[7:]
            if letter in _INDIC_VOWELS:
                out.append(_INDIC_VOWELS[letter])
                prev_cons = False
            else:
                base = _INDIC_CONS.get(letter, letter[:-1].lower()
                                       if letter.endswith("A") else letter.lower())
                out.append(base + "a")          # consonant with inherent 'a'
                prev_cons = True
        elif part.startswith("VOWEL SIGN ") or part.startswith("SIGN VIRAMA"):
            if prev_cons and out and out[-1].endswith("a"):
                out[-1] = out[-1][:-1]          # vowel sign / virama replaces the 'a'
            if part.startswith("VOWEL SIGN "):
                out.append(_INDIC_VOWELS.get(part[11:], ""))
            prev_cons = False
        elif part.startswith(("SIGN ANUSVARA", "SIGN CANDRABINDU")):
            out.append("n")
            prev_cons = False
        elif part.startswith("SIGN VISARGA"):
            out.append("h")
            prev_cons = False
        elif part.startswith("DIGIT "):
            out.append(str(unicodedata.digit(ch, "")))
            prev_cons = False
    # spoken Hindi drops the final inherent 'a' ("limiteda" -> "limited")
    return _SCHWA_END.sub(r"\1", "".join(out))


def _base_clean(text):
    """Romanise Indian scripts, lowercase, remove accents, turn '&' into 'and',
    drop apostrophes."""
    text = strip_accents(transliterate_indic(str(text)).lower())
    text = text.replace("&", " and ").replace("'", "").replace("\u2019", "")
    return _DOTTED_LETTER.sub(r"\1", text)


def _dedupe_consecutive(tokens):
    out = []
    for tok in tokens:
        if not out or out[-1] != tok:
            out.append(tok)
    return out


def _strip_zeros(tok):
    if tok.isdigit():
        return tok.lstrip("0") or "0"
    return tok


# ================================================================== names

_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})
_LEET_TOKEN = re.compile(r"^[a-z]+[0-9][a-z0-9]*$")


def _fix_leet(tok):
    """Digits typed for letters inside a word: 'techn0logy' -> 'technology',
    's0lutions' -> 'solutions'. Only for tokens that start with at least one
    letter and contain at least 3 letters, so '3m' or 'b2' stay unchanged."""
    if _LEET_TOKEN.match(tok) and sum(ch.isalpha() for ch in tok) >= 3:
        return tok.translate(_LEET)
    return tok


def normalize_name(name):
    """Returns (name_norm, name_core)."""
    text = _base_clean(name)
    text = _DOMAIN.sub(r" \1 ", text)
    text = _MS_PREFIX.sub(" ", text)
    text = _NAME_ID.sub(" ", text)
    text = _NON_ALNUM.sub(" ", text)

    tokens = []
    for tok in text.split():
        if tok.isdigit() and len(tok) > NAME_MAX_DIGITS:
            continue
        tok = _fix_leet(tok)
        mapped = NAME_ABBREV.get(tok, tok)
        tokens.extend(mapped.split())
    tokens = _dedupe_consecutive(tokens)

    core = [t for t in tokens if t not in NAME_STOP]
    name_norm = " ".join(tokens)
    name_core = " ".join(core) if core else name_norm
    return name_norm, name_core


# ================================================================ addresses

def _clean_component(component):
    text = _CARE_OF.sub(" ", component)
    text = _NA.sub(" ", text)
    text = _ORDINAL.sub(r"\1", text)
    text = _NON_ALNUM.sub(" ", text)
    text = _LETTER_DIGIT.sub(" ", text)
    return text.split()


def _match_state(raw_tokens, states):
    """Returns (state_code, trailing_digits) if the component is a state
    (optionally followed by a ZIP / PIN), else (None, None)."""
    if not states or not raw_tokens:
        return None, None
    whole = " ".join(raw_tokens)
    if whole in states:
        return states[whole], None
    if len(raw_tokens) >= 2 and raw_tokens[-1].isdigit():
        head = " ".join(raw_tokens[:-1])
        if head in states:
            return states[head], raw_tokens[-1]
    return None, None


def normalize_address(address, country):
    """Returns (addr_norm, addr_nums, postcode, state)."""
    country = (country or "").strip().lower()
    states = STATE_MAPS.get(country)
    abbrev = ADDR_MAPS.get(country, ADDR_MAPS["_other"])

    text = _base_clean(address)
    all_tokens = []
    numbers = []          # digit tokens exactly as written (for postcode rules)
    state = ""
    postcode = ""

    for component in _COMPONENT_SPLIT.split(text):
        raw = _clean_component(component)
        if not raw:
            continue

        code, zip_digits = _match_state(raw, states)
        if code:
            state = code
            all_tokens.append(code)
            if zip_digits:
                numbers.append(zip_digits)
                if len(zip_digits) in (5, 6):
                    postcode = zip_digits
                all_tokens.append(_strip_zeros(zip_digits))
            continue

        mapped = []
        for tok in raw:
            if tok in ADDR_STOP:
                continue
            mapped.extend(abbrev.get(tok, tok).split())
        mapped = _dedupe_consecutive(mapped)

        digits = [t for t in raw if t.isdigit()]
        numbers.extend(digits)

        # Postcode outside India: a 5-digit number starting a component that
        # has no street word and no other number, e.g. "75008 paris".
        if (country != "india" and len(raw) >= 1 and raw[0].isdigit()
                and len(raw[0]) == 5 and len(digits) == 1
                and not (set(mapped) & STREET_WORDS)):
            postcode = raw[0]
        elif country != "india" and len(raw) == 1 and raw[0].isdigit() \
                and len(raw[0]) == 5:
            postcode = raw[0]

        all_tokens.extend(_strip_zeros(t) for t in mapped)

    if country == "india":
        pins = [n for n in numbers if len(n) == 6 and n[0] != "0"]
        if pins:
            postcode = pins[-1]

    nums = {_strip_zeros(n) for n in numbers}
    if postcode:
        nums.discard(_strip_zeros(postcode))
    addr_nums = " ".join(sorted(nums, key=lambda x: (len(x), x)))

    addr_norm = " ".join(_dedupe_consecutive(all_tokens))
    return addr_norm, addr_nums, postcode, state


# ================================================================== records

def normalize_record(name, address, country):
    name_norm, name_core = normalize_name(name)
    addr_norm, addr_nums, postcode, state = normalize_address(address, country)
    return (name_norm, name_core, addr_norm, addr_nums, postcode, state)


OUT_COLUMNS = ["name_norm", "name_core", "addr_norm", "addr_nums",
               "postcode", "state"]


def _process_chunk(chunk):
    names, addresses, countries = chunk
    return [normalize_record(n, a, c) for n, a, c in zip(names, addresses, countries)]


def normalize_frame(df, workers=None, chunk_size=100_000):
    """df needs entity_id, business_name, business_address, country."""
    names = df["business_name"].tolist()
    addrs = df["business_address"].tolist()
    ctrys = df["country"].str.strip().str.lower().tolist()
    chunks = [(names[i:i + chunk_size], addrs[i:i + chunk_size], ctrys[i:i + chunk_size])
              for i in range(0, len(names), chunk_size)]

    workers = workers or os.cpu_count() or 1
    rows = []
    if workers == 1 or len(chunks) == 1:
        for ch in chunks:
            rows.extend(_process_chunk(ch))
    else:
        with Pool(workers) as pool:
            for part in pool.imap(_process_chunk, chunks):
                rows.extend(part)

    out = pd.DataFrame(rows, columns=OUT_COLUMNS)
    out.insert(0, "country", ctrys)
    out.insert(0, "entity_id", df["entity_id"].tolist())
    return out


# =============================================================== self test

SELF_TEST = [
    ("Davis Family Office", "88 Olive Circle, Lebanon, TN", "US"),
    ("Davis Family Offie", "88 OLIVE CIR, LEBANON, TN", "US"),
    ("Davis Family (Office)", "88 Olive Cir, Lebanon, Tennessee", "US"),
    ("Traum Peak Logistics P.C.", "82 Rogers Street, Atlanta, GA", "US"),
    ("DRAYEX  BERTO LLC", "##19821 WHEELWRIGHT DR, MONTGOMERY VILLAGE, MD", "US"),
    ("Grand Connecticut", "CALUMET CITY, 351 HOXIE AVE, N/A, IL", "US"),
    ("Chang,  Crawford and Williams Cayson",
     "Unit Unit 119, Wisconsin, Pewaukee, 00700 Quinlan Drive", "US"),
    ("MS Consultancy Corp",
     "Shymala Appts 1St Floor Flat No. 4 Opp Ratna Hospital Sb Road In Haveli, "
     "Pune, Maharashtra", "India"),
    ("MS [Consultancy]",
     "SHYMALA APPTS 1-1ST FLOOR FLAT NO. 4 OPP RATNA HOSPITAL SB ROAD IN HAVELI, "
     "PUNE, Maharashtra", "India"),
    ("TEAMAIR.COM", "FLAT NO:101, ANUSKA TOWERS, OPP. MERCEDES BENZ SHOW ROOM, "
     "LAKDI- KA, -POOL, \u0c24\u0c46\u0c32\u0c02\u0c17\u0c3e\u0c23", "India"),
    ("Team Air Pvt. Ltd.", "Flat No:101, Anuska Towers, Opp. Mercedes Benz Show Room, "
     "Lakdi- Ka, -Pool, Hyderabad, Telangana", "India"),
    ("Pvt XRW Moulding Ltd",
     "C/o Venkata Ramana N, Mullavandlakota, Chinnamandem, Cuddapah, AP", "India"),
    ("Vaibhav Formulations LLP",
     "15/64 Garden House, Vellayambalam, Trivandrum, Kerala 695010", "India"),
    ("Boulangerie Dupont SARL", "12 bis Av. des Champs-\u00c9lys\u00e9es, 75008 Paris",
     "France"),
]


def self_test():
    for name, addr, country in SELF_TEST:
        n_norm, n_core, a_norm, a_nums, pc, st = normalize_record(name, addr, country)
        print(f"\n[{country}] {name} | {addr}")
        print(f"   name_norm: {n_norm}")
        print(f"   name_core: {n_core}")
        print(f"   addr_norm: {a_norm}")
        print(f"   addr_nums: {a_nums!r}   postcode: {pc!r}   state: {st!r}")


# ===================================================================== CLI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="folder that contains train/ and test/")
    parser.add_argument("--work-dir", default="/kaggle/working/work")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--limit", type=int, default=None,
                        help="only read the first N rows of each file (for timing tests)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.data_dir:
        parser.error("--data-dir is required unless --self-test is used")

    out_dir = os.path.join(args.work_dir, "norm")
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()

    for split in args.splits.split(","):
        for i in (1, 2, 3):
            path = os.path.join(args.data_dir, split, f"{split}_source{i}.tsv")
            t0 = time.time()
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                             nrows=args.limit)
            out = normalize_frame(df, workers=args.workers)
            suffix = f"_limit{args.limit}" if args.limit else ""
            out_path = os.path.join(out_dir, f"{split}_source{i}{suffix}.parquet")
            out.to_parquet(out_path, index=False)
            print(f"[{time.time() - start:7.1f}s] {split}_source{i}: {len(out):,} rows "
                  f"in {time.time() - t0:.1f}s -> {out_path}", flush=True)
            del df, out

    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
