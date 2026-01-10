end = stage("Step 5: # ------------------")
# ------------------
# 4) Basic feature engineering (tables)
# ------------------
# Count of categories across train+test
all_counts = pd.concat([df_train["RecipeCategory"], df_test["RecipeCategory"]]).value_counts().to_dict()
df_train["RecipeCategory_count"] = df_train["RecipeCategory"].map(all_counts)
df_test["RecipeCategory_count"] = df_test["RecipeCategory"].map(all_counts)

# One-hot for RecipeCategory & Ingredients
df_all = pd.concat([df_train, df_test], axis=0)

# simple one-hot for RecipeCategory
df_cat_onehot = pd.get_dummies(df_all["RecipeCategory"], prefix="RecipeCategory").astype(int)

# Multi-one-hot for RecipeIngredientParts
mlb_ing = MultiLabelBinarizer()
df_gre_onehot = pd.DataFrame(
    mlb_ing.fit_transform(df_all["RecipeIngredientParts"].apply(to_list)),
    columns=[f"{c}_oh" for c in mlb_ing.classes_],
    index=df_all.index,
).astype(int)

df_all = pd.concat([df_all, df_cat_onehot, df_gre_onehot], axis=1)
df_train = df_all.iloc[: len(df_train)].copy()
df_test = df_all.iloc[len(df_train) :].copy()

# Times → minutes
for c in ["CookTime", "PrepTime", "TotalTime"]:
    df_train[c] = df_train[c].apply(duration_to_minutes).astype(np.int32)
    df_test[c] = df_test[c].apply(duration_to_minutes).astype(np.int32)


# Keywords multi-one-hot (fit on both)
def safe_to_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return []


for df_ in (df_train, df_test):
    df_["Keywords"] = df_["Keywords"].map(safe_to_list)

mlb_kw = MultiLabelBinarizer()
mlb_kw.fit(df_train["Keywords"].tolist() + df_test["Keywords"].tolist())
train_kw = pd.DataFrame(mlb_kw.transform(df_train["Keywords"]), columns=mlb_kw.classes_, index=df_train.index)
test_kw = pd.DataFrame(mlb_kw.transform(df_test["Keywords"]), columns=mlb_kw.classes_, index=df_test.index)
df_train = pd.concat([df_train, train_kw], axis=1)
df_test = pd.concat([df_test, test_kw], axis=1)


# text-list meta features
def compute_text_list_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def ensure_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                p = ast.literal_eval(v)
                if isinstance(p, list):
                    return p
            except Exception:
                return []
        return []

    df["Instruction_steps"] = df["RecipeInstructions"].apply(lambda x: len(ensure_list(x))).astype(np.int16)
    df["Instruction_char_len"] = (
        df["RecipeInstructions"].apply(lambda x: sum(len(str(s)) for s in ensure_list(x))).astype(np.int32)
    )
    df["Instruction_word_count"] = (
        df["RecipeInstructions"]
        .apply(lambda x: sum(len(str(s).split()) for s in ensure_list(x)))
        .astype(np.int32)
    )

    df["Images_count"] = df["Images"].apply(lambda x: len(ensure_list(x))).astype(np.int16)
    df["Has_images"] = (df["Images_count"] > 0).astype(np.int8)

    df["Description_char_len"] = df["Description"].fillna("").apply(len).astype(np.int32)
    df["Description_word_count"] = (
        df["Description"].fillna("").apply(lambda x: len(str(x).split())).astype(np.int32)
    )

    df["Name_char_len"] = df["Name"].fillna("").apply(len).astype(np.int32)
    df["Name_word_count"] = df["Name"].fillna("").apply(lambda x: len(str(x).split())).astype(np.int16)
    return df


df_train = compute_text_list_features(df_train)
df_test = compute_text_list_features(df_test)

# Ingredient pairing & quantity float
NO_QUANT_TERMS = set(
    map(
        str.lower,
        ["parsley", "cilantro", "basil", "mint", "chives", "dill", "oregano", "thyme", "rosemary", "sage", "coriander", "coriander", "leaves", "coriander", "leaf", "scallion", "scallions", "green", "onion", "green", "onions", "spring", "onion", "spring", "onions", "salt", "pepper", "black", "pepper", "white", "pepper", "paprika", "chili", "powder", "cayenne", "cumin", "turmeric", "cinnamon", "nutmeg", "allspice", "clove", "cloves", "garnish", "lemon", "zest", "lemon", "rind", "lemon", "peel", "orange", "zest", "orange", "rind", "orange", "peel", "zest", "rind", "peel"],
    )
)
NO_QUANT_REGEXES = [
    re.compile(r"\b(for|to)\s+(garnish|serve|serving)\b", re.I),
    re.compile(r"\b(zest|rind|peel)\b", re.I),
]


def clean_quantity_token(s: str) -> str:
    t = str(s).strip().replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)
    return t.strip(" ,;")


def is_noquant_candidate(part: str) -> bool:
    p = part.lower().strip()
    if p in NO_QUANT_TERMS:
        return True
    for rx in NO_QUANT_REGEXES:
        if rx.search(p):
            return True
    if "lemon" in p and re.search(r"\b(zest|rind|peel)\b", p):
        return True
    if "juice" in p:
        return False
    return False


def choose_0_positions(parts, needed):
    cands = [i for i, p in enumerate(parts) if is_noquant_candidate(p)]
    sel = cands[:needed]
    i = len(parts) - 1
    while len(sel) < needed and i >= 0:
        if i not in sel:
            sel.append(i)
        i -= 1
    return set(sel)


def compress_extra_quants(quants, target_len):
    if len(quants) <= target_len:
        return quants
    head = quants[: max(0, target_len - 1)]
    tail = [clean_quantity_token(x) for x in quants[max(0, target_len - 1) :]]
    return head + [" ".join(tail)]


def smart_align_row(parts, quants):
    parts = parts if isinstance(parts, list) else []
    quants = quants if isinstance(quants, list) else []
    if len(parts) == 0 and len(quants) > 0:
        return [], []
    if len(quants) > len(parts):
        quants = compress_extra_quants(quants, len(parts))
    if len(parts) > len(quants):
        needed = len(parts) - len(quants)
        zero_pos = choose_0_positions(parts, needed)
        aligned, j = [], 0
        for i in range(len(parts)):
            if i in zero_pos:
                aligned.append(0)
            else:
                aligned.append(quants[j] if j < len(quants) else 0)
                j += 1
        quants = aligned
    return parts, quants


def clean_lists(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RecipeIngredientParts"] = (
        df["RecipeIngredientParts"]
        .apply(to_list)
        .apply(lambda L: [str(x).strip() for x in L if str(x).strip() != ""])
    )
    df["RecipeIngredientQuantities"] = (
        df["RecipeIngredientQuantities"]
        .apply(to_list)
        .apply(lambda L: [clean_quantity_token(str(x)) for x in L if str(x).strip() != ""])
    )
    return df


def process_df_smart(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_lists(df)
    fixed = df.apply(
        lambda r: smart_align_row(r["RecipeIngredientParts"], r["RecipeIngredientQuantities"]), axis=1
    )
    df["RecipeIngredientParts"] = fixed.apply(lambda x: x[0])
    df["RecipeIngredientQuantities"] = fixed.apply(lambda x: x[1])
    df["Parts_len"] = df["RecipeIngredientParts"].apply(len)
    df["Quants_len"] = df["RecipeIngredientQuantities"].apply(len)
    df["Len_diff"] = (df["Parts_len"] - df["Quants_len"]).astype(np.int16)
    return df


df_train = process_df_smart(df_train)
df_test = process_df_smart(df_test)


def quantity_to_float(q):
    if q is None:
        return 0.0
    if not isinstance(q, str):
        try:
            return float(q)
        except Exception:
            return 0.0
    s = q.strip().lower()
    if any(kw in s for kw in ["taste", "needed", "pinch", "dash", "few", "several"]):
        return 0.0
    if re.match(r"^\d+(\s*\d*/\d+)?\s*-\s*\d+(\s*\d*/\d+)?$", s) or " to " in s:
        parts = re.split(r"-|to", s)
        vals = [quantity_to_float(p.strip()) for p in parts if p.strip()]
        vals = [v for v in vals if v is not None]
        return float(sum(vals) / len(vals)) if vals else 0.0
    if re.match(r"^\d+\s+\d+/\d+$", s):
        whole, frac = s.split()
        return float(whole) + float(Fraction(frac))
    if re.match(r"^\d+/\d+$", s):
        return float(Fraction(s))
    try:
        return float(s)
    except Exception:
        return 0.0


def convert_quantities(df):
    df = df.copy()
    df["RecipeIngredientQuantities"] = df["RecipeIngredientQuantities"].apply(
        lambda L: [quantity_to_float(x) for x in L]
    )
    return df


df_train = convert_quantities(df_train)
df_test = convert_quantities(df_test)


# Ingredient features + top ingredients one-hot
def build_ingredient_features(df):
    df = df.copy()

    def normalize_part(p):
        p = str(p).strip().lower()
        p = re.sub(r"\s+", " ", p)
        # 軽量ステミング
        if p.endswith("s"):
            p = p[:-1]
        # 簡易同義語（例）
        p = p.replace("scallion", "green_onion")
        return p

    df["IngredientParts_List"] = df["RecipeIngredientParts"].apply(
        lambda parts: [normalize_part(p) for p in parts if str(p).strip()]
    )

    def ensure_float_list(values):
        out = []
        for v in values if isinstance(values, list) else []:
            try:
                out.append(float(v))
            except:
                out.append(0.0)
        return out

    df["IngredientQuantities_List"] = df["RecipeIngredientQuantities"].apply(ensure_float_list)

    df["IngredientsText"] = df["IngredientParts_List"].apply(lambda parts: " ".join(parts))

    def pair_text(parts, quants):
        pairs = []
        for p, q in zip(parts, quants, strict=False):
            try:
                qv = float(q)
            except:
                qv = 0.0
            pairs.append(f"{p}_{qv:.3f}")
        return " ".join(pairs)

    df["IngredientPairText"] = [
        pair_text(p, q) for p, q in zip(df["IngredientParts_List"], df["IngredientQuantities_List"], strict=False)
    ]

    qs = df["IngredientQuantities_List"]
    df["IngredientQty_sum"] = qs.apply(lambda x: float(np.sum(x)) if x else 0.0)
    df["IngredientQty_mean"] = qs.apply(lambda x: float(np.mean(x)) if x else 0.0)
    df["IngredientQty_max"] = qs.apply(lambda x: float(np.max(x)) if x else 0.0)
    df["IngredientQty_min"] = qs.apply(lambda x: float(np.min(x)) if x else 0.0)
    df["IngredientQty_std"] = qs.apply(lambda x: float(np.std(x)) if len(x) > 1 else 0.0)
    df["IngredientQty_nonzero"] = qs.apply(lambda x: int(np.sum(np.array(x) > 0)) if x else 0)
    df["IngredientQty_count"] = qs.apply(len).astype(np.int16)

    df["Ingredient_unique_count"] = df["IngredientParts_List"].apply(lambda p: len(set(p))).astype(np.int16)
    df["Ingredient_name_char_len"] = (
        df["IngredientParts_List"].apply(lambda p: sum(len(s) for s in p)).astype(np.int32)
    )
    df["Ingredient_name_word_count"] = (
        df["IngredientParts_List"].apply(lambda p: sum(len(s.split()) for s in p)).astype(np.int16)
    )
    return df


df_train = build_ingredient_features(df_train)
df_test = build_ingredient_features(df_test)

# Top ingredients OHE（train+test合算で上位200）
counter = Counter()
for S in df_train["IngredientParts_List"].apply(set):
    counter.update(S)
for S in df_test["IngredientParts_List"].apply(set):
    counter.update(S)
TOP_ING = [ing for ing, _ in counter.most_common(200)]
exist_cols = set(df_train.columns) | set(df_test.columns)

for ing in TOP_ING:
    base = re.sub(r"[^0-9a-zA-Z]+", "_", ing).strip("_") or "ingredient"
    col = f"ing_{base}"
    k = 2
    while col in exist_cols:
        col = f"ing_{base}_{k}"
        k += 1
    exist_cols.add(col)
    df_train[col] = df_train["IngredientParts_List"].apply(lambda s: np.uint8(ing in set(s)))
    df_test[col] = df_test["IngredientParts_List"].apply(lambda s: np.uint8(ing in set(s)))


# Ingredient ratio features
def add_ingredient_ratio_features(df):
    df = df.copy()

    def safe_ratio(a, b):
        b = b.replace(0, np.nan)
        r = (a / b).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return r.astype(np.float32)

    df["IngredientQty_per_serving"] = safe_ratio(df["IngredientQty_sum"], df["RecipeServings"])
    df["IngredientQty_per_step"] = safe_ratio(df["IngredientQty_sum"], df["Instruction_steps"].astype(float))
    df["Ingredient_unique_ratio"] = safe_ratio(
        df["Ingredient_unique_count"].astype(float), df["IngredientQty_count"].astype(float)
    )
    df["Ingredient_density_per_min"] = safe_ratio(
        df["IngredientQty_count"].astype(float), df["TotalTime"].astype(float)
    )
    df["Ingredient_char_len_per_part"] = safe_ratio(
        df["Ingredient_name_char_len"].astype(float), df["IngredientQty_count"].astype(float)
    )
    # log transform
    for c in ["IngredientQty_sum", "IngredientQty_mean", "IngredientQty_std"]:
        df[f"log_{c}"] = np.log1p(df[c].clip(lower=0)).astype(np.float32)
    # interactions
    df["Cook_per_Step"] = (df["CookTime"] / (1.0 + df["Instruction_steps"])).astype(np.float32)
    return df


df_train = add_ingredient_ratio_features(df_train)
df_test = add_ingredient_ratio_features(df_test)


# DatePublished → stats（ns→secへ安全変換）
def date_stats(series, prefix):
    times = series.apply(lambda x: [pd.to_datetime(x).value / 1e9] if pd.notna(x) else [])
    stats = pd.DataFrame(index=series.index)
    stats[f"{prefix}_min"] = times.apply(lambda L: min(L) if len(L) > 0 else np.nan)
    stats[f"{prefix}_max"] = times.apply(lambda L: max(L) if len(L) > 0 else np.nan)
    stats[f"{prefix}_mean"] = times.apply(lambda L: sum(L) / len(L) if len(L) > 0 else np.nan)
    stats[f"{prefix}_count"] = times.apply(lambda L: len(L))
    return stats


df_train = pd.concat([df_train, date_stats(df_train["DatePublished"], "DatePublished")], axis=1)
df_test = pd.concat([df_test, date_stats(df_test["DatePublished"], "DatePublished")], axis=1)
end()
