# ruff: noqa
# %%
import ast
import hashlib
import math
import os
import re
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostRegressor
from PIL import Image
from requests.adapters import HTTPAdapter, Retry
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm.auto import tqdm


def stage(name):
    print(f"\n[Stage Start] {name}")
    start = time.time()
    return lambda: print(f"[Stage End] {name} ✅ ({time.time() - start:.1f}s)\n")


# %%
end = stage("Step 1: # ------------------")
# ------------------
# 0) ライブラリのインポート
# ------------------
warnings.filterwarnings("ignore")
pl.seed_everything(42)
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
end()

# %%
end = stage("Step 2: # ------------------")
# ------------------
# 1) 各種変数
# ------------------
DATA_DIR = "/kaggle/input/data-science-osaka-autumn-2025"
FAST_DEV_RUN = True  # 本番実行では False
MULTI_SEED = False
TOUGH_WAIT = False  # True→od_wait=300/400, False→50
GPU_ON = True
USE_GROUP_KFOLD = True  # True なら GroupKFold(RecipeId) を採用
N_SPLITS = 5  # CV 分割数
RANDOM_SEED = 42

IMG_SIZE = 128  # 96→128で表現力UP
MAX_IMG_PER_SAMPLE = 3
end()

# %%
end = stage("Step 3: # ------------------")
# ------------------
# 2) Load data
# ------------------
df_train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["DatePublished"], index_col=0)
df_test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["DatePublished"], index_col=0)

if FAST_DEV_RUN:
    df_train = df_train.sample(min(len(df_train), 500), random_state=71)
    df_test = df_test.sample(min(len(df_test), 500), random_state=71)
    print("[stage] sampled data", df_train.shape, df_test.shape)
end()

# %%
end = stage("Step 4: # ------------------")


# ------------------
# 3) Utilities
# ------------------
def to_list(x):
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


def duration_to_minutes(s: str) -> int:
    if pd.isna(s):
        return 0
    s = str(s)
    h = re.search(r"(\d+)H", s)
    m = re.search(r"(\d+)M", s)
    return (int(h.group(1)) if h else 0) * 60 + (int(m.group(1)) if m else 0)


def make_stratified_bins(values, n_bins=10):
    series = pd.Series(values)
    uniq = series.nunique()
    bins = min(n_bins, uniq if uniq > 1 else 1)
    if bins <= 1:
        return np.zeros(len(series), dtype=int)
    q = pd.qcut(series, q=bins, labels=False, duplicates="drop")
    return q.fillna(0).astype(int).to_numpy()


end()

# %%
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
        [
            "parsley",
            "cilantro",
            "basil",
            "mint",
            "chives",
            "dill",
            "oregano",
            "thyme",
            "rosemary",
            "sage",
            "coriander",
            "coriander",
            "leaves",
            "coriander",
            "leaf",
            "scallion",
            "scallions",
            "green",
            "onion",
            "green",
            "onions",
            "spring",
            "onion",
            "spring",
            "onions",
            "salt",
            "pepper",
            "black",
            "pepper",
            "white",
            "pepper",
            "paprika",
            "chili",
            "powder",
            "cayenne",
            "cumin",
            "turmeric",
            "cinnamon",
            "nutmeg",
            "allspice",
            "clove",
            "cloves",
            "garnish",
            "lemon",
            "zest",
            "lemon",
            "rind",
            "lemon",
            "peel",
            "orange",
            "zest",
            "orange",
            "rind",
            "orange",
            "peel",
            "zest",
            "rind",
            "peel",
        ],
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
        pair_text(p, q)
        for p, q in zip(df["IngredientParts_List"], df["IngredientQuantities_List"], strict=False)
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

# %%
end = stage("Step 6: # ------------------")


# ------------------
# 5) LoRA MobileNetV3 + OOF 特徴
# ------------------
# ============================================================
# ✅ cache_all_images: 保存時に強制リサイズ＆黒画像フォールバック
# ============================================================
def cache_all_images(df, output_dir="/tmp/images", max_imgs=1, img_size=128, timeout=4):
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()
    retry = Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=64, pool_connections=64)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    def resolve_urls(val):
        if isinstance(val, list):
            return val[:max_imgs]
        if isinstance(val, str):
            if val.startswith("["):
                try:
                    L = eval(val)
                    if isinstance(L, list):
                        return L[:max_imgs]
                except Exception:
                    return [val]
            return [val]
        return []

    urls = []
    for v in df["Images"]:
        urls.extend(resolve_urls(v))

    def fname_for(url):
        return os.path.join(output_dir, hashlib.md5(url.encode()).hexdigest() + ".jpg")

    def write_black(fpath):
        black = np.zeros((img_size, img_size, 3), np.uint8)
        cv2.imwrite(fpath, black, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def download_one(url):
        fpath = fname_for(url)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 4096:
            return fpath
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            # ✅ 強制リサイズ
            img = img.resize((img_size, img_size), Image.BILINEAR)
            img.save(fpath, "JPEG", quality=90)
        except Exception:
            write_black(fpath)
        return fpath

    print(f"[cache] downloading {len(urls)} images → {img_size}×{img_size} resized")
    with ThreadPoolExecutor(max_workers=32) as ex:
        for _ in tqdm(as_completed([ex.submit(download_one, u) for u in urls]), total=len(urls)):
            pass
    print("[cache] done.")


# ============================================================
# 2) LoRALinear + LoRAMobileNetV3（省略部は同じ）
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=16, dropout=0.05, bias=True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / float(r)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.B(self.dropout(self.A(x))) * self.scale


class LoRAMobileNetV3(nn.Module):
    def __init__(self, r=4, alpha=16, dropout=0.05):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self._inject(base, r, alpha, dropout)
        base.classifier = nn.Identity()
        self.backbone = base
        self.emb_dim = 576
        self.fc_calorie = nn.Linear(self.emb_dim, 1)
        self.fc_rating = nn.Linear(self.emb_dim, 1)

    def _inject(self, module, r, alpha, dropout):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                new = LoRALinear(child.in_features, child.out_features, r, alpha, dropout)
                new.base.weight.data.copy_(child.weight)
                if child.bias is not None:
                    new.base.bias.data.copy_(child.bias)
                setattr(module, name, new)
            else:
                self._inject(child, r, alpha, dropout)

    def forward(self, x):
        f = self.backbone(x)
        return self.fc_calorie(f).squeeze(1), self.fc_rating(f).squeeze(1)


# ============================================================
# ✅ Dataset: ロード時にも再リサイズ保険を追加
# ============================================================
class LocalImageDataset(Dataset):
    def __init__(self, df, transform, labels=True, img_size=128, cache_dir="/tmp/images"):
        self.df = df.copy()
        self.transform = transform
        self.labels = labels
        self.img_size = img_size
        self.cache_dir = cache_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        urls = row["Images"]
        if isinstance(urls, str):
            try:
                urls = ast.literal_eval(urls)
            except Exception:
                urls = [urls]
            else:
                if not isinstance(urls, list):
                    urls = [urls]
        if not isinstance(urls, list) or len(urls) == 0:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        else:
            fpath = os.path.join(self.cache_dir, hashlib.md5(urls[0].encode()).hexdigest() + ".jpg")
            img = cv2.imread(fpath)
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # ✅ 再リサイズ保険
                if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
                    img = cv2.resize(img, (self.img_size, self.img_size))
        x = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        x = self.transform(x).unsqueeze(0)
        if self.labels:
            y1 = torch.tensor(float(row["Calories"]), dtype=torch.float32)
            y2 = torch.tensor(float(row["RecipeRating"]), dtype=torch.float32)
            return x, y1, y2
        else:
            return x


def collate_variable_images(batch):
    if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 3:
        xs, yc, yr = [], [], []
        for x, y1, y2 in batch:
            xs.append(x.unsqueeze(0))
            yc.append(y1)
            yr.append(y2)
        return torch.cat(xs), torch.stack(yc), torch.stack(yr)
    else:
        xs = [x.unsqueeze(0) for x in batch]
        return torch.cat(xs)


# ============================================================
# 4) build_lora_features_fast_url（キャッシュ時リサイズ対応）
# ============================================================
def build_lora_features_fast_url(
    df_train,
    df_test,
    epochs=50,
    batch_size=2,
    lr=1e-4,
    precision="16-mixed",
    patience=3,
    cache_dir="/tmp/images",
    img_size=128,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # キャッシュ作成時にresize
    cache_all_images(df_train, cache_dir, max_imgs=1, img_size=img_size)
    cache_all_images(df_test, cache_dir, max_imgs=1, img_size=img_size)

    transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    val_frac = 0.1
    n_val = max(1, int(len(df_train) * val_frac))
    df_val = df_train.sample(n_val, random_state=42)
    df_trn = df_train.drop(df_val.index)

    train_dl = DataLoader(
        LocalImageDataset(df_trn, transform, True, cache_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_variable_images,
    )
    val_dl = DataLoader(
        LocalImageDataset(df_val, transform, True, cache_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_variable_images,
    )
    test_dl = DataLoader(
        LocalImageDataset(df_test, transform, False, cache_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_variable_images,
    )

    model = LoRAMobileNetV3(r=4, alpha=16, dropout=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=("16" in precision))
    best_loss = float("inf")
    best_state = None
    no_imp = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0
        for xb, yc, yr in tqdm(train_dl, desc=f"[Train]{epoch}/{epochs}"):
            xb, yc, yr = xb.to(device), yc.to(device), yr.to(device)
            B, N, C, H, W = xb.shape
            xb_flat = xb.view(B * N, C, H, W)
            with torch.cuda.amp.autocast(enabled=("16" in precision)):
                pc, pr = model(xb_flat)
                loss = F.mse_loss(pc.mean().expand_as(yc), yc) + F.mse_loss(pr.mean().expand_as(yr), yr)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
            del xb, yc, yr, xb_flat, pc, pr, loss
            torch.cuda.empty_cache()
        # validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yc, yr in val_dl:
                xb, yc, yr = xb.to(device), yc.to(device), yr.to(device)
                B, N, C, H, W = xb.shape
                xb_flat = xb.view(B * N, C, H, W)
                with torch.cuda.amp.autocast(enabled=("16" in precision)):
                    pc, pr = model(xb_flat)
                    loss = F.mse_loss(pc.mean().expand_as(yc), yc) + F.mse_loss(pr.mean().expand_as(yr), yr)
                val_loss += loss.item()
                del xb, yc, yr, xb_flat, pc, pr, loss
                torch.cuda.empty_cache()
        val_loss /= max(1, len(val_dl))
        print(f"[Epoch {epoch:03d}] val={val_loss:.4f}")
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"⏹️ Early stop at {epoch}, best val={best_loss:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    @torch.no_grad()
    def extract(df, loader):
        model.eval()
        out = np.zeros((len(df), 8), dtype=np.float32)
        ptr = 0
        for batch in tqdm(loader, desc="[Extract]"):
            xb = batch[0] if isinstance(batch, (list, tuple)) else batch
            xb = xb.to(device)
            if xb.ndim == 5:
                B, N, C, H, W = xb.shape
                xb_flat = xb.view(B * N, C, H, W)
            else:
                B, C, H, W = xb.shape
                xb_flat = xb
                N = 1
            with torch.cuda.amp.autocast(enabled=("16" in precision)):
                cal, rat = model(xb_flat)
            cal = cal.view(B, N)
            rat = rat.view(B, N)
            feats = (
                torch.stack(
                    [
                        cal.mean(1),
                        cal.std(1),
                        cal.max(1).values,
                        cal.min(1).values,
                        rat.mean(1),
                        rat.std(1),
                        rat.max(1).values,
                        rat.min(1).values,
                    ],
                    dim=1,
                )
                .cpu()
                .numpy()
            )
            out[ptr : ptr + B] = feats
            ptr += B
            del xb, xb_flat, cal, rat, feats
            torch.cuda.empty_cache()
        cols = ["cal_mean", "cal_std", "cal_max", "cal_min", "rat_mean", "rat_std", "rat_max", "rat_min"]
        return pd.DataFrame(out, columns=cols, index=df.index)

    tr_feats = extract(
        df_train,
        DataLoader(
            LocalImageDataset(df_train, transform, True, cache_dir),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
            collate_fn=collate_variable_images,
        ),
    )
    te_feats = extract(df_test, test_dl)
    return tr_feats, te_feats


end()

# %%
end = stage("Step 7: # ------------------")
# ------------------
# 6) review 側：Rating OOF を作ってから RecipeId 集約 (高速版)
# ------------------
print("[review] TF-IDF+SVD開始...")

# -------------------------------
# 1. データ読み込み + メタ特徴
# -------------------------------
df_review_train = pd.read_csv(f"{DATA_DIR}/train_review.csv", index_col=0)
df_review_test = pd.read_csv(f"{DATA_DIR}/test_review.csv", index_col=0)


def enrich_review_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = df["Review"].fillna("")
    df["Review_char_len"] = text.str.len().astype(np.int32)
    df["Review_word_count"] = text.str.split().apply(len).astype(np.int32)
    df["Review_line_count"] = text.apply(lambda s: s.count("\n") + 1 if len(s) > 0 else 0).astype(np.int16)
    df["Review_exclaim_count"] = text.str.count("!").astype(np.int16)
    df["Review_question_count"] = text.str.count("?").astype(np.int16)
    return df


df_review_train = enrich_review_meta(df_review_train)
df_review_test = enrich_review_meta(df_review_test)

# -------------------------------
# 2. TF-IDF + SVD（1回fitで高速化）
# -------------------------------
vec = TfidfVectorizer(max_features=8000, min_df=5, ngram_range=(1, 2), sublinear_tf=True)
svd = TruncatedSVD(n_components=32, random_state=71)

X_all = pd.concat([df_review_train["Review"], df_review_test["Review"]], axis=0).fillna("")
X_tfidf = vec.fit_transform(X_all)
X_svd = svd.fit_transform(X_tfidf)

n_train = len(df_review_train)
Xtr_svd, Xte_svd = X_svd[:n_train], X_svd[n_train:]

tfidf_cols = [f"Review_SVD_{i}" for i in range(32)]
df_review_train_fe = pd.concat(
    [df_review_train.reset_index(drop=True), pd.DataFrame(Xtr_svd, columns=tfidf_cols)], axis=1
)
df_review_test_fe = pd.concat(
    [df_review_test.reset_index(drop=True), pd.DataFrame(Xte_svd, columns=tfidf_cols)], axis=1
)

print("[review] TF-IDF+SVD完了. CatBoost準備開始...")

# -------------------------------
# 3. 学習データ準備
# -------------------------------
y_rev_raw = df_review_train_fe["Rating"].astype(float)
y_rev = y_rev_raw.fillna(y_rev_raw.median())
X_rev_tr = df_review_train_fe.drop(columns=["Review", "Rating"], errors="ignore")
X_rev_te = df_review_test_fe.drop(columns=["Review"], errors="ignore")

# -------------------------------
# 4. カテゴリ列を安全に検出＆変換
# -------------------------------
cat_cols_rev = [
    c
    for c in X_rev_tr.columns
    if str(X_rev_tr[c].dtype) in ("object", "category") or X_rev_tr[c].dtype == bool
]

for df_ in (X_rev_tr, X_rev_te):
    for c in cat_cols_rev:
        df_[c] = df_[c].astype(str).fillna("missing")

for df_ in (X_rev_tr, X_rev_te):
    for c in df_.columns:
        if c not in cat_cols_rev:
            df_[c] = pd.to_numeric(df_[c], errors="coerce").fillna(0.0)

X_rev_tr = X_rev_tr.fillna(0)
X_rev_te = X_rev_te.fillna(0)

# -------------------------------
# 5. 列整合性を完全保証
# -------------------------------
missing_cols = set(X_rev_tr.columns) - set(X_rev_te.columns)
for col in missing_cols:
    X_rev_te[col] = 0
X_rev_te = X_rev_te[X_rev_tr.columns]

# -------------------------------
# 6. StratifiedKFold準備
# -------------------------------
bins_rev = make_stratified_bins(y_rev, n_bins=10)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

# -------------------------------
# 7. CatBoost 設定
# -------------------------------
base_params = dict(
    iterations=20000 if not FAST_DEV_RUN else 200,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5.0,
    loss_function="RMSE",
    od_type="Iter",
    od_wait=100,
    random_seed=RANDOM_SEED,
    verbose=500,
    task_type="GPU" if (GPU_ON and torch.cuda.is_available()) else "CPU",
)

# -------------------------------
# 8. 学習＋OOF生成
# -------------------------------
oof_rev = np.zeros(len(X_rev_tr), dtype=np.float32)
pred_rev = np.zeros(len(X_rev_te), dtype=np.float32)

print("[review] CatBoost学習開始...")
for fold, (tr_idx, va_idx) in enumerate(cv.split(X_rev_tr, bins_rev), 1):
    print(f"[review] Fold {fold}/{cv.n_splits} ...")
    model = CatBoostRegressor(**base_params)
    model.fit(
        X_rev_tr.iloc[tr_idx],
        y_rev.iloc[tr_idx],
        eval_set=(X_rev_tr.iloc[va_idx], y_rev.iloc[va_idx]),
        cat_features=cat_cols_rev,
        use_best_model=True,
    )
    oof_rev[va_idx] = model.predict(X_rev_tr.iloc[va_idx])
    pred_rev += model.predict(X_rev_te) / cv.n_splits

# -------------------------------
# 9. 出力 + RecipeId集約
# -------------------------------
df_review_train["Rating_oof"] = np.clip(oof_rev, 0, 5)
df_review_test["Rating_oof"] = np.clip(pred_rev, 0, 5)
df_review_train_fe["Rating_oof"] = df_review_train["Rating_oof"].values
df_review_test_fe["Rating_oof"] = df_review_test["Rating_oof"].values


def build_review_stats(df: pd.DataFrame, id_col: str = "RecipeId", prefix: str = "review") -> pd.DataFrame:
    if id_col not in df.columns:
        return pd.DataFrame(columns=[id_col])
    grouped = df.groupby(id_col)
    agg_map = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == id_col:
            continue
        if col.startswith("Review_SVD_"):
            agg_map[col] = ["mean"]  # SVDは平均のみで軽量化
        else:
            agg_map[col] = ["mean", "median", "std", "min", "max"]
    if not agg_map:
        stats = grouped.size().to_frame(name=f"{prefix}_count")
    else:
        stats = grouped.agg(agg_map)
        stats.columns = [f"{prefix}_{col}_{stat}" for col, stat in stats.columns]
        stats[f"{prefix}_count"] = grouped.size()
    stats = stats.reset_index()
    for col in stats.columns:
        if col == id_col:
            continue
        if col.endswith("_count"):
            stats[col] = stats[col].fillna(0).astype(np.int16)
        else:
            stats[col] = stats[col].fillna(0.0).astype(np.float32)
    return stats


df_train_review = build_review_stats(df_review_train_fe)
df_test_review = build_review_stats(df_review_test_fe)

if {"review_Rating_mean", "review_Rating_oof_mean"}.issubset(df_train_review.columns):
    df_train_review["review_rating_gap_mean"] = (
        df_train_review["review_Rating_mean"] - df_train_review["review_Rating_oof_mean"]
    ).astype(np.float32)
if {"review_Rating_median", "review_Rating_oof_median"}.issubset(df_train_review.columns):
    df_train_review["review_rating_gap_median"] = (
        df_train_review["review_Rating_median"] - df_train_review["review_Rating_oof_median"]
    ).astype(np.float32)

df_train = df_train.merge(df_train_review, on="RecipeId", how="left")
df_test = df_test.merge(df_test_review, on="RecipeId", how="left")

for df_ in (df_train, df_test):
    review_cols = [c for c in df_.columns if c.startswith("review_")]
    for c in review_cols:
        if df_[c].dtype.kind == "f":
            df_[c] = df_[c].fillna(0.0).astype(np.float32)
        elif df_[c].dtype.kind in {"i", "u"}:
            df_[c] = df_[c].fillna(0).astype(np.int32)
    if "review_count" in df_.columns:
        df_["review_has_data"] = df_["review_count"].fillna(0).gt(0).astype(np.int8)

print("[review] 完了: Rating OOF生成＆RecipeId集約済み ✅")
end()

# %%
end = stage("Step 8: # ------------------")
# ------------------
# 7) LoRA 画像特徴（OOF）を作成して追加
# ------------------
ora_oof_tr, lora_te = build_lora_features_fast_url(
    df_train,
    df_test,
    epochs=50,
    batch_size=2,
    lr=1e-4,
    precision="16-mixed",
    patience=3,
    cache_dir="/tmp/images",
    img_size=128,  # 🔽 ←ここでキャッシュ時のリサイズ指定
)

df_train = pd.concat([df_train, lora_oof_tr], axis=1)
df_test = pd.concat([df_test, lora_te], axis=1)
end()

# %%
end = stage("Step 9: # ------------------")


# ------------------
# 8) テキスト → TF-IDF+SVD（縮小版）
# ------------------
def join_list_column(col):
    def convert(v):
        if isinstance(v, list):
            return " ".join(map(str, v))
        if isinstance(v, str) and v.startswith("["):
            try:
                p = ast.literal_eval(v)
                if isinstance(p, list):
                    return " ".join(map(str, p))
            except Exception:
                return v
        if pd.isna(v):
            return ""
        return str(v)

    return col.apply(convert)


def tfidf_svd(train_series, test_series, prefix, n_components=50, max_features=10000, min_df=3):
    """
    テキスト列を TF-IDF + SVD で次元圧縮し、DataFrameを返す。
    """
    vec = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=(1, 2), sublinear_tf=True)
    svd = TruncatedSVD(n_components=n_components, random_state=71)

    X_train = vec.fit_transform(train_series.fillna(""))
    X_test = vec.transform(test_series.fillna(""))

    X_train_svd = svd.fit_transform(X_train)
    X_test_svd = svd.transform(X_test)

    cols = [f"{prefix}_SVD_{i}" for i in range(n_components)]
    return (
        pd.DataFrame(X_train_svd, columns=cols, index=train_series.index),
        pd.DataFrame(X_test_svd, columns=cols, index=test_series.index),
    )


# build text fields
for split in (df_train, df_test):
    split["NameText"] = split["Name"].fillna("").astype(str)
    split["RecipeCategoryText"] = split["RecipeCategory"].fillna("").astype(str)

text_columns = [
    "Description",
    "RecipeInstructions",
    "IngredientsText",
    "IngredientPairText",
    "NameText",
    "RecipeCategoryText",
]
for col in text_columns:
    df_train[col] = join_list_column(df_train[col]).fillna("")
    df_test[col] = join_list_column(df_test[col]).fillna("")


def tfidf_block(colname, n_components, max_features, min_df):
    tr, te = tfidf_svd(
        df_train[colname],
        df_test[colname],
        colname,
        n_components=n_components,
        max_features=max_features,
        min_df=min_df,
    )
    return tr, te


desc_tr, desc_te = tfidf_block("Description", 60, 15000, 5)
instr_tr, instr_te = tfidf_block("RecipeInstructions", 80, 18000, 5)
ing_tr, ing_te = tfidf_block("IngredientsText", 60, 15000, 5)
pair_tr, pair_te = tfidf_block("IngredientPairText", 60, 15000, 5)
name_tr, name_te = tfidf_block("NameText", 32, 8000, 3)
catT_tr, catT_te = tfidf_block("RecipeCategoryText", 24, 6000, 3)

# concat text blocks
df_train = pd.concat([df_train, desc_tr, instr_tr, ing_tr, pair_tr, name_tr, catT_tr], axis=1)
df_test = pd.concat([df_test, desc_te, instr_te, ing_te, pair_te, name_te, catT_te], axis=1)
end()

# %%
end = stage("Step 10: # ------------------")
# ------------------
# 9) 型安全な列アライン
# ------------------
# まず列集合を合わせ、欠損埋めを dtype ごとに実施
common_cols = sorted(set(df_train.columns) | set(df_test.columns))
df_train = df_train.reindex(columns=common_cols)
df_test = df_test.reindex(columns=common_cols)

# object列と数値列で埋め分け
obj_cols = [c for c in df_train.columns if df_train[c].dtype == "object"]
num_cols = [c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c].dtype)]

df_train[obj_cols] = df_train[obj_cols].fillna("missing")
df_test[obj_cols] = df_test[obj_cols].fillna("missing")
df_train[num_cols] = df_train[num_cols].fillna(0.0)
df_test[num_cols] = df_test[num_cols].fillna(0.0)
end()

# %%
end = stage("Step 11: # ------------------")
# ------------------
# 10) 学習セット準備
# ------------------
target_cols = ["RecipeRating", "Calories"]
y_train = df_train[target_cols].copy()
X_train = df_train.drop(columns=target_cols)
X_test = df_test.drop(columns=target_cols)

# list型カラムはCatBoostで扱えないので除去
list_columns = [c for c in X_train.columns if X_train[c].apply(lambda x: isinstance(x, list)).any()]
if len(list_columns) > 0:
    X_train = X_train.drop(columns=list_columns)
    X_test = X_test.drop(columns=list_columns)

# カテゴリ列
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]


# ==========================
# 日付列をfloat(UNIX秒)に変換
# ==========================
def datetime_to_unix(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        dtype_str = str(df[c].dtype)

        # datetime64[ns, UTC] や datetime64[ns] に対応
        if "datetime64" in dtype_str:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True).view("int64") / 1e9

        # object列に日付文字列が混じっている場合も対応
        elif df[c].dtype == object:
            try:
                parsed = pd.to_datetime(df[c], errors="coerce", utc=True)
                if parsed.notna().sum() > 0:
                    df[c] = parsed.view("int64") / 1e9
            except Exception:
                pass

    return df


X_train = datetime_to_unix(X_train.copy())
X_test = datetime_to_unix(X_test.copy())
end()

# %%
end = stage("Step 12: # ------------------")


# ------------------
# 11) CatBoost 学習（損失最適化＋ES）
# ------------------
def normalized_gini(actual, pred):
    def gini(a, p):
        all_data = np.asarray(np.c_[a, p, np.arange(len(a))], dtype=float)
        all_data = all_data[np.lexsort((all_data[:, 2], -1 * all_data[:, 1]))]
        total = all_data[:, 0].sum()
        if total == 0:
            return 0.0
        cum = np.cumsum(all_data[:, 0])
        gsum = cum.sum() / total
        gsum -= (len(a) + 1) / 2.0
        return gsum / len(a)

    denom = gini(actual, actual)
    return 0.0 if denom == 0 else gini(actual, pred) / denom


SEEDS = [42] if not MULTI_SEED else [42, 2023, 2024]
n_folds = N_SPLITS
wait = 400 if TOUGH_WAIT else 50

target_configs = {
    "RecipeRating": {
        "index": 0,
        "n_bins": 20,
        "params": {
            "iterations": 30000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 6,
            "bagging_temperature": 1.0,
            "random_strength": 0.8,
            "loss_function": "Quantile:alpha=0.5",
            "od_type": "Iter",
            "od_wait": wait,
        },
    },
    "Calories": {
        "index": 1,
        "n_bins": 25,
        "params": {
            "iterations": 30000,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 7,
            "bagging_temperature": 1.0,
            "random_strength": 0.6,
            "loss_function": "MAE",
            "od_type": "Iter",
            "od_wait": wait,
        },
    },
}
if FAST_DEV_RUN:
    for cfg in target_configs.values():
        cfg["params"] = cfg["params"].copy()
        cfg["params"]["iterations"] = min(cfg["params"]["iterations"], 200)
        cfg["params"]["od_wait"] = min(cfg["params"].get("od_wait", 300), 20)

y_pred_train = np.zeros((len(X_train), 2), dtype=np.float32)
y_pred_counts = np.zeros((len(X_train), 2), dtype=np.float32)
y_pred_test = np.zeros((len(X_test), 2), dtype=np.float32)
fold_records = []

# GPU デバイス設定（複数GPUでも安全）
if GPU_ON and torch.cuda.is_available():
    device_ids = list(range(torch.cuda.device_count()))
    catboost_devices = ",".join(map(str, device_ids)) if device_ids else None
else:
    catboost_devices = None

# CV split
if USE_GROUP_KFOLD and ("RecipeId" in X_train.columns):
    groups = X_train["RecipeId"].astype(str).fillna("missing")
    splitter = GroupKFold(n_splits=n_folds)
    split_iter = lambda y, bins: splitter.split(X_train, y, groups=groups)
else:
    split_iter = lambda y, bins: StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED
    ).split(X_train, bins)

for target_name, cfg in target_configs.items():
    target_idx = cfg["index"]
    bins = make_stratified_bins(y_train[target_name], n_bins=cfg["n_bins"])
    for seed in SEEDS:
        for fold, (tr_idx, va_idx) in enumerate(split_iter(y_train[target_name], bins)):
            print(target_name, seed, fold)
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            params = cfg["params"].copy()
            params.update({"cat_features": cat_cols, "random_seed": seed, "verbose": 200})
            model = CatBoostRegressor(
                **params,
                task_type="GPU" if (GPU_ON and torch.cuda.is_available()) else "CPU",
                devices=catboost_devices,
            )
            model.fit(X_tr, y_tr[target_name], eval_set=(X_va, y_va[target_name]), use_best_model=True)
            val_pred = model.predict(X_va)
            y_pred_train[va_idx, target_idx] += val_pred
            y_pred_counts[va_idx, target_idx] += 1
            test_pred = model.predict(X_test)
            y_pred_test[:, target_idx] += test_pred / (n_folds * len(SEEDS))
            fold_records.append(
                {
                    "target": target_name,
                    "seed": seed,
                    "fold": fold,
                    "gini": normalized_gini(y_va[target_name].values, val_pred),
                }
            )

# 平均化
counts_safe = np.where(y_pred_counts == 0, 1.0, y_pred_counts)
y_pred_train = y_pred_train / counts_safe

# Rating を [0,5] にクリップ
y_pred_test[:, 0] = np.clip(y_pred_test[:, 0], 0, 5)

# OOF指標
scores_df = pd.DataFrame(fold_records)
rating_oof = normalized_gini(y_train["RecipeRating"].values, y_pred_train[:, 0])
calories_oof = normalized_gini(y_train["Calories"].values, y_pred_train[:, 1])
oof_mean = (rating_oof + calories_oof) / 2.0

print("Fold-level Normalized Gini:")
for tgt in target_configs:
    ts = scores_df[scores_df["target"] == tgt]["gini"]
    if len(ts) > 0:
        print(f"  {tgt}: {ts.mean():.4f} ± {ts.std():.4f}")
print("\nOOF Normalized Gini:")
print(f"  RecipeRating: {rating_oof:.4f}")
print(f"  Calories:     {calories_oof:.4f}")
print(f"  Mean:         {oof_mean:.4f}")
end()

# %%
end = stage("Step 13: # ------------------")
# ------------------
# 12) Submission
# ------------------
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", index_col=0)
sub["RecipeRating_and_Calories"] = [str(list(row)) for row in y_pred_test]
sub.to_csv("submission.csv")
print(sub.head())
end()

# %%
end = stage("Step 14: # BEST Ave:0.8550 LB:0.63873")
# BEST Ave:0.8550 LB:0.63873

# base Ave:0.3626 LB:0.29625
# Timeを実数 Ave:0.3687 LB:0.30734
# n_fold 5->10 Ave:0.3660 LB:?
# LGBM->CatBoost Ave:0.3458 LB:0.27067
# CatBoostにy_valを設定 Ave:0.3755 LB:0.32870
# Keywordsをone-hotに変更 Ave:0.4016 LB:0.35015
# 材料と数量をペアにする Ave:0.4092 LB:?
# カテゴリーデータ変換なし Ave:0.4221 LB:0.41243
# review情報追加(Rating無し) Ave:0.5662 LB:52133
# Rating予測追加 Ave:0.8365 LB:0.62281
# ReviewsをTF-IDF化 Ave:0.8408 LB:0.62507
# reviewの日時変換方法を変更 Ave:0.8402 LB:0.60415
# dfの日時変換方法を変更 Ave:0.8399 LB:0.62576
# kfoldをStratifiedKFoldに変更 Ave:0.8427 LB:0.62511
# reviewのRating算出ハイパラを修正 Ave:0.8427 LB:0.63040
# dfのCaloriesのscore算出ハイパラを修正 Ave0.8420: LB:0.63138
# dfのRecipeRatingのscore算出ハイパラを修正 Ave:0.8348 LB:0.62772
# Rating算出前に日時変換 Ave:0.8421 LB:0.61226
# 色々追加 Ave:0.8550 LB:0.63873
# 色々追加light Ave:0.8497 LB:0.63351
# reviewのRatingを整数化 Ave:0.8493 LB:0.63146
# reviewのRatingをclip、RecipeCategoryをone-hot、Rating回帰waitを300 Ave:0.8456 LB:0.63561
# RecipeRatingをclip&round Ave:0.8460 LB:0.54100
# RecipeRatingをclip Ave:0.8463 LB:0.63798
# カテゴリ列をフルに戻す Ave:0.8475 LB:0.63907
# waitを300に変更 Ave:0.8511 LB:0.64333
# RecipeIngredientPartsのone-hotを追加 Ave:0.8542 LB:0.64821
# LoRA追加して色々カイゼン Ave: LB:
end()
