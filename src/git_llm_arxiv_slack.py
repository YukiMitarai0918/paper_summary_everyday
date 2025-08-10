# src/arxiv_llm_slack.py
# -*- coding: utf-8 -*-
"""
arXiv → GitHub Models(LLM) で要約 → Slack投稿
- 2バケツ: 🔔最新 (LATEST_DAYS) / 🔥人気 (POPULAR_DAYS, 引用数降順)
- gpt-5 優先。権限/上限で 4.1→4o→4o-mini に自動フォールバック
- 長い Retry-After（日次上限）は待たずに切り上げ
- Orgエンドポイント対応、User-Agent付与
- キャッシュで投稿済みは再掲しない（人気は特に「毎日被らない」）
"""

import os, json, time, itertools, datetime as dt, requests, arxiv, re
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ========= 環境変数 =========
GITHUB_TOKEN      = os.environ["GITHUB_TOKEN"]
SLACK_BOT_TOKEN   = os.environ["SLACK_BOT_TOKEN"]
SLACK_CHANNEL_ID  = os.environ["SLACK_CHANNEL_ID"]
ORG_NAME          = os.getenv("GITHUB_ORG")
PREFERRED_MODEL   = os.getenv("GITHUB_MODEL", "openai/gpt-5")
OPENALEX_EMAIL    = os.getenv("OPENALEX_EMAIL", "")     # 任意（礼儀）

# ========= GitHub Models API =========
BASE_URL = f"https://models.github.ai/orgs/{ORG_NAME}/inference/chat/completions" if ORG_NAME \
           else "https://models.github.ai/inference/chat/completions"
CATALOG_URL = f"https://models.github.ai/orgs/{ORG_NAME}/catalog/models" if ORG_NAME \
              else "https://models.github.ai/catalog/models"
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "paper-summary-bot/1.0 (+github.com/yourname/yourrepo)",
    "Content-Type": "application/json",
}
CONNECT_TIMEOUT = 5
READ_TIMEOUT    = 30
MAX_RETRY_AFTER = 120  # 秒

# ========= トピック/カテゴリ =========
TOPICS = {
    "Accelerator": [
        "Transformer accelerator",
        "AI accelerator"
    ],
    "LLM": [
        "large language model",
        "LLM",
    ],
    "Memory": [
        "KV cache",
        "memory access",
    ],
}
CATEGORIES = {"cs.AI", "cs.AR", "cs.LG", "cs.CL", "cs.ET"}  # 必要なら追加: cs.DC, eess.SP, etc.

# ========= 期間・件数 =========
LATEST_DAYS             = 2     # 🔔最新は直近何日を見るか
POPULAR_DAYS            = 30    # 🔥人気は直近何日で人気を見るか
MAX_LATEST_PER_TOPIC    = 6     # 各トピックの候補プール上限（最新）
MAX_POPULAR_PER_TOPIC   = 10    # 各トピックの候補プール上限（人気）

# ========= LLM 呼び出し/キャッシュ =========
LLM_BATCH_SIZE         = 5      # 1リクエストに詰める論文数
MAX_LLM_CALLS_PER_RUN  = 2      # 1回あたり最大呼び出し回数（= 最新1 + 人気1 を想定）
CACHE_PATH             = ".summary_cache.json"  # 投稿済みIDを保持（再掲防止）

# ========= arXiv クライアント =========
arxiv_client = arxiv.Client(page_size=5, delay_seconds=3, num_retries=1)

# ========= クエリビルド =========
def build_phrase_or_and(term: str) -> str:
    term_clean = term.strip('"')
    words = term_clean.split()
    phrase = f'(ti:"{term_clean}" OR abs:"{term_clean}")'
    if len(words) <= 1:
        return phrase
    ti_and  = " AND ".join(f"ti:{w}"  for w in words)
    abs_and = " AND ".join(f"abs:{w}" for w in words)
    return f'({phrase} OR ({ti_and}) OR ({abs_and}))'

def build_topic_core_query(terms: list[str]) -> str:
    return "(" + " OR ".join(build_phrase_or_and(t) for t in terms) + ")"

def build_query_for_range(terms: list[str], start_dt: dt.datetime, end_dt: dt.datetime) -> str:
    cats_expr = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    core = build_topic_core_query(terms)
    return f"{core} AND ({cats_expr}) AND submittedDate:[{start_dt:%Y%m%d%H%M%S} TO {end_dt:%Y%m%d%H%M%S}]"

# ========= arXiv検索 =========
def search_arxiv_with_query(query: str, limit: int):
    search = arxiv.Search(
        query=query, max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate, sort_order=arxiv.SortOrder.Descending,
    )
    results, seen = [], set()
    for r in itertools.islice(arxiv_client.results(search), limit * 2):
        if not (set(r.categories) & CATEGORIES):  # 念のため後段でも確認
            continue
        if r.entry_id in seen:
            continue
        seen.add(r.entry_id)
        results.append(r)
        if len(results) >= limit:
            break
        time.sleep(0.2)
    return results

# ========= OpenAlex（引用数） =========
OPENALEX_BASE = "https://api.openalex.org/works"
OPENALEX_TIMEOUT = 10
CITATIONS_CACHE_PATH = ".citations_cache.json"

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)

CIT_CACHE = load_json(CITATIONS_CACHE_PATH, {})

def arxiv_id_from_entry_id(entry_id: str) -> str:
    return entry_id.split("/abs/")[-1]  # 2508.01234v1

def strip_version(aid: str) -> str:
    return re.sub(r"v\d+$", "", aid)    # 2508.01234

def get_citations(aid_nover: str) -> int:
    if aid_nover in CIT_CACHE:
        return int(CIT_CACHE[aid_nover])
    url = f"{OPENALEX_BASE}/https://arxiv.org/abs/{aid_nover}"
    params = {"select": "id,cited_by_count"}
    if OPENALEX_EMAIL:
        params["mailto"] = OPENALEX_EMAIL
    try:
        r = requests.get(url, params=params, timeout=OPENALEX_TIMEOUT)
        if r.status_code == 404:
            CIT_CACHE[aid_nover] = 0;  return 0
        r.raise_for_status()
        c = int(r.json().get("cited_by_count", 0))
        CIT_CACHE[aid_nover] = c
        return c
    except requests.RequestException:
        return 0

# ========= モデル候補（高→安） =========
CANDIDATES = ["openai/gpt-5", "openai/gpt-4.1", "openai/gpt-4o", "openai/gpt-4o-mini"]

def get_model_ladder(prefer_first: str | None = PREFERRED_MODEL):
    r = requests.get(CATALOG_URL, headers=HEADERS, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()
    available = {m["id"] for m in r.json()}
    ladder = [m for m in CANDIDATES if m in available]
    if prefer_first in ladder:
        ladder.remove(prefer_first); ladder = [prefer_first] + ladder
    if not ladder:
        raise RuntimeError(f"No candidate models available. Available={sorted(available)}")
    print("[INFO] Model ladder:", " > ".join(ladder))
    return ladder

# ========= LLM（要約） =========
SYSTEM = """### 指示 ###
以下の複数の論文について、それぞれ独立に要約してください。
各論文ごとに:
- タイトル(和名) を1行
- 箇条書き 最大3点（各50文字以内、日本語）

### 出力フォーマット（論文ごとに繰り返す）###
## {index}. タイトル(和名)
- 箇条1
- 箇条2
- 箇条3
"""

def call_models(model_name: str, user_prompt: str) -> str:
    payload = {"model": model_name, "messages": [
        {"role": "system", "content": "You are a helpful Japanese academic summarization assistant."},
        {"role": "user", "content": user_prompt},
    ], "temperature": 0.2}
    r = requests.post(BASE_URL, headers=HEADERS, data=json.dumps(payload),
                      timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    if r.status_code == 429:
        ra = int(r.headers.get("Retry-After", "0") or "0")
        print(f"[WARN] 429 rate limited. Retry-After={ra}s")
        if ra > MAX_RETRY_AFTER: raise RuntimeError(f"RATE_LIMIT_LONG:{ra}")
        time.sleep(ra); raise RuntimeError("RATE_LIMIT_SHORT")
    if r.status_code == 403:
        xh = {k: v for k, v in r.headers.items() if k.startswith("X-")}
        print(f"[ERROR] 403 no_access. hint={xh}"); raise PermissionError(r.text)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def batch_summarize(papers: list[dict], model_name: str) -> str:
    parts = [f"### 論文 {i}\nTitle: {p['title']}\nAbstract: {p['abstract']}\n"
             for i, p in enumerate(papers, 1)]
    prompt = f"""{SYSTEM}

以下が対象の論文です：
{chr(10).join(parts)}
"""
    for _ in range(3):
        try:
            return call_models(model_name, prompt)
        except RuntimeError as e:
            if str(e).startswith("RATE_LIMIT_SHORT"): continue
            raise
    raise RuntimeError("RETRY_EXHAUSTED")

def batch_summarize_with_fallback(papers: list[dict], ladder: list[str]) -> tuple[str, str]:
    errors = []
    for model_name in ladder:
        try:
            return batch_summarize(papers, model_name), model_name
        except PermissionError:
            errors.append((model_name, "403/no_access")); print(f"[INFO] no_access → fallback: {model_name}")
            continue
        except RuntimeError as e:
            msg = str(e); errors.append((model_name, msg))
            if msg.startswith("RATE_LIMIT_LONG") or msg == "RETRY_EXHAUSTED":
                print(f"[INFO] {msg} → fallback: {model_name}"); continue
            print(f"[INFO] error '{msg}' → fallback: {model_name}"); continue
    raise RuntimeError(f"All models failed: {errors}")

# ========= Slack =========
slack = WebClient(token=SLACK_BOT_TOKEN)
def post_to_slack(text: str):
    MAX_CHUNK = 3000
    chunks = [text[i:i+MAX_CHUNK] for i in range(0, len(text), MAX_CHUNK)] or [text]
    for c in chunks:
        try: slack.chat_postMessage(channel=SLACK_CHANNEL_ID, text=c)
        except SlackApiError as e:
            print(f"[ERROR] Slack送信失敗: {e.response.get('error')}"); raise
        time.sleep(0.4)

# ========= キャッシュ =========
def load_cache() -> set[str]:
    return set(load_json(CACHE_PATH, []))
def save_cache(done: set[str]) -> None:
    save_json(CACHE_PATH, sorted(done))

# ========= ヘルパ =========
def interleave_round_robin(grouped: dict[str, list[dict]], capacity: int) -> list[dict]:
    picked = []
    # 各トピックから最低1件
    for name in TOPICS.keys():
        lst = grouped.get(name, [])
        if lst:
            picked.append(lst.pop(0))
            if len(picked) >= capacity: return picked
    # 残りはラウンドロビン
    while len(picked) < capacity:
        moved = False
        for name in TOPICS.keys():
            lst = grouped.get(name, [])
            if lst:
                picked.append(lst.pop(0)); moved = True
                if len(picked) >= capacity: break
        if not moved: break
    return picked

# ========= メイン =========
def main():
    now = dt.datetime.today()
    done = load_cache()

    # --- 🔔 最新（直近 LATEST_DAYS 日、投稿日順） ---
    grouped_latest = {name: [] for name in TOPICS.keys()}
    latest_from = now - dt.timedelta(days=LATEST_DAYS)
    for name, terms in TOPICS.items():
        print(f"[INFO] 最新検索: {name}")
        q = build_query_for_range(terms, latest_from, now)
        res = search_arxiv_with_query(q, MAX_LATEST_PER_TOPIC)
        bucket = []
        for r in res:
            if r.entry_id in done: continue
            bucket.append({"id": r.entry_id, "title": r.title, "abstract": r.summary,
                           "date": r.published})
        # 新しい順（APIの順と同じだが一応）
        bucket.sort(key=lambda x: x["date"] or dt.datetime.min, reverse=True)
        grouped_latest[name] = bucket

    # --- 🔥 人気（直近 POPULAR_DAYS 日、引用数降順） ---
    grouped_popular = {name: [] for name in TOPICS.keys()}
    pop_from = now - dt.timedelta(days=POPULAR_DAYS)
    for name, terms in TOPICS.items():
        print(f"[INFO] 人気検索: {name}")
        q = build_query_for_range(terms, pop_from, now)
        res = search_arxiv_with_query(q, MAX_POPULAR_PER_TOPIC)
        bucket = []
        for r in res:
            if r.entry_id in done: continue
            aid_nover = strip_version(arxiv_id_from_entry_id(r.entry_id))
            cites = get_citations(aid_nover)
            bucket.append({"id": r.entry_id, "title": r.title, "abstract": r.summary,
                           "citations": cites, "date": r.published})
            time.sleep(0.1)  # OpenAlex礼儀
        # 引用数降順 → 同点は新しい方
        bucket.sort(key=lambda x: (x["citations"], x["date"] or dt.datetime.min), reverse=True)
        grouped_popular[name] = bucket

    ladder = get_model_ladder(PREFERRED_MODEL)

    calls = 0
    posted = 0

    # まず 🔔最新 を 1バッチ（最大 LLM_BATCH_SIZE 件）
    latest_items = interleave_round_robin(grouped_latest, LLM_BATCH_SIZE)
    if latest_items:
        txt, used = batch_summarize_with_fallback(latest_items, ladder)
        post_to_slack(f"[model: {used}] 🔔 最新\n" + txt + "\n\n" + "\n".join(p["id"] for p in latest_items))
        for p in latest_items: done.add(p["id"]); posted += 1
        calls += 1
        time.sleep(0.6)

    # 次に 🔥人気 を 1バッチ（残り枠があれば）
    if calls < MAX_LLM_CALLS_PER_RUN:
        popular_items = interleave_round_robin(grouped_popular, LLM_BATCH_SIZE)
        if popular_items:
            txt, used = batch_summarize_with_fallback(popular_items, ladder)
            post_to_slack(f"[model: {used}] 🔥 人気（直近 {POPULAR_DAYS}日・引用数順）\n"
                          + txt + "\n\n" + "\n".join(p["id"] for p in popular_items))
            for p in popular_items: done.add(p["id"]); posted += 1
            calls += 1
            time.sleep(0.6)

    if posted == 0:
        post_to_slack("本日は要約対象がありませんでした（新着なし/既出・権限・上限などの可能性）。")

    save_cache(done)
    save_json(CITATIONS_CACHE_PATH, CIT_CACHE)

if __name__ == "__main__":
    main()
