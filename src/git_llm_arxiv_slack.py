# src/arxiv_llm_slack.py
# -*- coding: utf-8 -*-
"""
arXiv → GitHub Models(LLM) でバッチ要約 → Slack投稿
- gpt-5 優先。権限/上限に当たったら 4.1→4o→4o-mini に自動フォールバック
- 長い Retry-After（日次上限）は待たずに切り上げ
- Orgエンドポイント対応、User-Agent付与
- キャッシュで同じ論文は再要約しない
"""

import os, json, time, itertools, datetime as dt, requests, arxiv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# ========= 環境変数 =========
GITHUB_TOKEN      = os.environ["GITHUB_TOKEN"]          # Fine-grained PAT（models:read 必須）
SLACK_BOT_TOKEN   = os.environ["SLACK_BOT_TOKEN"]       # xoxb-...
SLACK_CHANNEL_ID  = os.environ["SLACK_CHANNEL_ID"]      # C... / G...
ORG_NAME          = os.getenv("GITHUB_ORG")             # 例: "your-org"（無ければ個人文脈）
PREFERRED_MODEL   = os.getenv("GITHUB_MODEL", "openai/gpt-5")

# ========= GitHub Models API =========
BASE_URL = (
    f"https://models.github.ai/orgs/{ORG_NAME}/inference/chat/completions"
    if ORG_NAME else
    "https://models.github.ai/inference/chat/completions"
)
CATALOG_URL = (
    f"https://models.github.ai/orgs/{ORG_NAME}/catalog/models"
    if ORG_NAME else
    "https://models.github.ai/catalog/models"
)
HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "paper-summary-bot/1.0 (+github.com/yourname/yourrepo)",
    "Content-Type": "application/json",
}
CONNECT_TIMEOUT = 5
READ_TIMEOUT    = 30
MAX_RETRY_AFTER = 120  # 秒。これ超えは日次上限とみなして即フォールバック/終了

# ========= arXiv 検索条件 =========
KEYWORDS   = ["LLM", "memory", "accelerator"]   # 好みで調整
CATEGORIES = {"cs.AI", "cs.AR", "cs.LG", "cs.CL", "cs.ET"}
MAX_RESULT_PER_KEYWORD = 4
N_DAYS     = 3
QUERY_TEMPLATE = '%28 ti:"{}" OR abs:"{}" %29 AND submittedDate:[{} TO {}]'
today     = dt.datetime.today()
base_date = today - dt.timedelta(days=N_DAYS)

# ========= バッチ/上限/キャッシュ =========
LLM_BATCH_SIZE         = 5    # 1リクエストに詰める論文数
MAX_LLM_CALLS_PER_RUN  = 2    # 1回の実行での LLM 呼び出し上限
CACHE_PATH             = ".summary_cache.json"

# ========= arXiv クライアント =========
arxiv_client = arxiv.Client(
    page_size=5,
    delay_seconds=3,   # arXivの礼儀
    num_retries=1,
)

def build_query(keyword: str) -> str:
    return QUERY_TEMPLATE.format(
        keyword, keyword,
        base_date.strftime("%Y%m%d%H%M%S"),
        today.strftime("%Y%m%d%H%M%S"),
    )

def search_arxiv(keyword: str, limit: int):
    search = arxiv.Search(
        query=build_query(keyword),
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    results, seen = [], set()
    for r in itertools.islice(arxiv_client.results(search), limit * 2):
        if not (set(r.categories) & CATEGORIES):
            continue
        if r.entry_id in seen:
            continue
        seen.add(r.entry_id)
        results.append(r)
        if len(results) >= limit:
            break
        time.sleep(0.2)  # 小休止（連投防止）
    return results

# ========= モデル候補（高→安）と利用可能ラダー作成 =========
CANDIDATES = [
    "openai/gpt-5",
    "openai/gpt-4.1",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
]

def get_model_ladder(prefer_first: str | None = PREFERRED_MODEL):
    r = requests.get(CATALOG_URL, headers=HEADERS, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    r.raise_for_status()
    available = {m["id"] for m in r.json()}
    ladder = [m for m in CANDIDATES if m in available]
    if prefer_first in ladder:
        ladder.remove(prefer_first)
        ladder = [prefer_first] + ladder
    if not ladder:
        raise RuntimeError(f"No candidate models available. Available={sorted(available)}")
    print("[INFO] Model ladder:", " > ".join(ladder))
    return ladder

# ========= LLM（バッチ要約） =========
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
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful Japanese academic summarization assistant."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    r = requests.post(
        BASE_URL, headers=HEADERS, data=json.dumps(payload),
        timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    if r.status_code == 429:
        ra = int(r.headers.get("Retry-After", "0") or "0")
        print(f"[WARN] 429 rate limited. Retry-After={ra}s")
        if ra > MAX_RETRY_AFTER:
            raise RuntimeError(f"RATE_LIMIT_LONG:{ra}")
        time.sleep(ra)
        raise RuntimeError("RATE_LIMIT_SHORT")
    if r.status_code == 403:
        # SSOが必要な場合ヘッダにヒントが来る
        xh = {k: v for k, v in r.headers.items() if k.startswith("X-")}
        print(f"[ERROR] 403 no_access. hint={xh}")
        raise PermissionError(r.text)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def batch_summarize(papers: list[dict], model_name: str) -> str:
    parts = []
    for i, p in enumerate(papers, 1):
        parts.append(f"### 論文 {i}\nTitle: {p['title']}\nAbstract: {p['abstract']}\n")
    prompt = f"""{SYSTEM}

以下が対象の論文です：
{chr(10).join(parts)}
"""
    # 短い429のみリトライ
    for _ in range(3):
        try:
            return call_models(model_name, prompt)
        except RuntimeError as e:
            if str(e).startswith("RATE_LIMIT_SHORT"):
                continue
            raise
    raise RuntimeError("RETRY_EXHAUSTED")

def batch_summarize_with_fallback(papers: list[dict], ladder: list[str]) -> tuple[str, str]:
    errors = []
    for model_name in ladder:
        try:
            txt = batch_summarize(papers, model_name=model_name)
            return txt, model_name
        except PermissionError:
            errors.append((model_name, "403/no_access"))
            print(f"[INFO] no_access → fallback: {model_name}")
            continue
        except RuntimeError as e:
            msg = str(e)
            errors.append((model_name, msg))
            # 長いRetry-After（=日次上限）は即フォールバック
            if msg.startswith("RATE_LIMIT_LONG") or msg == "RETRY_EXHAUSTED":
                print(f"[INFO] {msg} → fallback: {model_name}")
                continue
            # その他の例外も次候補へ
            print(f"[INFO] error '{msg}' → fallback: {model_name}")
            continue
    raise RuntimeError(f"All models failed: {errors}")

# ========= Slack =========
slack = WebClient(token=SLACK_BOT_TOKEN)

def post_to_slack(text: str):
    MAX_CHUNK = 3000
    chunks = [text[i:i+MAX_CHUNK] for i in range(0, len(text), MAX_CHUNK)] or [text]
    for c in chunks:
        try:
            slack.chat_postMessage(channel=SLACK_CHANNEL_ID, text=c)
        except SlackApiError as e:
            print(f"[ERROR] Slack送信失敗: {e.response.get('error')}")
            raise
        time.sleep(0.4)

# ========= キャッシュ =========
def load_cache() -> set[str]:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_cache(done: set[str]) -> None:
    tmp = f"{CACHE_PATH}.tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done), f, ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# ========= メイン =========
def main():
    # 1) arXiv収集
    all_res = []
    for kw in KEYWORDS:
        print(f"[INFO] 検索中: {kw}")
        all_res.extend(search_arxiv(kw, MAX_RESULT_PER_KEYWORD))

    # 2) 未処理だけ
    done = load_cache()
    items = []
    for r in all_res:
        if r.entry_id in done:
            continue
        items.append({"id": r.entry_id, "title": r.title, "abstract": r.summary})

    if not items:
        post_to_slack("新規の対象論文はありませんでした。")
        return

    # 3) 利用可能モデルのラダー作成（gpt-5優先）
    ladder = get_model_ladder(PREFERRED_MODEL)

    # 4) バッチ要約（フォールバック＆上限）
    calls = 0
    posted = 0
    for batch in chunks(items, LLM_BATCH_SIZE):
        if calls >= MAX_LLM_CALLS_PER_RUN:
            break
        try:
            summary_md, used_model = batch_summarize_with_fallback(batch, ladder)
        except RuntimeError as e:
            post_to_slack(f"【要約失敗】候補モデル全滅: {str(e)[:180]}")
            break

        header = f"[model: {used_model}]"
        body = summary_md + "\n\n" + "\n".join(p["id"] for p in batch)
        post_to_slack(header + "\n" + body)

        for p in batch:
            done.add(p["id"])
            posted += 1

        calls += 1
        time.sleep(0.6)

    if posted == 0:
        post_to_slack("本日の要約は作成できませんでした（権限/上限/エラーの可能性）。")

    save_cache(done)

if __name__ == "__main__":
    main()