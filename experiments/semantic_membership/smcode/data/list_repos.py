"""Списки репозиториев и их детерминированное распределение по ролям (DESIGN.md §2.1).

Основные функции:
- load_repos_yaml / save_repos_yaml — чтение и запись configs/repos.yaml;
- split_repos(cfg, repos_yaml) — детерминированный (seed) выбор protected / hard_neg из
  protected_pool со стратификацией по языку (каждый язык присутствует в обеих ролях) и
  разбиение public по репозиториям 0.6/0.2/0.2 (стратификация по языку);
  результат сохраняется в data/functions/repo_splits.json вместе с seed, n_protected, долями и
  sha1 списков; при повторном вызове читается оттуда, а при несовпадении с текущим конфигом/YAML
  поднимается RuntimeError (force=True — пересчитать);
- extend_public(...) — необязательное расширение списка public через api.github.com
  (сетевой вызов; тестами никогда не вызывается, есть --dry-run).

CLI (используется scripts/00_clone.sh):
    python -m smcode.data.list_repos --config configs/default.yaml --print-splits [--force]
печатает строки вида "<split> <owner>/<name>".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from smcode.config import ROOT, get_rng, load_config, resolve_path
from smcode.normalize import canon_lang
from smcode.types import SPLITS

log = logging.getLogger(__name__)

DEFAULT_REPOS_YAML = ROOT / "configs" / "repos.yaml"
REPO_SPLITS_FILE = "repo_splits.json"
REPO_SPLITS_VERSION = 2
PUBLIC_SPLITS = ("public_train", "public_calib", "public_test")
GITHUB_LANG = {"python": "Python", "c": "C", "cpp": "C++", "go": "Go", "java": "Java", "javascript": "JavaScript"}

_WORD_RE = re.compile(r"[a-z0-9]+")


# ----------------------------------------------------------------------------- yaml


def load_repos_yaml(path: str | Path | None = None) -> dict[str, Any]:
    """Читает configs/repos.yaml и нормализует записи (repo, lang)."""
    path = Path(path) if path else DEFAULT_REPOS_YAML
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for key in ("protected_pool", "public"):
        data[key] = [normalize_entry(e) for e in data.get(key) or []]
    data.setdefault("domain_keywords", [])
    data.setdefault("n_protected", 12)
    return data


def save_repos_yaml(data: dict[str, Any], path: str | Path | None = None) -> Path:
    """Сохраняет списки репозиториев в YAML (ключи в фиксированном порядке)."""
    path = Path(path) if path else DEFAULT_REPOS_YAML
    out = {
        "n_protected": int(data.get("n_protected", 12)),
        "protected_pool": [dict(e) for e in data.get("protected_pool", [])],
        "public": [dict(e) for e in data.get("public", [])],
        "domain_keywords": list(data.get("domain_keywords", [])),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False, default_flow_style=None)
    return path


def normalize_entry(e: Any) -> dict[str, Any]:
    """Приводит запись к виду {repo: "owner/name", lang: <canon>, ...}; ValueError при ошибке."""
    if isinstance(e, str):
        e = {"repo": e}
    if not isinstance(e, dict) or "repo" not in e:
        raise ValueError(f"bad repo entry: {e!r}")
    out = dict(e)
    repo = str(out["repo"]).strip().strip("/")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(f"repo must be 'owner/name': {repo!r}")
    out["repo"] = repo
    if out.get("lang"):
        out["lang"] = canon_lang(str(out["lang"]))
    return out


def repo_dir_name(repo: str) -> str:
    """'owner/name' -> 'owner__name' (имя каталога клона в data/raw/<split>/)."""
    owner, name = repo.split("/", 1)
    return f"{owner}__{name}"


def repo_from_dir_name(dirname: str) -> str:
    """'owner__name' -> 'owner/name'."""
    if "__" not in dirname:
        return dirname
    owner, name = dirname.split("__", 1)
    return f"{owner}/{name}"


def repo_url(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def repos_digest(data: dict[str, Any]) -> str:
    """sha1 нормализованного содержимого repos.yaml (пул, public, ключевые слова, n_protected)."""
    payload = {
        "n_protected": int(data.get("n_protected") or 12),
        "protected_pool": sorted((e["repo"].lower(), e.get("lang") or "") for e in data.get("protected_pool", [])),
        "public": sorted((e["repo"].lower(), e.get("lang") or "") for e in data.get("public", [])),
        "domain_keywords": sorted(str(k).lower() for k in data.get("domain_keywords") or []),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- domain filter


def matches_domain(text: str, keywords: Iterable[str]) -> str | None:
    """Возвращает первое ключевое слово домена, найденное в тексте (по словам; слова длиной >= 4 —
    и как подстроки слов), иначе None."""
    words = set(_WORD_RE.findall(text.lower()))
    for kw in keywords:
        k = str(kw).lower().strip()
        if not k:
            continue
        if k in words:
            return k
        if len(k) >= 4 and any(k in w for w in words):
            return k
    return None


def entry_text(e: dict[str, Any]) -> str:
    parts = [e.get("repo", ""), str(e.get("description") or ""), " ".join(e.get("topics") or [])]
    return " ".join(parts)


# ----------------------------------------------------------------------------- split


def _stratified_public_split(
    public: list[dict[str, Any]], fractions: dict[str, float], rng
) -> dict[str, list[dict[str, Any]]]:
    """Разбиение public по репозиториям с перемешиванием внутри каждого языка."""
    total = sum(fractions.values()) or 1.0
    f_train = fractions.get("public_train", 0.6) / total
    f_calib = fractions.get("public_calib", 0.2) / total
    out: dict[str, list[dict[str, Any]]] = {s: [] for s in PUBLIC_SPLITS}
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for e in public:
        by_lang.setdefault(e.get("lang") or "unknown", []).append(e)
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang], key=lambda x: x["repo"])
        rng.shuffle(items)
        n = len(items)
        b1 = int(round(n * f_train))
        b2 = min(n, b1 + int(round(n * f_calib)))
        if n >= 3:  # гарантируем непустые calib/test при достаточном числе репозиториев
            b1 = min(b1, n - 2)
            b2 = max(min(b2, n - 1), b1 + 1)
        out["public_train"].extend(items[:b1])
        out["public_calib"].extend(items[b1:b2])
        out["public_test"].extend(items[b2:])
    for s in PUBLIC_SPLITS:
        out[s].sort(key=lambda x: x["repo"])
    return out


def protected_quotas(counts: dict[str, int], n_protected: int) -> dict[str, int]:
    """Квоты protected по языкам: пропорционально размеру пула, но >= 1 на язык (если язык
    представлен) и <= n_lang - 1 (чтобы язык остался и в hard_neg; при n_lang == 1 — весь в protected).
    Сумма квот == min(n_protected, |pool|). Детерминированно (ключи сортируются)."""
    langs = sorted(counts)
    total = sum(counts.values())
    n_protected = max(0, min(n_protected, total))
    if not langs or n_protected == 0:
        return {lang: 0 for lang in langs}
    cap = {lang: max(1, counts[lang] - 1) for lang in langs}
    q = {lang: max(1, min(cap[lang], round(n_protected * counts[lang] / total))) for lang in langs}
    # приводим сумму к n_protected, сохраняя ограничения 1 <= q <= cap
    while sum(q.values()) > n_protected:
        cand = [lang for lang in langs if q[lang] > 1]
        if not cand:  # n_protected < число языков: часть языков останется только в hard_neg
            victim = max(langs, key=lambda l: (q[l], l))
            q[victim] -= 1
            continue
        victim = max(cand, key=lambda l: (q[l], counts[l], l))
        q[victim] -= 1
    while sum(q.values()) < n_protected:
        cand = [lang for lang in langs if q[lang] < cap[lang]]
        if not cand:
            cand = [lang for lang in langs if q[lang] < counts[lang]]
            if not cand:
                break
        target = max(cand, key=lambda l: (counts[l] - q[l], l))
        q[target] += 1
    return q


def _stratified_pool_split(pool: list[dict[str, Any]], n_protected: int, rng) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """protected / hard_neg из protected_pool: квоты по языкам (protected_quotas), внутри языка —
    перемешивание seed-ГПСЧ."""
    by_lang: dict[str, list[dict[str, Any]]] = {}
    for e in pool:
        by_lang.setdefault(e.get("lang") or "unknown", []).append(e)
    quotas = protected_quotas({lang: len(v) for lang, v in by_lang.items()}, n_protected)
    protected: list[dict[str, Any]] = []
    hard_neg: list[dict[str, Any]] = []
    for lang in sorted(by_lang):
        items = sorted(by_lang[lang], key=lambda x: x["repo"])
        rng.shuffle(items)
        q = quotas[lang]
        protected.extend(items[:q])
        hard_neg.extend(items[q:])
        if q == 0 or q == len(items):
            log.warning("язык %s: %d репозиториев в пуле — представлен только в %s", lang, len(items),
                        "hard_neg" if q == 0 else "protected")
    protected.sort(key=lambda x: x["repo"])
    hard_neg.sort(key=lambda x: x["repo"])
    return protected, hard_neg


def _split_params(cfg: dict[str, Any], data: dict[str, Any]) -> tuple[int, dict[str, float]]:
    n_protected = int(cfg.get("splits", {}).get("n_protected") or data.get("n_protected") or 12)
    fractions = {s: float(cfg.get("splits", {}).get(s, d)) for s, d in zip(PUBLIC_SPLITS, (0.6, 0.2, 0.2))}
    return n_protected, fractions


def _stale_reasons(saved: dict[str, Any], cfg: dict[str, Any], n_protected: int, fractions: dict[str, float], digest: str) -> list[str]:
    """Расхождения сохранённого repo_splits.json с текущим конфигом и repos.yaml."""
    reasons: list[str] = []
    if saved.get("seed") != cfg.get("seed"):
        reasons.append(f"seed {saved.get('seed')!r} -> {cfg.get('seed')!r}")
    if int(saved.get("n_protected") or 0) != n_protected:
        reasons.append(f"n_protected {saved.get('n_protected')} -> {n_protected}")
    sf = {k: float(v) for k, v in (saved.get("fractions") or {}).items()}
    if any(abs(sf.get(k, -1.0) - v) > 1e-9 for k, v in fractions.items()):
        reasons.append(f"fractions {sf} -> {fractions}")
    if saved.get("repos_sha1") != digest:
        reasons.append(f"repos.yaml sha1 {saved.get('repos_sha1')} -> {digest}")
    if int(saved.get("version") or 1) != REPO_SPLITS_VERSION:
        reasons.append(f"version {saved.get('version')} -> {REPO_SPLITS_VERSION}")
    return reasons


def split_repos(
    cfg: dict[str, Any],
    repos_yaml: dict[str, Any] | str | Path | None = None,
    persist: bool = True,
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Распределение репозиториев по сплитам (детерминировано по cfg.seed).

    protected_pool -> n_protected репозиториев `protected` (квоты по языкам: каждый язык есть и в
    protected, и в hard_neg), остальные `hard_neg`; public -> public_train/public_calib/public_test
    по долям cfg.splits (по репозиториям, со стратификацией по языку); репозитории public,
    совпадающие с protected_pool или с ключевыми словами домена, исключаются.
    При persist результат пишется в data/functions/repo_splits.json (seed, n_protected, fractions,
    repos_sha1, splits). Если файл есть и не force: при совпадении параметров возвращается сохранённое
    распределение, при расхождении — RuntimeError (пересчёт меняет роли уже склонированных
    репозиториев; нужен явный --force и повторное клонирование).
    """
    data = repos_yaml if isinstance(repos_yaml, dict) else load_repos_yaml(repos_yaml)
    n_protected, fractions = _split_params(cfg, data)
    digest = repos_digest(data)
    out_path = resolve_path(cfg, "functions") / REPO_SPLITS_FILE if persist else None
    if out_path is not None and out_path.exists() and not force:
        with open(out_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        reasons = _stale_reasons(saved, cfg, n_protected, fractions, digest)
        if reasons:
            msg = (f"{out_path} устарел: " + "; ".join(reasons) +
                   ". Пересчёт меняет роли репозиториев: запустите с --force (и повторите клонирование).")
            log.error(msg)
            raise RuntimeError(msg)
        log.info("repo_splits.json найден (%s), параметры совпадают — используем сохранённое распределение", out_path)
        return {s: list(saved["splits"].get(s, [])) for s in SPLITS}

    pool = sorted((normalize_entry(e) for e in data.get("protected_pool", [])), key=lambda x: x["repo"])
    public_raw = [normalize_entry(e) for e in data.get("public", [])]
    keywords = list(data.get("domain_keywords") or [])

    rng = get_rng(cfg, "split_repos")
    protected, hard_neg = _stratified_pool_split(pool, n_protected, rng)

    pool_names = {e["repo"].lower() for e in pool}
    public: list[dict[str, Any]] = []
    seen: set[str] = set()
    for e in public_raw:
        key = e["repo"].lower()
        if key in seen:
            continue
        seen.add(key)
        if key in pool_names:
            log.warning("public repo %s совпадает с protected_pool — исключён", e["repo"])
            continue
        kw = matches_domain(entry_text(e), keywords)
        if kw:
            log.warning("public repo %s исключён по ключевому слову домена %r", e["repo"], kw)
            continue
        public.append(e)
    pub_splits = _stratified_public_split(public, fractions, rng)

    result: dict[str, list[dict[str, Any]]] = {"protected": protected, "hard_neg": hard_neg, **pub_splits}
    for s in SPLITS:
        langs = sorted({e.get("lang") or "?" for e in result[s]})
        log.info("split %-13s: %d репозиториев, языки: %s", s, len(result[s]), ",".join(langs))
    if out_path is not None:
        payload = {
            "version": REPO_SPLITS_VERSION,
            "seed": cfg.get("seed"),
            "n_protected": n_protected,
            "fractions": fractions,
            "repos_sha1": digest,
            "splits": result,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        log.info("распределение сохранено: %s", out_path)
    return result


def load_repo_splits(cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]] | None:
    """Читает data/functions/repo_splits.json (None, если файла нет)."""
    p = resolve_path(cfg, "functions") / REPO_SPLITS_FILE
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return {s: list(v) for s, v in json.load(f)["splits"].items()}


def iter_split_repo_lines(splits: dict[str, list[dict[str, Any]]]) -> Iterable[str]:
    """Строки "<split> <owner>/<name>" для scripts/00_clone.sh."""
    for s in SPLITS:
        for e in splits.get(s, []):
            yield f"{s} {e['repo']}"


# ----------------------------------------------------------------------------- github (сеть; не для тестов)


def github_search(query: str, token: str | None = None, per_page: int = 100, page: int = 1) -> list[dict[str, Any]]:
    """Один запрос к api.github.com/search/repositories (сортировка по звёздам). Сетевой вызов."""
    import urllib.parse
    import urllib.request

    url = "https://api.github.com/search/repositories?" + urllib.parse.urlencode(
        {"q": query, "sort": "stars", "order": "desc", "per_page": per_page, "page": page}
    )
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "smcode-list-repos"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    return list(payload.get("items", []))


def extend_public(
    data: dict[str, Any],
    cfg: dict[str, Any],
    per_lang: int = 30,
    min_stars: int = 1000,
    max_stars: int = 60000,
    pages: int = 2,
    token: str | None = None,
    dry_run: bool = True,
    repos_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Дополняет data['public'] репозиториями из поиска GitHub (по языкам cfg.languages),
    исключая домен protected/hard_neg и уже известные репозитории. Возвращает новые записи;
    при dry_run=False дописывает их в YAML (после этого repo_splits.json нужно пересчитать с --force)."""
    known = {e["repo"].lower() for e in data.get("public", [])} | {e["repo"].lower() for e in data.get("protected_pool", [])}
    keywords = list(data.get("domain_keywords") or [])
    added: list[dict[str, Any]] = []
    for lang in cfg.get("languages", list(GITHUB_LANG)):
        gh_lang = GITHUB_LANG.get(lang, lang)
        got = 0
        for page in range(1, pages + 1):
            q = f"language:{gh_lang} stars:{min_stars}..{max_stars} archived:false fork:false"
            try:
                items = github_search(q, token=token, page=page)
            except Exception as exc:  # noqa: BLE001 — сетевые ошибки логируем и идём дальше
                log.error("github search failed (%s, page %d): %s", lang, page, exc)
                break
            for it in items:
                repo = it.get("full_name", "")
                if not repo or repo.lower() in known:
                    continue
                entry = {
                    "repo": repo,
                    "lang": lang,
                    "description": (it.get("description") or "")[:200],
                    "topics": list(it.get("topics") or []),
                    "stars": int(it.get("stargazers_count") or 0),
                }
                kw = matches_domain(entry_text(entry), keywords)
                if kw:
                    log.debug("skip %s (domain keyword %r)", repo, kw)
                    continue
                known.add(repo.lower())
                added.append({"repo": repo, "lang": lang, "stars": entry["stars"]})
                got += 1
                if got >= per_lang:
                    break
            if got >= per_lang or len(items) < 100:
                break
        log.info("%s: добавлено %d репозиториев", lang, got)
    if added and not dry_run:
        data.setdefault("public", []).extend(added)
        save_repos_yaml(data, repos_path)
        log.info("YAML обновлён: %s", repos_path or DEFAULT_REPOS_YAML)
    return added


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Списки репозиториев и распределение по сплитам")
    ap.add_argument("--config", default=None, help="путь к default.yaml")
    ap.add_argument("--repos", default=None, help="путь к repos.yaml")
    ap.add_argument("--print-splits", action="store_true", help="печатать строки '<split> <owner>/<name>'")
    ap.add_argument("--force", action="store_true", help="пересчитать repo_splits.json")
    ap.add_argument("--no-persist", action="store_true", help="не сохранять repo_splits.json")
    ap.add_argument("--extend", action="store_true", help="расширить public через api.github.com (сеть)")
    ap.add_argument("--dry-run", action="store_true", help="с --extend: только показать, не писать YAML")
    ap.add_argument("--per-lang", type=int, default=30)
    ap.add_argument("--min-stars", type=int, default=1000)
    ap.add_argument("--max-stars", type=int, default=60000)
    ap.add_argument("--token", default=None, help="GitHub token (или переменная GITHUB_TOKEN)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    data = load_repos_yaml(args.repos)
    if args.extend:
        import os

        token = args.token or os.environ.get("GITHUB_TOKEN")
        added = extend_public(
            data, cfg, per_lang=args.per_lang, min_stars=args.min_stars, max_stars=args.max_stars,
            token=token, dry_run=args.dry_run, repos_path=args.repos,
        )
        for e in added:
            print(f"{e['lang']} {e['repo']} stars={e.get('stars', '?')}")
        if args.dry_run:
            log.info("dry-run: %d кандидатов, YAML не изменён", len(added))
    try:
        splits = split_repos(cfg, data, persist=not args.no_persist, force=args.force)
    except RuntimeError as exc:
        log.error("%s", exc)
        return 3
    if args.print_splits:
        for line in iter_split_repo_lines(splits):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
