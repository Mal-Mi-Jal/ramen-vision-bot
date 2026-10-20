"""벤치마크(Benchmark): 평가 결과에 측정 조건을 기록하고, 두 결과를 사진 단위로 비교한다.

- 측정 조건(env): Ollama가 0.34.4 -> 0.40.3으로 자동 업데이트된 것만으로 예측이 바뀐 적이 있다.
  조건을 결과 파일에 남겨야 "내 수정 때문인지, 환경 때문인지"를 구분할 수 있다.
- 비교(compare): 전체 정답률이 같아도 사진별로는 고쳐진 것과 새로 틀린 것이 섞여 있을 수 있다.
"""
import json
import statistics
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------- 측정 조건 ----------

def _ollama_version() -> str:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=5) as r:
            return json.load(r)["version"]
    except OSError:
        return "unknown"


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def collect_env() -> dict:
    """지금 측정 환경. 평가 결과 summary["env"]에 저장한다"""
    import ollama

    from ramen_bot import judge, knowledge, vision

    digests = {m.model: m.digest[:12] for m in ollama.list().models}
    models = {vision.MODEL, knowledge.EMBED_MODEL}
    return {
        "ollama": _ollama_version(),
        # 같은 이름이라도 다시 받으면 내용이 바뀔 수 있어서 해시(digest)를 남긴다
        "models": {m: digests.get(m if ":" in m else f"{m}:latest", "missing") for m in sorted(models)},
        "max_side": vision.MAX_SIDE,
        "num_ctx": vision.NUM_CTX,
        "top_k": judge.TOP_K,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        # 커밋하지 않은 수정(프롬프트·문서 등)이 있었는지. True면 git_commit만으로는 재현할 수 없다
        # results/는 평가가 만드는 파일이라 뺀다 (앞 세트의 결과 파일 때문에 다음 세트가 dirty로 찍혔다)
        "git_dirty": _git("status", "--porcelain", "--", ".", ":!results") not in ("", "unknown"),
    }


# ---------- 시간 ----------

def timing(rows: list[dict]) -> dict:
    seconds = [r["seconds"] for r in rows]
    slowest = max(rows, key=lambda r: r["seconds"])
    return {"mean": round(statistics.mean(seconds), 1), "median": round(statistics.median(seconds), 1),
            "max": slowest["seconds"], "max_file": slowest["file"]}


# ---------- 비교 ----------

def classify(old_row: dict, new_row: dict) -> str | None:
    """같은 사진의 이전/새 결과 -> 변화 종류.
    "fixed"   : 틀렸다가 맞힘 (X -> O)
    "broken"  : 맞혔다가 틀림 (O -> X)
    "changed" : 둘 다 틀렸는데 틀린 답이 바뀜 (예: ieke -> shio)
    None      : 변화 없음"""
    old_ok, new_ok = old_row["correct"], new_row["correct"]  # True / False
    if not old_ok and new_ok:
        return "fixed"
    if old_ok and not new_ok:
        return "broken"
    if (not old_ok and not new_ok) and old_row["style"] != new_row["style"]:
        return "changed"
    return None


def compare(old: dict, new: dict) -> dict:
    """결과 JSON 두 개({"summary", "rows"}) -> 차이"""
    old_rows = {r["file"]: r for r in old["rows"]}
    new_rows = {r["file"]: r for r in new["rows"]}
    common = sorted(old_rows.keys() & new_rows.keys())

    changes = {"fixed": [], "broken": [], "changed": []}
    retrieval = {"gained": [], "lost": []}
    for f in common:
        o, n = old_rows[f], new_rows[f]
        kind = classify(o, n)
        if kind:
            changes[kind].append({"file": f, "answer": n["answer"], "old": o["style"], "new": n["style"]})
        if o["retrieval_hit"] != n["retrieval_hit"]:
            retrieval["gained" if n["retrieval_hit"] else "lost"].append(f)

    # 공정한 비교가 아닐 수 있는 이유들
    warnings = []
    old_env, new_env = old["summary"].get("env"), new["summary"].get("env")
    if not old_env or not new_env:
        warnings.append("측정 조건(env) 기록이 없는 결과가 있음 -> Ollama 버전 등이 같은지 알 수 없음")
    else:
        for key in sorted(old_env.keys() | new_env.keys()):
            if key in ("git_commit", "git_dirty"):
                continue  # 코드가 바뀐 건 비교하려는 대상 자체
            if old_env.get(key) != new_env.get(key):
                warnings.append(f"측정 조건이 다름: {key} {old_env.get(key)} -> {new_env.get(key)}")
    if old["summary"]["samples"] != new["summary"]["samples"]:
        warnings.append(f"사진 폴더가 다름: {old['summary']['samples']} -> {new['summary']['samples']}")
    only_old, only_new = sorted(old_rows.keys() - new_rows.keys()), sorted(new_rows.keys() - old_rows.keys())
    if only_old or only_new:
        warnings.append(f"사진 목록이 다름: 이전에만 {only_old}, 새 결과에만 {only_new}")

    def score(rows: dict) -> dict:
        return {"correct": sum(rows[f]["correct"] for f in common),
                "hits": sum(rows[f]["retrieval_hit"] for f in common), "total": len(common)}

    return {"old": score(old_rows), "new": score(new_rows), "changes": changes, "retrieval": retrieval,
            "timing": {"old": timing(old["rows"]), "new": timing(new["rows"])}, "warnings": warnings}


def format_report(diff: dict) -> str:
    o, n, total = diff["old"], diff["new"], diff["old"]["total"]
    lines = [f"공통 사진 {total}장 기준",
             f"  정답률     {o['correct']}/{total} -> {n['correct']}/{total} ({n['correct'] - o['correct']:+d}장)",
             f"  검색 재현율 {o['hits']}/{total} -> {n['hits']}/{total} ({n['hits'] - o['hits']:+d}장)"]
    labels = {"fixed": "고쳐짐 X->O", "broken": "새로 틀림 O->X", "changed": "틀린 답이 바뀜"}
    for kind, label in labels.items():
        items = diff["changes"][kind]
        lines.append(f"\n{label}: {len(items)}장")
        lines += [f"  {c['file']:22} 정답={c['answer']:11} {c['old']} -> {c['new']}" for c in items]
    for kind, label in (("gained", "검색에 새로 잡힘"), ("lost", "검색에서 빠짐")):
        if diff["retrieval"][kind]:
            lines.append(f"\n{label}: {', '.join(diff['retrieval'][kind])}")
    to, tn = diff["timing"]["old"], diff["timing"]["new"]
    lines.append(f"\n시간(초) 평균 {to['mean']} -> {tn['mean']}, 중앙값 {to['median']} -> {tn['median']}, "
                 f"최대 {to['max']}({to['max_file']}) -> {tn['max']}({tn['max_file']})")
    if diff["warnings"]:
        lines.append("\n⚠ 공정한 비교가 아닐 수 있음:")
        lines += [f"  - {w}" for w in diff["warnings"]]
    return "\n".join(lines)


# ---------- 관문(gate) 평가 ----------

# 통과해야 정답인 카테고리. 다른 면 요리는 일부러 통과시킨다(진짜 라멘을 거절하는 것보다 낫다, 실험 10)
EXPECT_PASS = {"ramen", "other_noodle"}


def gate_passed(gate: str) -> bool:
    """is_noodle_dish 결과 -> 파이프라인으로 넘어갔나. invalid(응답 깨짐)도 거절하지 않고 넘긴다"""
    return gate != "no"


def gate_summary(rows: list[dict]) -> dict:
    """rows: [{"file", "category", "gate"}] -> 카테고리별 통과 수와 두 종류의 실수"""
    per_category = {}
    for r in rows:
        c = per_category.setdefault(r["category"], {"passed": 0, "total": 0})
        c["passed"] += gate_passed(r["gate"])
        c["total"] += 1
    reject_rows = [r for r in rows if r["category"] not in EXPECT_PASS]
    pass_rows = [r for r in rows if r["category"] in EXPECT_PASS]
    # 잘못 통과(false accept): 거절해야 하는데 통과시킨 사진
    false_accept = [r["file"] for r in reject_rows if gate_passed(r["gate"])]
    # 잘못 거절(false reject): 통과해야 하는데 거절한 사진
    false_reject = [r["file"] for r in pass_rows if not gate_passed(r["gate"])]
    return {"per_category": per_category,
            "false_accept": false_accept, "false_accept_rate": len(false_accept) / len(reject_rows) if reject_rows else 0,
            "false_reject": false_reject, "false_reject_rate": len(false_reject) / len(pass_rows) if pass_rows else 0}


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
