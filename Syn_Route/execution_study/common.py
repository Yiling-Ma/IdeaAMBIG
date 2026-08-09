from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen


REPO_URL = "https://github.com/NoviScl/AI-Researcher.git"
PAPER_URL = "https://openreview.net/forum?id=Fllp8l6Puy"
DATA_URL = "https://drive.google.com/file/d/1PpxeTz_-xHHcMXyUwv1Oed1avTaiD5vv/view?usp=sharing"
DATA_FILE_ID = "1PpxeTz_-xHHcMXyUwv1Oed1avTaiD5vv"

LEVEL2_TO_LEVEL1 = {
    "Ambiguous Definition": "Ambiguity",
    "Ambiguous Procedure": "Ambiguity",
    "Missing Method Procedure": "Incompleteness",
    "Missing Configuration Protocol": "Incompleteness",
    "Missing Model Structure": "Incompleteness",
    "Missing Evaluation Specification": "Incompleteness",
    "Missing Data Specification": "Incompleteness",
    "Conflicting Objective": "Inconsistency",
    "Conflicting Model Design": "Inconsistency",
    "Conflicting Formal Definition": "Inconsistency",
}

VALID_LEVEL2 = set(LEVEL2_TO_LEVEL1.keys())

VALID_SPECIFICATION_SLOTS = {
    "TASK_AND_IO",
    "CORE_ALGORITHM",
    "MODEL_ARCHITECTURE",
    "OBJECTIVE_AND_SUPERVISION",
    "TRAINING_PROCEDURE",
    "DATA_AND_PREPROCESSING",
    "INFERENCE_AND_DECISION",
    "EVALUATION_PROTOCOL",
    "INTERNAL_CONSISTENCY",
    "NONE",
}

LEVEL2_DEFAULT_SPECIFICATION_SLOT = {
    "Ambiguous Definition": "CORE_ALGORITHM",
    "Ambiguous Procedure": "CORE_ALGORITHM",
    "Missing Method Procedure": "CORE_ALGORITHM",
    "Missing Model Structure": "MODEL_ARCHITECTURE",
    "Missing Data Specification": "DATA_AND_PREPROCESSING",
    "Missing Configuration Protocol": "TRAINING_PROCEDURE",
    "Missing Evaluation Specification": "EVALUATION_PROTOCOL",
    "Conflicting Objective": "INTERNAL_CONSISTENCY",
    "Conflicting Model Design": "INTERNAL_CONSISTENCY",
    "Conflicting Formal Definition": "INTERNAL_CONSISTENCY",
}


def normalize_level2(label: Any) -> str:
    text = str(label or "").strip()
    for valid in VALID_LEVEL2:
        if text.lower() == valid.lower():
            return valid
    return text


def normalize_specification_slot(slot: Any) -> str:
    text = str(slot or "").strip().upper().replace(" ", "_").replace("-", "_")
    text = re.sub(r"_+", "_", text)
    if text in VALID_SPECIFICATION_SLOTS:
        return text
    return str(slot or "").strip()


def infer_codification_slot_from_level2(level2: Any) -> str:
    normalized = normalize_level2(level2)
    return LEVEL2_DEFAULT_SPECIFICATION_SLOT.get(normalized, "NONE")

DIAGNOSTIC_LEAKAGE_PHRASES = [
    "missing",
    "unspecified",
    "not provided",
    "the removed detail",
    "gold",
    "defect",
    "benchmark",
    "synthetic defect",
    "corrupted",
    "omitted",
    "not specified",
    "not described",
    "not defined",
    "lacks",
    "gap",
]

WEAK_DETAIL_PATTERNS = [
    r"\bbatch size\b",
    r"\blearning rate\b",
    r"\brandom seed\b",
    r"\bseed\b",
    r"\bepochs?\b",
    r"\bweight decay\b",
    r"\bhardware\b",
    r"\bgpu\b",
    r"\bcuda\b",
]

TEXT_SUFFIXES = {".txt", ".md", ".json", ".jsonl", ".csv", ".tex", ".rst"}
CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".json", ".ipynb"}
PAPER_SUFFIXES = {".pdf", ".md", ".txt", ".tex"}


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_path", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(row: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: Any, n: int = 12) -> str:
    text = "::".join(str(p or "") for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def safe_name(value: Any, max_len: int = 140) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_len] or "unknown")


def read_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            return extract_pdf_text(path, max_chars=max_chars)
        if path.suffix.lower() == ".docx":
            return extract_docx_text(path, max_chars=max_chars)
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text[:max_chars]
    except Exception:
        return ""

def extract_docx_text(path: Path, max_chars: int = 200_000) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        xml = re.sub(r"</w:p>", "\n", xml)
        text = re.sub(r"<[^>]+>", " ", xml)
        text = (
            text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", '"')
            .replace("&apos;", "'")
        )
        return re.sub(r"[ \t\r\f\v]+", " ", text)[:max_chars]
    except Exception:
        return ""


def extract_pdf_text(path: Path, max_chars: int = 200_000) -> str:
    if shutil.which("pdftotext"):
        proc = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if proc.stdout.strip():
            return proc.stdout[:max_chars]
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path))
        chunks = []
        for page in reader.pages[:30]:
            chunks.append(page.extract_text() or "")
            if sum(len(c) for c in chunks) >= max_chars:
                break
        return "\n".join(chunks)[:max_chars]
    except Exception:
        return ""


def list_files(root: Path, max_files: int = 50_000) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file():
            files.append(path)
            if len(files) >= max_files:
                break
    return sorted(files)


def file_inventory(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for path in list_files(root):
        try:
            stat = path.stat()
        except OSError:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "suffix": path.suffix.lower(),
                "type_hint": infer_file_type(path),
            }
        )
    return rows


def infer_file_type(path: Path) -> str:
    name = path.name.lower()
    parts = [p.lower() for p in path.parts]
    suffix = path.suffix.lower()
    if suffix == ".pdf" or "paper" in name or "papers" in parts:
        return "paper"
    if "review" in name or "reviews" in parts:
        return "review"
    if "idea" in name or "ideas" in parts or "proposal" in name:
        return "idea"
    if suffix in CODE_SUFFIXES or any(p in parts for p in ["code", "codebase", "src", "scripts"]):
        return "code"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix == ".zip":
        return "archive"
    return "other"


def summarize_inventory(root: Path) -> Dict[str, Any]:
    files = file_inventory(root)
    by_type = Counter(f["type_hint"] for f in files)
    by_suffix = Counter(f["suffix"] or "<none>" for f in files)
    project_dirs = []
    for path in root.iterdir() if root.exists() else []:
        if path.is_dir() and path.name != ".git":
            count = len(list_files(path, max_files=500))
            project_dirs.append({"path": path.name, "file_count_sample": count})
    return {
        "root": str(root),
        "num_files": len(files),
        "total_size_bytes": sum(f["size_bytes"] for f in files),
        "file_type_counts": dict(by_type),
        "suffix_counts": dict(by_suffix),
        "possible_project_level_directories": project_dirs,
        "data_files": files,
    }


def git_commit(repo_dir: Path) -> Optional[str]:
    if not (repo_dir / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc.stdout.strip() or None


def clone_repo(repo_dir: Path, resume: bool = False) -> Dict[str, Any]:
    if repo_dir.exists() and (repo_dir / ".git").exists() and resume:
        return {"status": "reused", "path": str(repo_dir), "commit": git_commit(repo_dir)}
    if repo_dir.exists() and not resume:
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            return {"status": "failed", "stderr": proc.stderr}
    return {"status": "ok", "path": str(repo_dir), "commit": git_commit(repo_dir)}


def try_download_google_drive(file_id: str, out_path: Path) -> Dict[str, Any]:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 1024:
        return {"status": "exists", "path": str(out_path), "size_bytes": out_path.stat().st_size}
    if shutil.which("gdown"):
        proc = subprocess.run(
            ["gdown", file_id, "-O", str(out_path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
            return {"status": "ok", "method": "gdown-cli", "path": str(out_path), "size_bytes": out_path.stat().st_size}
    try:
        import gdown  # type: ignore

        gdown.download(id=file_id, output=str(out_path), quiet=True)
        if out_path.exists() and out_path.stat().st_size > 1024:
            return {"status": "ok", "method": "gdown-python", "path": str(out_path), "size_bytes": out_path.stat().st_size}
    except Exception:
        pass
    return {
        "status": "manual_download_required",
        "download_url": DATA_URL,
        "file_id": file_id,
        "place_file_at": str(out_path),
        "continue_command": "python3 step_00_download_dataset.py --out_dir outputs_step0_raw --resume",
    }


def unzip_if_needed(zip_path: Path, extract_dir: Path, resume: bool = False) -> Dict[str, Any]:
    if not zip_path.exists():
        return {"status": "missing_zip", "zip_path": str(zip_path)}
    if extract_dir.exists() and resume and list(extract_dir.iterdir()):
        return {"status": "reused", "path": str(extract_dir)}
    if extract_dir.exists() and not resume:
        shutil.rmtree(extract_dir)
    ensure_dir(extract_dir)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        return {"status": "ok", "path": str(extract_dir)}
    except zipfile.BadZipFile:
        return {"status": "bad_zip", "zip_path": str(zip_path)}


def columnar_reviews_to_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if not isinstance(obj, dict):
        return []
    if all(isinstance(v, list) for v in obj.values()):
        keys = list(obj)
        n = max((len(obj[k]) for k in keys), default=0)
        rows = []
        for i in range(n):
            row = {}
            for k in keys:
                values = obj.get(k, [])
                row[k] = values[i] if i < len(values) else None
            rows.append(row)
        return rows
    rows = []
    for key, value in obj.items():
        if isinstance(value, dict):
            row = {"_record_key": key}
            row.update(value)
            rows.append(row)
    return rows


def group_reviews_by_project(repo_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    review_path = repo_dir / "reviews_execution" / "data_points_all_execution.json"
    rows = columnar_reviews_to_rows(load_json(review_path, {}))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        idea_id = str(row.get("idea_id") or row.get("project_id") or row.get("_record_key") or "").strip()
        if idea_id:
            grouped[idea_id].append(row)
    return dict(grouped)


def compact_reviews_text(rows: List[Dict[str, Any]], max_chars: int = 20_000) -> str:
    chunks = []
    fields = [
        "overall_rationale",
        "novelty_rationale",
        "excitement_rationale",
        "soundness_rationale",
        "effectiveness_rationale",
        "faithfulness_rationale",
    ]
    for i, row in enumerate(rows[:8], 1):
        parts = [f"Review {i}"]
        for score in ["overall_score", "novelty_score", "soundness_score", "effectiveness_score", "faithfulness_score"]:
            if row.get(score) not in (None, ""):
                parts.append(f"{score}: {row.get(score)}")
        for field in fields:
            value = str(row.get(field) or "").strip()
            if value:
                parts.append(f"{field}: {value}")
        chunks.append("\n".join(parts))
    return "\n\n".join(chunks)[:max_chars]


def find_extracted_data_root(raw_dir: Path) -> Optional[Path]:
    candidates = [p for p in raw_dir.iterdir() if p.is_dir() and p.name not in {"AI-Researcher", "__MACOSX"}]
    if len(candidates) == 1:
        root = candidates[0]
        nested = [p for p in root.iterdir() if p.is_dir()]
        if len(nested) == 1 and len(list_files(root, max_files=20)) == len(list_files(nested[0], max_files=20)):
            return nested[0]
        return root
    for candidate in candidates:
        names = " ".join(p.name.lower() for p in candidate.rglob("*") if p.is_file())
        if any(token in names for token in ["paper", "code", "idea"]):
            return candidate
    return None


def collect_candidate_project_dirs(data_root: Path) -> List[Path]:
    dirs = []
    for path in [data_root] + [p for p in data_root.rglob("*") if p.is_dir()]:
        files = list_files(path, max_files=300)
        if not files:
            continue
        hints = Counter(infer_file_type(f) for f in files)
        if hints["paper"] or hints["code"] or hints["idea"]:
            if path == data_root or path.parent == data_root or hints["paper"] >= 1:
                dirs.append(path)
    unique = []
    seen = set()
    for path in sorted(dirs, key=lambda p: (len(p.parts), str(p))):
        if any(parent in seen for parent in path.parents):
            continue
        seen.add(path)
        unique.append(path)
    return unique


def first_matching_file(root: Path, keywords: List[str], suffixes: Optional[set[str]] = None) -> Optional[Path]:
    matches = []
    for path in list_files(root):
        low = str(path.relative_to(root)).lower()
        if suffixes and path.suffix.lower() not in suffixes:
            continue
        if any(k in low for k in keywords):
            matches.append(path)
    if matches:
        return sorted(matches, key=lambda p: (p.stat().st_size if p.exists() else 0, str(p)))[-1]
    return None


def choose_paper_file(project_dir: Path) -> Optional[Path]:
    return first_matching_file(project_dir, ["paper", "report", "manuscript", "writeup", ".pdf"], PAPER_SUFFIXES)


def choose_idea_file(project_dir: Path) -> Optional[Path]:
    return first_matching_file(project_dir, ["edited", "idea", "proposal"], TEXT_SUFFIXES)


def choose_codebase_dir(project_dir: Path) -> Optional[Path]:
    named = []
    for p in project_dir.rglob("*"):
        if p.is_dir() and any(k in p.name.lower() for k in ["code", "codebase", "src", "repo"]):
            named.append(p)
    if named:
        return sorted(named, key=lambda p: len(list_files(p, max_files=1000)), reverse=True)[0]
    code_files = [p for p in list_files(project_dir) if p.suffix.lower() in CODE_SUFFIXES]
    if len(code_files) >= 2:
        return project_dir
    return None


def extract_idea_fields(text: str) -> Tuple[str, str, str]:
    original = ""
    edited = ""
    source = "unknown"
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            original = str(obj.get("original_idea") or obj.get("idea") or obj.get("proposal") or "")
            edited = str(obj.get("edited_idea") or obj.get("edited") or obj.get("final_idea") or original)
            source = normalize_source(obj.get("source") or obj.get("condition") or obj.get("idea_source"))
            return original, edited, source
    except Exception:
        pass
    source = normalize_source(text)
    edited_match = re.search(r"(?:edited idea|edited proposal|final proposal)\s*[:#-]\s*(.+)", text, re.I | re.S)
    original_match = re.search(r"(?:original idea|seed idea|initial proposal)\s*[:#-]\s*(.+?)(?:\n#{1,6}|\nedited idea|\nfinal proposal|$)", text, re.I | re.S)
    if original_match:
        original = original_match.group(1).strip()
    if edited_match:
        edited = edited_match.group(1).strip()
    if not edited:
        edited = text.strip()
    return original[:20_000], edited[:40_000], source


def normalize_source(value: Any) -> str:
    text = str(value or "").lower()
    if "human" in text or "expert" in text:
        return "Human"
    if "ai" in text or "llm" in text or "machine" in text or "agent" in text:
        return "AI"
    return "unknown"


def summarize_codebase(code_dir: Optional[Path], max_chars: int = 60_000) -> str:
    if not code_dir or not code_dir.exists():
        return ""
    priority_patterns = [
        "readme",
        "main",
        "train",
        "eval",
        "test",
        "model",
        "dataset",
        "data",
        "config",
        "loss",
        "prompt",
        "inference",
    ]
    files = [p for p in list_files(code_dir) if p.suffix.lower() in CODE_SUFFIXES or p.name.lower().startswith("readme")]
    def score(path: Path) -> Tuple[int, int, str]:
        low = path.name.lower()
        pri = max([len(priority_patterns) - i for i, pat in enumerate(priority_patterns) if pat in low] or [0])
        size = path.stat().st_size if path.exists() else 0
        return (pri, -min(size, 30_000), str(path))
    chunks = []
    total = 0
    for path in sorted(files, key=score, reverse=True)[:40]:
        text = read_text(path, max_chars=12_000)
        if not text.strip():
            continue
        rel = path.relative_to(code_dir)
        block = f"\n\n### FILE: {rel}\n{text[:12_000]}"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "".join(chunks).strip()


def spec_field_value(spec: Dict[str, Any], field: str) -> Any:
    return spec.get("paper_derived_specification", {}).get(field)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def contains_leak(text: str, needles: Iterable[str]) -> bool:
    low = normalize_ws(text).lower()
    for needle in needles:
        n = normalize_ws(needle).lower()
        if len(n) >= 5 and n in low:
            return True
    return False


def strip_json_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_loads(text: str) -> Any:
    text = strip_json_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class LLMJsonClient:
    def __init__(self, model: str, enabled: bool) -> None:
        self.model = model
        self.enabled = enabled
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None,
        )
        return self._client

    def call_json(self, system: str, user: str, raw_path: Path, retries: int = 2) -> Any:
        if not self.enabled:
            raise RuntimeError("LLM is disabled")
        last = None
        for attempt in range(retries + 1):
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content or ""
            ensure_dir(raw_path.parent)
            raw_path.write_text(text, encoding="utf-8")
            try:
                return safe_json_loads(text)
            except Exception as exc:
                last = exc
                user = user + "\n\nPrevious output was not valid JSON. Return strict JSON only."
        raise RuntimeError(f"LLM JSON parse failed: {last}")


def maybe_limit(rows: List[Any], limit: Optional[int]) -> List[Any]:
    return rows if limit is None else rows[:limit]


def set_seed(seed: int) -> None:
    random.seed(seed)
