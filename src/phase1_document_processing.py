"""
Phase 1 – Document Processing  (UTF-8 / encoding-safe)
Extracts, cleans, chunks text from PDF, DOCX, EML and TXT files.
"""

import json
import re
import fitz                 # PyMuPDF
import docx
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from tqdm import tqdm
import yaml
import mimetypes


# --------------------------------------------------------------------------- #
# Helper-level functions
# --------------------------------------------------------------------------- #
def safe_read_text(path: Path) -> str:
    """
    Read a plain-text file with utf-8 and graceful fallback.
    Replaces undecodable bytes so the pipeline never crashes.
    """
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.error(f"Failed reading {path.name}: {exc}")
        return ""


def is_text_file(path: Path) -> bool:
    """Naïve mime check to avoid trying to open a binary as text."""
    mime, _ = mimetypes.guess_type(path.as_posix())
    return bool(mime and mime.startswith("text"))


# --------------------------------------------------------------------------- #
class DocumentProcessor:
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.raw_docs_path = Path(self.config["paths"]["raw_documents"])
        self.processed_path = Path(self.config["paths"]["processed_data"])
        self.processed_path.mkdir(parents=True, exist_ok=True)

        self.supported_formats = {
            ".pdf",
            ".docx",
            ".eml",
            ".txt",
        }
        self.chunk_size = self.config["document_processing"]["chunk_size"]
        self.chunk_overlap = self.config["document_processing"]["chunk_overlap"]

    # --------------------------------------------------------------------- #
    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # --------------------------------------------------------------------- #
    # 1.  Scan folder → create document_log.json
    # --------------------------------------------------------------------- #
    def create_document_log(self) -> List[Dict[str, Any]]:
        if not self.raw_docs_path.exists():
            logger.error(f"Raw docs path not found: {self.raw_docs_path}")
            return []

        logger.info(f"Scanning {self.raw_docs_path}")
        log: List[Dict[str, Any]] = []

        for p in self.raw_docs_path.iterdir():
            if p.suffix.lower() not in self.supported_formats:
                continue

            entry = {
                "file_name": p.name,
                "file_path": str(p),
                "file_size": p.stat().st_size,
                "doc_type": self._detect_doc_type(p.name),
                "format": p.suffix.lower(),
            }
            log.append(entry)
            logger.debug(f"Added: {p.name}")

        (self.processed_path / "document_log.json").write_text(
            json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Document log created with {len(log)} entries")
        return log

    @staticmethod
    def _detect_doc_type(fname: str) -> str:
        name = fname.lower()
        if any(k in name for k in ("policy", "insurance", "claim")):
            return "insurance_policy"
        if any(k in name for k in ("contract", "agreement")):
            return "contract"
        if fname.endswith(".eml"):
            return "email"
        return "document"

    # --------------------------------------------------------------------- #
    # 2.  Extraction helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pdf_to_text(path: Path) -> str:
        try:
            text = ""
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extract failed {path.name}: {e}")
            return ""

    @staticmethod
    def _docx_to_text(path: Path) -> str:
        try:
            d = docx.Document(path)
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
        except Exception as e:
            logger.error(f"DOCX extract failed {path.name}: {e}")
            return ""

    @staticmethod
    def _eml_to_text(path: Path) -> str:
        try:
            body: List[str] = []
            in_body = False
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if in_body:
                        body.append(line)
                    elif line.strip() == "":
                        in_body = True
            return "".join(body).strip()
        except Exception as e:
            logger.error(f"EML extract failed {path.name}: {e}")
            return ""

    # --------------------------------------------------------------------- #
    # 3. Extract all
    # --------------------------------------------------------------------- #
    def extract_all_text(self) -> List[Dict[str, Any]]:
        log_entries = self.create_document_log()
        extracted: List[Dict[str, Any]] = []
        logger.info("Extracting text ...")

        for entry in tqdm(log_entries, desc="Extracting"):
            p = Path(entry["file_path"])
            fmt = entry["format"]
            text = ""

            if fmt == ".pdf":
                text = self._pdf_to_text(p)
            elif fmt == ".docx":
                text = self._docx_to_text(p)
            elif fmt == ".eml":
                text = self._eml_to_text(p)
            elif fmt == ".txt" and is_text_file(p):
                text = safe_read_text(p)

            if text:
                extracted.append(
                    {
                        "file_name": entry["file_name"],
                        "doc_type": entry["doc_type"],
                        "content": text,
                        "char_count": len(text),
                        "word_count": len(text.split()),
                    }
                )
                logger.debug(f"✓ {entry['file_name']} ({len(text)} chars)")
            else:
                logger.warning(f"⚠️  No text extracted from {entry['file_name']}")

        (self.processed_path / "extracted_text.json").write_text(
            json.dumps(extracted, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Text extraction complete: {len(extracted)} docs")
        return extracted

    # --------------------------------------------------------------------- #
    # 4. Cleaning helpers
    # --------------------------------------------------------------------- #
    def _default_clean_patterns(self) -> Dict[str, str]:
        return {
            "page_numbers": r"Page \\d+ of \\d+",
            "headers": r"CONFIDENTIAL|PROPRIETARY|INTERNAL USE ONLY",
            "email_headers": r"^(From|To|Subject|Date|CC|BCC):.*$",
            "urls": r"http[s]?://\\S+",
            "extra_spaces": r"\\s{3,}",
        }

    def clean_text(self, text: str) -> str:
        patterns = self._default_clean_patterns()
        for pat in patterns.values():
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"\\n{2,}", "\\n\\n", text)
        text = re.sub(r"\\s+", " ", text)
        return text.strip()

    # --------------------------------------------------------------------- #
    # 5. Chunking
    # --------------------------------------------------------------------- #
    def chunk(self, text: str) -> List[str]:
        words = text.split()
        size, overlap = self.chunk_size, self.chunk_overlap
        return [
            " ".join(words[i : i + size]).strip()
            for i in range(0, len(words), size - overlap)
            if words[i : i + size]
        ]

    # --------------------------------------------------------------------- #
    # 6. Full pipeline
    # --------------------------------------------------------------------- #
    def process_all_documents(self) -> List[Dict[str, Any]]:
        logger.info("🚀 Full document-processing pipeline ...")

        extracted = self.extract_all_text()
        chunked: List[Dict[str, Any]] = []

        for doc in tqdm(extracted, desc="Cleaning & chunking"):
            cleaned = self.clean_text(doc["content"])
            for idx, ch in enumerate(self.chunk(cleaned), 1):
                chunked.append(
                    {
                        "file_name": doc["file_name"],
                        "chunk_id": f"{Path(doc['file_name']).stem}-C{idx}",
                        "doc_type": doc["doc_type"],
                        "chunk_index": idx,
                        "word_count": len(ch.split()),
                        "text": ch,
                    }
                )

        (self.processed_path / "chunked_documents.json").write_text(
            json.dumps(chunked, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.success(f"Chunking complete: {len(chunked)} chunks")
        return chunked


# --------------------------------------------------------------------------- #
def main() -> None:
    proc = DocumentProcessor()
    chunks = proc.process_all_documents()

    print("\n📊 Processing Summary")
    print(f"   Total chunks generated : {len(chunks)}")
    print(f"   Output folder          : {proc.processed_path}")
    print("   Files created:")
    print("     - document_log.json")
    print("     - extracted_text.json")
    print("     - chunked_documents.json")


if __name__ == "__main__":
    main()
