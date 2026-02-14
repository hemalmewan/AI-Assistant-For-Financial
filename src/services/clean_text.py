import re
from collections import Counter


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph structure.
    """
    # Replace multiple spaces with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize newlines (preserve paragraphs)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_repeated_lines(pages, threshold=0.7):
    """
    Detect lines that appear on a large fraction of pages
    (headers / footers).
    """
    line_counter = Counter()
    total_pages = len(pages)

    for page in pages:
        lines = set(page.splitlines())
        for line in lines:
            cleaned = line.strip()
            if cleaned:
                line_counter[cleaned] += 1

    repeated_lines = {
        line for line, count in line_counter.items()
        if count / total_pages >= threshold
    }

    return repeated_lines


def remove_headers_footers(text: str, repeated_lines: set) -> str:
    """
    Remove detected headers and footers from page text.
    """
    cleaned_lines = []
    for line in text.splitlines():
        if line.strip() not in repeated_lines:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_page_numbers(text: str) -> str:
    """
    Remove common page number patterns.
    """
    patterns = [
        r"^\s*Page\s+\d+\s*$",
        r"^\s*\d+\s*$",
        r"^\s*[-–—]\s*\d+\s*[-–—]\s*$",
        r"^\s*\|\s*\d+\s*\|\s*$",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        if not any(re.match(p, line.strip()) for p in patterns):
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def clean_page_text(page_text: str, repeated_lines: set) -> str:
    """
    Full cleaning pipeline for a single page.
    """
    text = normalize_whitespace(page_text)
    text = remove_headers_footers(text, repeated_lines)
    text = remove_page_numbers(text)

    return text.strip()


def clean_document(pages: list[str]) -> list[str]:
    """
    Clean an entire document page-by-page.

    Args:
        pages: List of raw page texts extracted from PDF.

    Returns:
        List of cleaned page texts.
    """
    repeated_lines = detect_repeated_lines(pages)

    cleaned_pages = []
    for page in pages:
        cleaned = clean_page_text(page, repeated_lines)
        if cleaned:
            cleaned_pages.append(cleaned)

    return cleaned_pages
