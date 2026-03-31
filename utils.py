"""
Shared utilities: SVG validation, extraction, constraint enforcement.
Matches competition rules exactly.
"""
import re
import xml.etree.ElementTree as ET

# Prevent ET.tostring() from adding ns0: prefixes to SVG elements
ET.register_namespace('', 'http://www.w3.org/2000/svg')

from config import ALLOWED_TAGS, FALLBACK_SVG, MAX_PATH_ELEMENTS, MAX_SVG_CHARS

SVG_PATTERN = re.compile(r"(<svg[\s\S]*?</svg>)", re.IGNORECASE)
SVG_OPEN_PATTERN = re.compile(r"(<svg[\s\S]*)", re.IGNORECASE)


def extract_svg(text: str) -> str:
    """Extract the first <svg>...</svg> block from generated text.
    If SVG was truncated (no closing tag), try to close it."""
    match = SVG_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    # Handle truncated SVGs — model hit max_new_tokens before </svg>
    open_match = SVG_OPEN_PATTERN.search(text)
    if open_match:
        partial = open_match.group(1).strip()
        # Close any unclosed tags and append </svg>
        partial = _close_truncated_svg(partial)
        return partial

    return ""


def _close_truncated_svg(partial: str) -> str:
    """Try to close a truncated SVG by removing incomplete tail and adding </svg>."""
    # Remove any incomplete tag at the end (e.g., "<path d="m10 20...)
    # Find the last complete > before end of string
    last_gt = partial.rfind(">")
    if last_gt > 0:
        partial = partial[:last_gt + 1]

    # Append closing tag
    if not partial.rstrip().endswith("</svg>"):
        partial = partial + "</svg>"

    return partial


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from a tag."""
    if "}" in tag:
        return tag.split("}")[-1]
    return tag


def is_valid_svg(svg_text: str) -> bool:
    """Check if a string is parseable, valid SVG with root <svg> element."""
    if not svg_text or not svg_text.strip():
        return False
    try:
        root = ET.fromstring(svg_text)
        return _strip_ns(root.tag).lower() == "svg"
    except ET.ParseError:
        return False


def check_constraints(svg_text: str) -> tuple[bool, str]:
    """
    Validate SVG against competition constraints.
    Returns (is_valid, reason).
    """
    if not svg_text:
        return False, "empty"

    # Max character length
    if len(svg_text) > MAX_SVG_CHARS:
        return False, f"too long ({len(svg_text)} > {MAX_SVG_CHARS})"

    # Must parse as XML with <svg> root
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"

    if _strip_ns(root.tag).lower() != "svg":
        return False, f"root is <{root.tag}>, not <svg>"

    # Check allowed tags and count paths
    path_count = 0
    for elem in root.iter():
        tag = _strip_ns(elem.tag).lower()
        if tag not in ALLOWED_TAGS:
            return False, f"disallowed tag: <{tag}>"
        if tag == "path":
            path_count += 1

    if path_count > MAX_PATH_ELEMENTS:
        return False, f"too many <path> elements ({path_count} > {MAX_PATH_ELEMENTS})"

    return True, "ok"


def remove_disallowed_tags(svg_text: str) -> str:
    """Remove elements with disallowed tags from SVG."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    def _clean(element):
        to_remove = []
        for child in element:
            tag = _strip_ns(child.tag).lower()
            if tag not in ALLOWED_TAGS:
                to_remove.append(child)
            else:
                _clean(child)
        for child in to_remove:
            element.remove(child)

    _clean(root)
    # Re-serialize
    return ET.tostring(root, encoding="unicode")


def ensure_svg_attrs(svg_text: str) -> str:
    """Ensure the SVG has required attributes and correct 256x256 dimensions."""
    if 'xmlns=' not in svg_text:
        svg_text = svg_text.replace(
            "<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1
        )

    # Force correct viewBox
    svg_text = re.sub(r'viewBox="[^"]*"', 'viewBox="0 0 256 256"', svg_text, count=1)
    if 'viewBox=' not in svg_text:
        svg_text = svg_text.replace("<svg", '<svg viewBox="0 0 256 256"', 1)

    # Force correct width/height
    svg_text = re.sub(r'width="[^"]*"', 'width="256"', svg_text, count=1)
    svg_text = re.sub(r'height="[^"]*"', 'height="256"', svg_text, count=1)
    if 'width=' not in svg_text:
        svg_text = svg_text.replace("<svg", '<svg width="256"', 1)
    if 'height=' not in svg_text:
        svg_text = svg_text.replace("<svg", '<svg height="256"', 1)

    return svg_text


def truncate_paths(svg_text: str, max_paths: int = MAX_PATH_ELEMENTS) -> str:
    """If SVG has too many <path> elements, keep only the first max_paths."""
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return svg_text

    paths = list(root.iter("{http://www.w3.org/2000/svg}path")) + list(root.iter("path"))
    if len(paths) <= max_paths:
        return svg_text

    for path in paths[max_paths:]:
        parent = find_parent(root, path)
        if parent is not None:
            parent.remove(path)

    return ET.tostring(root, encoding="unicode")


def find_parent(root, target):
    """Find parent element of target in XML tree."""
    for parent in root.iter():
        for child in parent:
            if child is target:
                return parent
    return None


def postprocess_svg(raw_text: str) -> str:
    """
    Full post-processing pipeline:
    1. Extract SVG from generated text
    2. Ensure required attributes
    3. Remove disallowed tags
    4. Truncate excess paths
    5. Truncate if too long
    6. Validate — fallback if invalid
    """
    svg = extract_svg(raw_text)
    if not svg:
        return FALLBACK_SVG

    svg = ensure_svg_attrs(svg)
    svg = remove_disallowed_tags(svg)
    svg = truncate_paths(svg)

    # Truncate if still too long (aggressive fallback)
    if len(svg) > MAX_SVG_CHARS:
        return FALLBACK_SVG

    valid, reason = check_constraints(svg)
    if not valid:
        return FALLBACK_SVG

    return svg


def pick_first_field(example: dict, field_names: list) -> str:
    """Return the first non-empty value from a list of candidate field names."""
    for key in field_names:
        if key in example and example[key] is not None:
            val = str(example[key]).strip()
            if val:
                return val
    return ""
