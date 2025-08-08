import argparse
import hashlib
import os
import re
from pathlib import Path


STYLE_ATTR_PATTERN = re.compile(
    r"<([a-zA-Z][^\s>/]*)([^>]*)\sstyle=(\"|\')(?P<style>[^\"']*)(\3)([^>]*)>"
)
CLASS_ATTR_PATTERN = re.compile(r"\sclass=(\"|\')(?P<class>[^\"']*)(\1)")


def generate_class_name(style_value: str) -> str:
    normalized = ";".join(
        [
            seg.strip()
            for seg in style_value.strip().strip(";").split(";")
            if seg.strip()
        ]
    )
    style_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"inl-{style_hash}", normalized


def compute_relative_css_path(
    html_path: Path, website_root: Path, css_rel_path: str
) -> str:
    try:
        rel = html_path.relative_to(website_root)
    except ValueError:
        return css_rel_path
    depth = len(rel.parents) - 1  # number of directories from file to website root
    prefix = "../" * depth
    return f"{prefix}{css_rel_path}"


def ensure_css_link(content: str, css_href: str) -> str:
    link_tag = f'<link rel="stylesheet" href="{css_href}">'
    if link_tag in content:
        return content
    # Insert before </head>
    head_close_idx = content.lower().rfind("</head>")
    if head_close_idx != -1:
        return (
            content[:head_close_idx]
            + "\n    "
            + link_tag
            + "\n"
            + content[head_close_idx:]
        )
    # If no head, prepend
    return link_tag + "\n" + content


def replace_inline_styles(html: str, global_style_map: dict) -> tuple[str, bool]:
    changed = False

    def _replacer(match: re.Match) -> str:
        nonlocal changed
        tag = match.group(1)
        before = match.group(2) or ""
        style_value = match.group("style") or ""
        after = match.group(6) or ""

        class_name, normalized = generate_class_name(style_value)
        if normalized not in global_style_map:
            global_style_map[normalized] = class_name
        else:
            class_name = global_style_map[normalized]

        attrs = before + after
        class_match = CLASS_ATTR_PATTERN.search(attrs)
        if class_match:
            existing = class_match.group("class").strip()
            classes = set([c for c in existing.split() if c])
            classes.add(class_name)
            combined = " ".join(sorted(classes))
            # Replace class attribute content
            attrs = (
                attrs[: class_match.start()]
                + f' class="{combined}"'
                + attrs[class_match.end() :]
            )
        else:
            # Insert class attribute at the beginning of attrs
            attrs = f' class="{class_name}"' + attrs

        # Remove any residual style= attribute from attrs (safety)
        attrs = re.sub(r"\sstyle=(\"|\')[^\"']*(\1)", "", attrs)

        changed = True
        return f"<{tag}{attrs}>"

    new_html = STYLE_ATTR_PATTERN.sub(_replacer, html)
    return new_html, changed


def write_css(style_map: dict, out_css: Path):
    # style_map keys are normalized declarations; values are class names
    lines = [
        "/* Auto-generated from inline styles. Consolidated for maintainability. */",
        ":root { --inline-cleanup-generated: 1; }",
        "",
    ]
    for normalized, cls in sorted(style_map.items(), key=lambda x: x[1]):
        lines.append(f".{cls} {{ {normalized}; }}")
    out_css.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Refactor inline styles into a consolidated CSS file and update HTML."
    )
    parser.add_argument(
        "--base",
        default="website",
        help="Base directory to scan for HTML files (default: website)",
    )
    parser.add_argument(
        "--css",
        default="css/inline-extracted.css",
        help="CSS path relative to website root (default: css/inline-extracted.css)",
    )
    parser.add_argument(
        "--dry", action="store_true", help="Dry run; do not modify files"
    )
    args = parser.parse_args()

    website_root = Path(args.base).resolve()
    css_rel_path = args.css
    css_abs_path = website_root / css_rel_path
    css_abs_path.parent.mkdir(parents=True, exist_ok=True)

    global_style_map: dict[str, str] = {}
    modified_files: list[Path] = []

    html_files: list[Path] = []
    for root, _dirs, files in os.walk(website_root):
        for f in files:
            if f.lower().endswith(".html"):
                html_files.append(Path(root) / f)

    for file_path in html_files:
        try:
            original = file_path.read_text(encoding="utf-8")
        except Exception:
            continue
        updated, changed = replace_inline_styles(original, global_style_map)
        if changed:
            css_href = compute_relative_css_path(file_path, website_root, css_rel_path)
            updated_with_link = ensure_css_link(updated, css_href)
            if not args.dry:
                file_path.write_text(updated_with_link, encoding="utf-8")
            modified_files.append(file_path)

    if global_style_map and not args.dry:
        write_css(global_style_map, css_abs_path)

    print(
        f"Processed {len(html_files)} HTML files. Modified {len(modified_files)}. Consolidated {len(global_style_map)} unique inline styles into {css_abs_path}."
    )


if __name__ == "__main__":
    main()
