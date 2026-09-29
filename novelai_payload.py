"""NovelAI V4/V5 structured prompt payload helpers."""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

_SUBJECT_TAG_RE = re.compile(
    r"(?<![\w])(?:(?:\d+|6\+)\s*)?(girls?|boys?|others?)(?![\w])",
    re.IGNORECASE,
)
_COUNTED_SUBJECT_RE = re.compile(
    r"(?<![\w])(?:\d+|6\+)\s*(girls?|boys?|others?)(?![\w])",
    re.IGNORECASE,
)
_WEIGHT_BLOCK_RE = re.compile(
    r"^\s*(?P<weight>[-+]?(?:\d+(?:\.\d+)?|\.\d+)?)\s*::"
    r"(?P<body>.*?)::\s*$",
    re.DOTALL,
)


def _split_prompt_commas(text: str) -> list[str]:
    """Split prompt commas without breaking NovelAI weight/randomizer blocks."""
    def split_unweighted_commas(value: str) -> list[str]:
        """Split a malformed/unclosed weight tail without losing its commas."""
        result: list[str] = []
        current: list[str] = []
        in_randomizer = False
        index = 0
        while index < len(value):
            pair = value[index : index + 2]
            if pair == "||":
                in_randomizer = not in_randomizer
                current.append(pair)
                index += 2
                continue
            if value[index] == "," and not in_randomizer:
                result.append("".join(current).strip())
                current = []
            else:
                current.append(value[index])
            index += 1
        result.append("".join(current).strip())
        return result

    parts: list[str] = []
    current: list[str] = []
    in_weight = False
    in_randomizer = False
    index = 0
    text = str(text or "")
    while index < len(text):
        pair = text[index : index + 2]
        if pair == "||":
            in_randomizer = not in_randomizer
            current.append(pair)
            index += 2
            continue
        if pair == "::" and not in_randomizer:
            in_weight = not in_weight
            current.append(pair)
            index += 2
            continue
        if text[index] == "," and not in_weight and not in_randomizer:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(text[index])
        index += 1
    # Keep valid weight blocks intact, but do not let one unclosed ``::``
    # consume every following negative tag.  This is common in copied prompts
    # containing a malformed attention block such as ``{{{blurry, ...::``.
    tail = "".join(current)
    if in_weight:
        parts.extend(split_unweighted_commas(tail))
    else:
        parts.append(tail.strip())
    return parts


def _split_uc_prompt(text: str) -> tuple[str, str]:
    """Separate ``--uc`` from the positive prompt without changing its contents."""
    match = re.search(r"(?<![\w-])--uc\b", str(text or ""), re.IGNORECASE)
    if not match:
        return str(text or ""), ""
    return str(text[: match.start()]).rstrip(" ,"), str(text[match.start() :]).strip()


def _normalize_prompt_tag(tag: str) -> str:
    """Return a conservative comparison key for one prompt tag.

    Prompt tags are compared case-insensitively and with collapsed whitespace,
    but punctuation, underscores, braces, and other syntax are deliberately
    kept intact.  This avoids treating merely related tags such as ``blurry``
    and ``blurry background`` as duplicates.
    """
    return re.sub(r"\s+", " ", str(tag or "").strip(" ,")).casefold()


def _normalize_weight_key(weight: str | None) -> str | None:
    """Normalize a numeric weight for negative-side duplicate detection."""
    if weight is None:
        return None
    if not weight.strip():
        return ""
    try:
        return format(Decimal(weight), "f").rstrip("0").rstrip(".") or "0"
    except (InvalidOperation, ValueError):
        return weight.strip()


def _weighted_tag_parts(segment: str) -> tuple[str | None, list[str]]:
    """Return ``(weight, tags)`` for one top-level prompt segment.

    NovelAI's ``1.2::tag, another tag::`` is kept as one top-level segment by
    :func:`_split_prompt_commas`.  Splitting its body here lets us remove one
    duplicate without accidentally deleting the other tags in the weighted
    group.
    """
    match = _WEIGHT_BLOCK_RE.fullmatch(str(segment or ""))
    if not match:
        return None, [str(segment or "").strip()]
    return match.group("weight"), _split_prompt_commas(match.group("body"))


def deduplicate_negative_prompt(
    negative_prompt: str,
    positive_prompt: str = "",
) -> str:
    """Clean duplicate tags from a NovelAI negative prompt.

    Tags that also occur in the positive prompt are removed only from the
    negative side.  Matching ignores a surrounding numeric weight, while
    duplicate negative entries retain distinct weights (for example,
    ``0.5::blurry::`` and ``1.5::blurry::`` are both preserved).  Commas inside
    weighted blocks remain protected and weighted groups are rebuilt after
    filtering so their NovelAI syntax stays valid.
    """
    positive_text, _ = _split_uc_prompt(str(positive_prompt or ""))
    positive_tags: set[str] = set()
    for segment in _split_prompt_commas(positive_text):
        _, tags = _weighted_tag_parts(segment)
        positive_tags.update(
            key for tag in tags if (key := _normalize_prompt_tag(tag))
        )

    negative_text, _ = _split_uc_prompt(str(negative_prompt or ""))
    result: list[str] = []
    seen_negative: set[tuple[str, str | None]] = set()
    for segment in _split_prompt_commas(negative_text):
        segment = segment.strip(" ,")
        if not segment:
            continue
        weight, tags = _weighted_tag_parts(segment)
        kept_tags: list[str] = []
        weight_key = _normalize_weight_key(weight)
        for tag in tags:
            tag = tag.strip(" ,")
            key = _normalize_prompt_tag(tag)
            if not key:
                continue
            if key in positive_tags:
                continue
            identity = (key, weight_key)
            if identity in seen_negative:
                continue
            seen_negative.add(identity)
            kept_tags.append(tag)

        if not kept_tags:
            continue
        if weight is None:
            result.extend(kept_tags)
        else:
            result.append(f"{weight.strip()}::{', '.join(kept_tags)}::")
    return ", ".join(result).strip(" ,")


def remove_negative_prompt_tags(
    negative_prompt: str,
    tags_to_remove: set[str] | list[str] | tuple[str, ...],
) -> str:
    """Remove exact negative tags while preserving weighted-block syntax."""
    targets = {
        key
        for tag in tags_to_remove
        if (key := _normalize_prompt_tag(tag))
    }
    if not targets:
        return str(negative_prompt or "").strip(" ,")

    result: list[str] = []
    for segment in _split_prompt_commas(str(negative_prompt or "")):
        segment = segment.strip(" ,")
        if not segment:
            continue
        weight, tags = _weighted_tag_parts(segment)
        kept_tags = [
            tag.strip(" ,")
            for tag in tags
            if _normalize_prompt_tag(tag) not in targets
            and _normalize_prompt_tag(tag)
        ]
        if not kept_tags:
            continue
        if weight is None:
            result.extend(kept_tags)
        else:
            result.append(f"{weight.strip()}::{', '.join(kept_tags)}::")
    return ", ".join(result).strip(" ,")


def remove_positive_duplicates_from_negative(
    positive_prompt: str,
    negative_prompt: str,
) -> str:
    """Compatibility-oriented name for positive-vs-UC duplicate removal."""
    return deduplicate_negative_prompt(
        negative_prompt,
        positive_prompt=positive_prompt,
    )


def _infer_character_sections(
    base_caption: str,
    character_hints: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, list[str]]:
    """Recover character fields from a flattened prompt conservatively.

    Repeated unnumbered ``girl``/``boy``/``other`` tags are unambiguous.  A
    single character is only recovered when a caller supplies a known character
    hint (normally a plugin character preset/database binding); this prevents a
    generic ``girl, red hair`` prompt from being moved into a character field.
    ``--uc`` is kept in the base prompt and never becomes a character attribute.
    """
    text, uc_prompt = _split_uc_prompt(base_caption)
    parts = _split_prompt_commas(text)
    subject_indexes = [
        index
        for index, part in enumerate(parts)
        if re.fullmatch(r"(?i)(?:girl|boy|other)", part)
    ]
    all_subject_indexes = [
        index for index, part in enumerate(parts) if _SUBJECT_TAG_RE.fullmatch(part)
    ]

    # First recover the unambiguous flattened multi-character form.
    if len(subject_indexes) >= 2:
        first_boundary = subject_indexes[0]
        inferred_base_parts = parts[:first_boundary]
        inferred_characters = [
            ", ".join(parts[start:end]).strip(" ,")
            for offset, start in enumerate(subject_indexes)
            for end in [
                subject_indexes[offset + 1]
                if offset + 1 < len(subject_indexes)
                else len(parts)
            ]
        ]
        inferred_characters = [part for part in inferred_characters if part]
        if len(inferred_characters) >= 2:
            if uc_prompt:
                inferred_base_parts.append(uc_prompt)
            return ", ".join(inferred_base_parts).strip(" ,"), inferred_characters

    # A single character requires an explicit known identity.  Hints may be a
    # full preset (``1girl, phoebe, ...``), so compare each comma token.
    # Do not reinterpret an explicit multi-subject count such as ``2girls``.
    # A lone ``1girl`` is allowed because it is equivalent to the local marker
    # used by Character Prompt fields.
    all_subject_parts = [parts[index] for index in all_subject_indexes]
    subject_types = {
        (
            "girl"
            if re.search(r"girls?", part, re.IGNORECASE)
            else "boy"
            if re.search(r"boys?", part, re.IGNORECASE)
            else "other"
        )
        for part in all_subject_parts
    }
    has_multi_count = any(
        _COUNTED_SUBJECT_RE.fullmatch(part)
        and not re.fullmatch(r"(?i)1\s*(?:girls?|boys?|others?)", part)
        for part in all_subject_parts
    )
    if (
        len(subject_indexes) > 1
        or len(subject_types) > 1
        or has_multi_count
        or not character_hints
    ):
        return base_caption, []
    normalized_hints = [
        " ".join(str(hint or "").split()).casefold()
        for hint in character_hints
        if str(hint or "").strip()
    ]
    if not normalized_hints:
        return base_caption, []

    identity_indexes: list[int] = []
    for index, part in enumerate(parts):
        folded_part = part.casefold()
        if any(
            hint in folded_part
            or folded_part in hint
            for hint in normalized_hints
            if hint
        ):
            identity_indexes.append(index)
    if not identity_indexes:
        return base_caption, []

    identity_index = identity_indexes[0]
    preceding_subjects = [
        index
        for index, part in enumerate(parts[: identity_index + 1])
        if _SUBJECT_TAG_RE.fullmatch(part)
    ]
    if not preceding_subjects:
        return base_caption, []
    start = preceding_subjects[-1]
    # A counted subject before a known identity is still a local subject marker;
    # normalize_v5_prompt will remove its count and rebuild the global count.
    # Repeated ``girl, 1girl`` markers from a preset expansion are harmless;
    # the local normalizer deduplicates them.  Mixed types/counts were rejected
    # above.

    # Negative weighted blocks are global controls, not character attributes.
    end = len(parts)
    for index in range(identity_index + 1, len(parts)):
        if re.match(r"^-\s*(?:\d+(?:\.\d+)?|\d*\.\d+)\s*::", parts[index]):
            end = index
            break
    character = ", ".join(parts[start:end]).strip(" ,")
    if not character:
        return base_caption, []
    inferred_base_parts = parts[:start] + parts[end:]
    if uc_prompt:
        inferred_base_parts.append(uc_prompt)
    return ", ".join(inferred_base_parts).strip(" ,"), [character]


def _normalize_segment(segment: str) -> str:
    text = str(segment or "")
    text = text.replace("，", ",").replace("、", ",")

    # Do not insert spaces into ``1.2::tag, tag::`` or randomizer blocks:
    # commas inside these constructs belong to the weighted/randomized tag
    # expression and changing them can alter NovelAI's parser boundaries.
    normalized_chars: list[str] = []
    in_weight = False
    in_randomizer = False
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair == "||":
            in_randomizer = not in_randomizer
            normalized_chars.append(pair)
            index += 2
            continue
        if pair == "::" and not in_randomizer:
            in_weight = not in_weight
            normalized_chars.append(pair)
            index += 2
            continue
        if text[index] == "," and not in_weight and not in_randomizer:
            while normalized_chars and normalized_chars[-1] in {" ", "\t"}:
                normalized_chars.pop()
            if not normalized_chars or normalized_chars[-1] != ",":
                normalized_chars.append(",")
            normalized_chars.append(" ")
            index += 1
            while index < len(text) and text[index] in " \t,":
                index += 1
            continue
        normalized_chars.append(text[index])
        index += 1
    text = "".join(normalized_chars)
    paragraphs = []
    for paragraph in re.split(r"(?:\r?\n\s*){2,}", text):
        normalized_lines = [
            " ".join(line.split()) for line in paragraph.splitlines() if line.strip()
        ]
        normalized_paragraph = " ".join(normalized_lines).strip(" ,")
        if normalized_paragraph:
            paragraphs.append(normalized_paragraph)
    return "\n\n".join(paragraphs)


def split_v5_prompt(prompt: str) -> tuple[str, list[str]]:
    """Split the compact ``base | character`` syntax safely.

    Prompt Randomizer expressions use ``||option|option||``. Single bars inside
    those expressions are preserved instead of being mistaken for character
    separators.

    Args:
        prompt: Compact NovelAI prompt text.

    Returns:
        Normalized base caption and ordered character captions.
    """
    text = str(prompt or "")
    segments: list[str] = []
    current: list[str] = []
    inside_randomizer = False
    index = 0

    while index < len(text):
        if text[index : index + 2] == "||":
            inside_randomizer = not inside_randomizer
            current.append("||")
            index += 2
            continue
        if text[index] == "|" and not inside_randomizer:
            segments.append(_normalize_segment("".join(current)))
            current = []
            index += 1
            continue
        current.append(text[index])
        index += 1

    segments.append(_normalize_segment("".join(current)))
    base_caption = segments[0] if segments else ""
    char_captions = [segment for segment in segments[1:] if segment]
    return base_caption, char_captions


def normalize_v5_prompt(
    prompt: str,
    max_characters: int = 22,
    character_hints: list[str] | tuple[str, ...] | None = None,
    infer_character_sections: bool = False,
) -> str:
    """Normalize a prompt into NovelAI V5 compact multi-character syntax.

    Args:
        prompt: User-provided prompt.
        max_characters: Maximum number of individual character captions.
        character_hints: Known character names or preset contents.  These enable
            conservative recovery of a single flattened character section.
        infer_character_sections: Whether to recover flattened multi-character
            prompts without explicit ``|`` separators. Disabled by default so
            ordinary prompts are submitted without semantic restructuring.

    Returns:
        Normalized prompt in ``base | character`` form.

    Raises:
        ValueError: If the prompt contains more character captions than allowed.
    """
    base_caption, char_captions = split_v5_prompt(prompt)
    # Explicit ``|`` separators are the only normal image-generation path for
    # Character Prompt fields.  Inference remains available as an opt-in
    # compatibility feature for callers that deliberately request it.
    if not char_captions and infer_character_sections:
        base_caption, char_captions = _infer_character_sections(
            base_caption, character_hints=character_hints
        )
    if len(char_captions) > max_characters:
        raise ValueError(
            f"角色提示词数量为 {len(char_captions)}，当前模型最多支持 {max_characters} 个"
        )

    if not char_captions:
        return base_caption

    normalized_characters: list[str] = []
    subject_counts = {"girl": 0, "boy": 0, "other": 0}
    for index, caption in enumerate(char_captions, start=1):
        singular = ""
        local_parts: list[str] = []
        for part in _split_prompt_commas(caption):
            clean_part = part.strip()
            if not clean_part or clean_part.casefold() == "solo":
                continue
            subject_match = _SUBJECT_TAG_RE.fullmatch(clean_part)
            if not subject_match:
                local_parts.append(clean_part)
                continue
            subject = subject_match.group(1).lower()
            if subject.startswith("girl"):
                normalized_subject = "girl"
            elif subject.startswith("boy"):
                normalized_subject = "boy"
            else:
                normalized_subject = "other"
            if singular and normalized_subject != singular:
                raise ValueError(f"角色段 {index} 包含互相冲突的主体类型")
            singular = normalized_subject

        if not singular:
            raise ValueError(
                f"角色段 {index} 必须以 girl、boy 或 other 之一标明主体类型"
            )
        subject_counts[singular] += 1
        local_parts.insert(0, singular)
        normalized_characters.append(_normalize_segment(", ".join(local_parts)))

    # Rebuild subject counts from character sections so the Base Prompt cannot drift.
    base_parts = [
        part.strip() for part in _split_prompt_commas(base_caption) if part.strip()
    ]
    base_parts = [
        part
        for part in base_parts
        if part.casefold() != "solo"
        and _COUNTED_SUBJECT_RE.fullmatch(part) is None
        and _SUBJECT_TAG_RE.fullmatch(part) is None
    ]
    count_tags: list[str] = []
    for subject in ("girl", "boy", "other"):
        count = subject_counts[subject]
        if not count:
            continue
        if count == 1:
            count_tags.append(f"1{subject}")
        elif count <= 5:
            count_tags.append(f"{count}{subject}s")
        else:
            count_tags.append(f"6+{subject}s")

    if len(normalized_characters) == 1:
        count_tags.append("solo")
    normalized_base = _normalize_segment(", ".join(count_tags + base_parts))
    return " | ".join([normalized_base, *normalized_characters])


def _character_centers(
    count: int,
    width: int | None = None,
    height: int | None = None,
) -> list[dict[str, float]]:
    if count <= 0:
        return []

    aspect_ratio = (width / height) if width and height else 1.0
    columns = min(count, max(1, math.ceil(math.sqrt(count * aspect_ratio))))
    rows = math.ceil(count / columns)
    centers: list[dict[str, float]] = []

    for row in range(rows):
        row_start = row * columns
        items_in_row = min(columns, count - row_start)
        for column in range(items_in_row):
            centers.append(
                {
                    "x": round((column + 1) / (items_in_row + 1), 4),
                    "y": round((row + 1) / (rows + 1), 4),
                }
            )

    return centers


def build_v4_prompt(
    prompt: str,
    width: int | None = None,
    height: int | None = None,
    max_characters: int = 22,
    character_positions: list[dict[str, float] | None] | None = None,
) -> dict:
    """Build the structured positive prompt used by V4 and V5 APIs.

    Args:
        prompt: Normalized base or compact multi-character prompt.
        width: Output width used to arrange automatic character coordinates.
        height: Output height used to arrange automatic character coordinates.
        max_characters: Maximum number of character captions for the model.
        character_positions: Optional planner-selected centers aligned with the
            character captions. Missing or invalid entries use automatic centers.

    Returns:
        Structured payload accepted by the historical ``v4_prompt`` API field.
    """
    normalized = normalize_v5_prompt(
        prompt,
        max_characters=max_characters,
        infer_character_sections=False,
    )
    base_caption, char_captions = split_v5_prompt(normalized)
    centers = _character_centers(len(char_captions), width, height)
    use_coords = bool(character_positions) and any(
        isinstance(position, dict) and bool(position)
        for position in character_positions[: len(centers)]
    )
    if character_positions:
        for index, position in enumerate(character_positions[: len(centers)]):
            if not isinstance(position, dict):
                continue
            try:
                x = float(position.get("x", centers[index]["x"]))
                y = float(position.get("y", centers[index]["y"]))
            except (TypeError, ValueError):
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                centers[index] = {"x": round(x, 4), "y": round(y, 4)}
    characters = [
        {"char_caption": caption, "centers": [centers[index]]}
        for index, caption in enumerate(char_captions)
    ]
    return {
        "caption": {
            "base_caption": base_caption,
            "char_captions": characters,
        },
        "use_coords": bool(characters) and use_coords,
        "use_order": True,
        "legacy_uc": False,
    }


def build_v4_negative_prompt(
    base_caption: str,
    character_count: int = 0,
    width: int | None = None,
    height: int | None = None,
    character_positions: list[dict[str, float] | None] | None = None,
    character_captions: list[str] | None = None,
    use_coords: bool | None = None,
) -> dict:
    """Build the matching structured negative prompt for V4/V5 models.

    Args:
        base_caption: Global undesired-content prompt.
        character_count: Number of positive character captions to align with.
        width: Output width used for matching character coordinates.
        height: Output height used for matching character coordinates.
        character_positions: Optional centers aligned with the positive character
            captions. Missing or invalid entries use automatic centers.
        character_captions: Optional character-specific undesired-content captions.
            Missing entries are kept empty and extra entries are ignored.
        use_coords: Whether custom character coordinates are active. When omitted,
            infer the value from explicit character positions.

    Returns:
        Structured payload accepted by ``v4_negative_prompt``.
    """
    centers = _character_centers(character_count, width, height)
    if use_coords is None:
        use_coords = bool(character_positions) and any(
            isinstance(position, dict) and bool(position)
            for position in character_positions[: len(centers)]
        )
    if character_positions:
        for index, position in enumerate(character_positions[: len(centers)]):
            if not isinstance(position, dict):
                continue
            try:
                x = float(position.get("x", centers[index]["x"]))
                y = float(position.get("y", centers[index]["y"]))
            except (TypeError, ValueError):
                continue
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                centers[index] = {"x": round(x, 4), "y": round(y, 4)}
    negative_captions = character_captions or []
    characters = [
        {
            "char_caption": _normalize_segment(
                negative_captions[index] if index < len(negative_captions) else ""
            ),
            "centers": [center],
        }
        for index, center in enumerate(centers)
    ]
    return {
        "caption": {
            "base_caption": _normalize_segment(base_caption),
            "char_captions": characters,
        },
        "use_coords": bool(characters) and bool(use_coords),
        "use_order": False,
        "legacy_uc": False,
    }
