from data.plugins.astrbot_plugin_novelai.novelai_payload import (
    build_v4_negative_prompt,
    build_v4_prompt,
    deduplicate_negative_prompt,
    normalize_v5_prompt,
    remove_positive_duplicates_from_negative,
    split_v5_prompt,
)


def test_v4_prompt_uses_an_empty_character_caption_list():
    payload = build_v4_prompt("beach")
    assert set(payload) == {"caption", "use_coords", "use_order", "legacy_uc"}
    assert set(payload["caption"]) == {"base_caption", "char_captions"}
    assert payload["caption"]["base_caption"] == "beach"
    assert payload["caption"]["char_captions"] == []
    assert payload["legacy_uc"] is False


def test_v4_negative_prompt_uses_an_empty_character_caption_list():
    payload = build_v4_negative_prompt("bad anatomy")
    assert payload["legacy_uc"] is False
    assert payload["caption"]["base_caption"] == "bad anatomy"
    assert payload["caption"]["char_captions"] == []


def test_negative_prompt_removes_tags_repeated_in_positive_prompt():
    assert remove_positive_duplicates_from_negative(
        "girl, chromatic aberration, film grain",
        "blurry, chromatic aberration, film grain, bad anatomy",
    ) == "blurry, bad anatomy"


def test_negative_prompt_compares_weighted_and_unweighted_tags_without_breaking_groups():
    assert remove_positive_duplicates_from_negative(
        "1.5::chromatic aberration::, film grain",
        "1.2::chromatic aberration, film grain::, bad anatomy",
    ) == "bad anatomy"


def test_negative_prompt_deduplicates_same_weight_but_keeps_distinct_weights():
    assert deduplicate_negative_prompt(
        "0.5::blurry::, 1.5::blurry::, 0.5::blurry::, blurry, blurry"
    ) == "0.5::blurry::, 1.5::blurry::, blurry"


def test_negative_prompt_does_not_remove_related_but_different_tags():
    assert remove_positive_duplicates_from_negative(
        "blurry background, no text",
        "blurry, text, bad anatomy",
    ) == "blurry, text, bad anatomy"


def test_negative_prompt_recovers_after_an_unclosed_attention_block():
    assert remove_positive_duplicates_from_negative(
        "film grain, chromatic aberration",
        "lowres, {{{blurry, film grain, chromatic aberration, huge breasts::",
    ) == "lowres, {{{blurry, huge breasts::"


def test_v5_prompt_normalizes_multi_character_subjects_and_solo():
    prompt = normalize_v5_prompt(
        "train station， solo | 1girl, red hair | 1boy, black coat"
    )

    assert prompt == ("1girl, 1boy, train station | girl, red hair | boy, black coat")


def test_v5_prompt_preserves_flattened_prompt_without_pipe():
    prompt = normalize_v5_prompt(
        "cinematic lighting, girl, red hair, black dress, boy, blue coat, french kiss"
    )

    assert prompt == (
        "cinematic lighting, girl, red hair, black dress, boy, blue coat, french kiss"
    )


def test_v5_prompt_does_not_split_counted_subject_tags_without_character_boundaries():
    prompt = normalize_v5_prompt("2girls, cinematic lighting, red hair, blue hair")

    assert prompt == "2girls, cinematic lighting, red hair, blue hair"


def test_v5_structured_payload_does_not_infer_character_prompts():
    payload = build_v4_prompt(
        "cinematic lighting, girl, red hair, boy, blue coat",
        width=1216,
        height=832,
    )

    assert payload["caption"]["base_caption"] == (
        "cinematic lighting, girl, red hair, boy, blue coat"
    )
    assert payload["caption"]["char_captions"] == []


def test_v5_prompt_removes_local_counts_solo_and_duplicate_subject_tags():
    prompt = normalize_v5_prompt(
        "park | 1girl, solo, red hair, girl | 1boy, solo, black coat"
    )

    assert prompt == "1girl, 1boy, park | girl, red hair | boy, black coat"


def test_v5_prompt_rejects_conflicting_character_subject_types():
    try:
        normalize_v5_prompt("park | 1girl, red hair, 2boys")
    except ValueError as exc:
        assert "互相冲突的主体类型" in str(exc)
    else:
        raise AssertionError("Expected conflicting subject tags to be rejected")


def test_v5_prompt_recomputes_base_subject_counts_from_character_sections():
    prompt = normalize_v5_prompt("1girl, park | girl, red hair | girl, blue hair")

    assert prompt == "2girls, park | girl, red hair | girl, blue hair"


def test_v5_prompt_adds_solo_only_for_one_structured_character():
    prompt = normalize_v5_prompt("street | girl, red hair")

    assert prompt == "1girl, solo, street | girl, red hair"


def test_v5_prompt_rejects_a_character_section_without_subject_type():
    try:
        normalize_v5_prompt("street | red hair, white coat")
    except ValueError as exc:
        assert "必须以 girl、boy 或 other" in str(exc)
    else:
        raise AssertionError("Expected a missing character subject to be rejected")


def test_v5_prompt_preserves_blank_line_prompt_blocks():
    prompt = normalize_v5_prompt(
        "1girl, cafe\n\nCharacter 1 reaches for the cup. | girl, red hair"
    )

    assert prompt == (
        "1girl, solo, cafe\n\nCharacter 1 reaches for the cup. | girl, red hair"
    )


def test_v5_prompt_preserves_randomizer_bars():
    base, characters = split_v5_prompt(
        "2girls, ||day|night|| | girl, red hair | girl, blue hair"
    )

    assert base == "2girls, ||day|night||"
    assert characters == ["girl, red hair", "girl, blue hair"]


def test_v5_structured_prompt_keeps_automatic_centers_without_enabling_coords():
    payload = build_v4_prompt(
        "2girls, side-by-side | girl, red hair | girl, blue hair",
        width=1216,
        height=832,
    )
    characters = payload["caption"]["char_captions"]

    assert payload["use_coords"] is False
    assert [item["char_caption"] for item in characters] == [
        "girl, red hair",
        "girl, blue hair",
    ]
    assert characters[0]["centers"][0]["x"] < characters[1]["centers"][0]["x"]
    assert characters[0]["centers"][0]["y"] == characters[1]["centers"][0]["y"]


def test_v5_negative_prompt_matches_centers_without_enabling_automatic_coords():
    positive = build_v4_prompt(
        "1girl, 1boy | girl, red hair | boy, blue hair",
        width=832,
        height=1216,
    )
    negative = build_v4_negative_prompt(
        "bad anatomy",
        character_count=2,
        width=832,
        height=1216,
    )

    assert negative["use_coords"] is False
    assert negative["use_order"] is False
    assert [item["centers"] for item in negative["caption"]["char_captions"]] == [
        item["centers"] for item in positive["caption"]["char_captions"]
    ]


def test_v5_negative_prompt_uses_aligned_character_captions():
    negative = build_v4_negative_prompt(
        "bad anatomy",
        character_count=2,
        character_captions=["blue hair", "red hair"],
    )

    assert [item["char_caption"] for item in negative["caption"]["char_captions"]] == [
        "blue hair",
        "red hair",
    ]


def test_v5_structured_prompt_uses_scene_planner_positions():
    positions = [{"x": 0.2, "y": 0.8}, {"x": 0.8, "y": 0.2}]
    positive = build_v4_prompt(
        "2girls | girl, red hair | girl, blue hair",
        width=832,
        height=1216,
        character_positions=positions,
    )
    negative = build_v4_negative_prompt(
        "bad anatomy",
        character_count=2,
        width=832,
        height=1216,
        character_positions=positions,
    )

    assert positive["use_coords"] is True
    assert negative["use_coords"] is True
    assert [
        item["centers"][0] for item in positive["caption"]["char_captions"]
    ] == positions
    assert [
        item["centers"][0] for item in negative["caption"]["char_captions"]
    ] == positions


def test_v5_structured_prompt_merges_partial_depth_coordinates():
    positions = [{"y": 0.8}, {"y": 0.2}]
    positive = build_v4_prompt(
        "2girls | girl, red coat | girl, blue coat",
        width=1216,
        height=832,
        character_positions=positions,
    )
    positive_centers = [
        item["centers"][0] for item in positive["caption"]["char_captions"]
    ]
    negative = build_v4_negative_prompt(
        "bad anatomy",
        character_count=2,
        width=1216,
        height=832,
        character_positions=positions,
    )

    assert positive_centers[0]["x"] < positive_centers[1]["x"]
    assert [center["y"] for center in positive_centers] == [0.8, 0.2]
    assert [
        item["centers"][0] for item in negative["caption"]["char_captions"]
    ] == positive_centers


def test_v5_prompt_rejects_more_than_supported_character_sections():
    prompt = "scene | " + " | ".join("girl" for _ in range(23))

    try:
        normalize_v5_prompt(prompt)
    except ValueError as exc:
        assert "最多支持 22 个" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for 23 character prompts")


def test_v5_prompt_preserves_uc_in_a_flattened_prompt():
    prompt = normalize_v5_prompt(
        "cinematic lighting, girl, red hair, boy, blue coat, "
        "--uc blurry, lowres, bad anatomy"
    )

    assert prompt == (
        "cinematic lighting, girl, red hair, boy, blue coat, "
        "--uc blurry, lowres, bad anatomy"
    )


def test_v5_prompt_does_not_promote_a_known_single_character_by_hint():
    prompt = normalize_v5_prompt(
        "artist tags, girl, Phoebe (wuthering waves), official alternate costume, "
        "-1::nude,::, --uc blurry, lowres",
        character_hints=["Phoebe (wuthering waves)", "phoebe_(wuthering_waves)"],
    )

    assert prompt == (
        "artist tags, girl, Phoebe (wuthering waves), official alternate costume, "
        "-1::nude,::, --uc blurry, lowres"
    )


def test_v5_prompt_does_not_promote_explicit_multi_subject_count():
    prompt = normalize_v5_prompt(
        "artist tags, 2girls, Phoebe (wuthering waves), official alternate costume",
        character_hints=["Phoebe (wuthering waves)"],
    )

    assert prompt == (
        "artist tags, 2girls, Phoebe (wuthering waves), official alternate costume"
    )


def test_v5_prompt_preserves_commas_inside_weight_blocks_without_inference():
    prompt = normalize_v5_prompt(
        "1.2::best quality, absurdres::, girl, 1.1::red hair, blue eyes::, "
        "boy, black coat"
    )

    assert prompt == (
        "1.2::best quality, absurdres::, girl, "
        "1.1::red hair, blue eyes::, boy, black coat"
    )
