import json

from data.plugins.astrbot_plugin_novelai.scene_prompt import (
    SCENE_PLANNER_SYSTEM_PROMPT,
    build_scene_planner_request,
    collect_preset_bindings,
    compile_scene_plan,
    format_compiled_prompt_structure,
    expand_bindings_for_fallback,
    parse_scene_plan,
    protect_manual_prompt_fragments,
    restore_manual_prompt_fragments,
)

CHARACTER_PRESETS = {
    "达妮娅": "1girl, daniya, blonde hair, blue eyes",
    "爱弥斯": "1girl, amis, pink hair, green eyes",
    "西格莉卡": "1girl, sigilica, silver hair, red eyes",
    "莫宁": "1girl, moning, black hair, purple eyes",
}
CHARACTER_CATEGORIES = {"人物": list(CHARACTER_PRESETS)}


def test_collect_preset_bindings_protects_names_inside_natural_language():
    protected, bindings = collect_preset_bindings(
        "达妮娅在吹泡泡，爱弥斯在其中一个泡泡里",
        CHARACTER_PRESETS,
        CHARACTER_CATEGORIES,
        {},
    )

    assert "达妮娅" not in protected
    assert "爱弥斯" not in protected
    assert {item["name"] for item in bindings} == {"达妮娅", "爱弥斯"}
    assert all(item["kind"] == "character" for item in bindings)
    assert all(item["ref"] in protected for item in bindings)


def test_manual_weight_fragments_are_hidden_and_restored_verbatim():
    prompt = (
        "1.2::artist alpha::, three girls in a garden, "
        "1.35::artist beta, artist gamma::, denia and chisa are smiling"
    )

    semantic_prompt, fragments = protect_manual_prompt_fragments(prompt)

    assert "artist alpha" not in semantic_prompt
    assert "artist beta" not in semantic_prompt
    assert "three girls in a garden" in semantic_prompt
    assert len(fragments) == 2
    compiled = restore_manual_prompt_fragments(
        "3girls, garden | girl, denia | girl, chisa",
        fragments,
    )
    assert "1.2::artist alpha::" in compiled
    assert "1.35::artist beta, artist gamma::" in compiled
    assert compiled.count("1.2::artist alpha::") == 1


def test_scene_planner_request_does_not_expose_locked_raw_syntax():
    prompt = (
        "1.2::artist alpha::, three girls in a garden, "
        "1.35::artist beta::, denia and chisa"
    )
    semantic_prompt, fragments = protect_manual_prompt_fragments(prompt)
    request = json.loads(
        build_scene_planner_request(
            semantic_prompt,
            semantic_prompt,
            [],
            fragments,
        )
    )
    assert all("raw" not in item for item in request["locked_fragments"])
    assert len(request["locked_fragments"]) == 2


def test_scene_planner_uses_english_visual_tags_and_preserves_visible_text():
    assert "Convert drawable descriptive fields to concise English NovelAI visual tags" in (
        SCENE_PLANNER_SYSTEM_PROMPT
    )
    assert "NEVER translate visible text" in SCENE_PLANNER_SYSTEM_PROMPT
    assert 'text:"..."' in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Do not add a setting, prop, character" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Protected preset content is restored verbatim" in SCENE_PLANNER_SYSTEM_PROMPT


def test_collect_preset_bindings_matches_english_names_case_insensitively():
    protected, bindings = collect_preset_bindings(
        "ALICE holding a flower",
        {"Alice": "1girl, alice, blonde hair"},
        {"人物": ["Alice"]},
        {},
    )

    assert protected == "PRESET_1 holding a flower"
    assert bindings[0]["name"] == "Alice"


def test_collect_preset_bindings_uses_longest_non_overlapping_character_name():
    protected, bindings = collect_preset_bindings(
        "达妮娅",
        {
            "达妮娅": "girl, denia_(wuthering_waves)",
            "妮娅": "girl, nia_(xenoblade)",
        },
        {"人物": ["达妮娅", "妮娅"]},
        {},
    )

    assert protected == "PRESET_1"
    assert [item["name"] for item in bindings] == ["达妮娅"]


def test_collect_preset_bindings_keeps_standalone_shorter_character_name():
    protected, bindings = collect_preset_bindings(
        "达妮娅和妮娅",
        {
            "达妮娅": "girl, denia_(wuthering_waves)",
            "妮娅": "girl, nia_(xenoblade)",
        },
        {"人物": ["达妮娅", "妮娅"]},
        {},
    )

    assert protected == "PRESET_1和PRESET_2"
    assert [item["name"] for item in bindings] == ["达妮娅", "妮娅"]


def test_scene_compiler_deduplicates_identity_repeated_by_planner():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "1girl, solo, beach",
                "entities": [
                    {
                        "id": "phoebe",
                        "kind": "character",
                        "subject": "girl",
                        "preset_refs": ["PRESET_1"],
                        "prompt": "girl, phoebe_(wuthering_waves), white dress",
                    }
                ],
            }
        ),
        {"PRESET_1"},
    )
    bindings = [
        {
            "ref": "PRESET_1",
            "name": "菲比",
            "content": "girl, phoebe_(wuthering_waves)",
            "category": "人物",
            "kind": "character",
        }
    ]

    prompt, _, _, _ = compile_scene_plan(plan, bindings, 6)

    assert prompt.count("phoebe_(wuthering_waves)") == 1
    assert "white dress" in prompt


def test_bubble_scene_compiles_presets_relationships_and_positions():
    source = (
        "达妮娅在吹泡泡，有好几个泡泡，几个泡泡里分别有爱弥斯、"
        "西格莉卡、莫宁，达妮娅也是Q版，达妮娅在角落里，四个人位置基本均匀"
    )
    protected, bindings = collect_preset_bindings(
        source,
        CHARACTER_PRESETS,
        CHARACTER_CATEGORIES,
        {},
    )
    refs = {item["name"]: item["ref"] for item in bindings}
    raw_plan = {
        "global_prompt": (
            "chibi, several soap bubbles, balanced four-subject composition"
        ),
        "global_preset_refs": [],
        "entities": [
            {
                "id": "daniya",
                "kind": "character",
                "preset_refs": [refs["达妮娅"]],
                "subject": "girl",
                "prompt": "chibi, blowing soap bubbles",
                "position": "bottom-left",
                "depth": "foreground",
                "size": "small",
            },
            {
                "id": "amis",
                "kind": "character",
                "preset_refs": [refs["爱弥斯"]],
                "subject": "girl",
                "prompt": "chibi, inside a separate soap bubble",
                "position": "top-left",
                "depth": "midground",
                "size": "small",
            },
            {
                "id": "sigilica",
                "kind": "character",
                "preset_refs": [refs["西格莉卡"]],
                "subject": "girl",
                "prompt": "chibi, inside a separate soap bubble",
                "position": "top-right",
                "depth": "midground",
                "size": "small",
            },
            {
                "id": "moning",
                "kind": "character",
                "preset_refs": [refs["莫宁"]],
                "subject": "girl",
                "prompt": "chibi, inside a separate soap bubble",
                "position": "bottom-right",
                "depth": "midground",
                "size": "small",
            },
        ],
        "relationships": [
            {
                "subject": "amis",
                "relation": "inside",
                "object": "bubble_1",
                "prompt": "amis character enclosed inside her own soap bubble",
            }
        ],
        "rendered_text": [],
        "negative_prompt": "photorealistic",
    }
    request = json.loads(build_scene_planner_request(protected, source, bindings))
    plan = parse_scene_plan(json.dumps(raw_plan), set(refs.values()))
    prompt, positions, negative, character_negatives = compile_scene_plan(
        plan, bindings, 22
    )

    assert request["user_prompt"] == protected
    assert all(name not in protected for name in CHARACTER_PRESETS)
    assert all(content in prompt for content in CHARACTER_PRESETS.values())
    assert prompt.count(" | ") == 4
    assert "blowing soap bubbles" in prompt
    assert "inside a separate soap bubble" in prompt
    assert positions == [
        {"x": 0.2, "y": 0.8},
        {"x": 0.2, "y": 0.2},
        {"x": 0.8, "y": 0.2},
        {"x": 0.8, "y": 0.8},
    ]
    assert negative == "photorealistic"
    assert character_negatives == ["", "", "", ""]


def test_scene_plan_supports_general_non_character_scenes():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "a red teapot on a wooden table, warm window light",
                "entities": [
                    {
                        "id": "teapot",
                        "kind": "object",
                        "prompt": "red ceramic teapot, centered",
                        "position": "center",
                        "depth": "foreground",
                        "size": "large",
                    }
                ],
                "relationships": [],
                "rendered_text": [],
                "negative_prompt": "people",
            }
        ),
        set(),
    )

    prompt, positions, negative, character_negatives = compile_scene_plan(plan, [], 22)

    assert "red ceramic teapot" in prompt
    assert " | " not in prompt
    assert positions == []
    assert negative == "people"
    assert character_negatives == []


def test_fallback_expands_embedded_presets_without_translating_their_names():
    source = "达妮娅在吹泡泡"
    _, bindings = collect_preset_bindings(
        source,
        CHARACTER_PRESETS,
        CHARACTER_CATEGORIES,
        {},
    )

    prompt = expand_bindings_for_fallback(source, bindings)

    assert "达妮娅" not in prompt
    assert CHARACTER_PRESETS["达妮娅"] in prompt


def test_compiled_prompt_structure_is_loggable():
    structure = format_compiled_prompt_structure(
        "1girl, ||red|blue|| hair | girl, alice"
    )

    assert structure == {
        "base_prompt": "1girl, ||red|blue|| hair",
        "character_prompts": ["girl, alice"],
    }


def test_scene_planner_policy_uses_nai5_tag_and_sentence_roles():
    assert "Plan in this order: memorable visual event" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Use concise tags for discrete facts" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Use natural-language sentences for relationships" in (
        SCENE_PLANNER_SYSTEM_PROMPT
    )
    assert "source#action" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "relationship.subject_tags and object_tags" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Use layout only when the user requests a comic" in (
        SCENE_PLANNER_SYSTEM_PROMPT
    )
    assert "Put only identity, appearance, clothing, and accessories" in (
        SCENE_PLANNER_SYSTEM_PROMPT
    )
    assert "For a named copyrighted character" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "For an original character" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Never write a manual Text: block" in SCENE_PLANNER_SYSTEM_PROMPT
    assert "Do not add generic quality tags" in SCENE_PLANNER_SYSTEM_PROMPT


def test_scene_plan_compiles_character_uc_and_visible_text_position():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "2girls, cafe, cowboy shot",
                "entities": [
                    {
                        "id": "left_girl",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "red hair",
                        "negative_prompt": "blue hair",
                        "position": "left",
                    },
                    {
                        "id": "right_girl",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "blue hair",
                        "negative_prompt": "red hair",
                        "position": "right",
                    },
                ],
                "relationships": [],
                "rendered_text": [
                    {
                        "content": "欢迎",
                        "carrier": "wooden menu board",
                        "position": "behind the counter",
                    }
                ],
                "negative_prompt": "photorealistic",
            },
            ensure_ascii=False,
        ),
        set(),
    )

    prompt, positions, negative, character_negatives = compile_scene_plan(plan, [], 22)
    base_prompt = prompt.split(" | ", 1)[0]

    assert 'A wooden menu board behind the counter reads "欢迎"' in base_prompt
    assert "english text" not in base_prompt
    assert "Text:" not in base_prompt
    assert positions == [{"x": 0.2, "y": 0.5}, {"x": 0.8, "y": 0.5}]
    assert negative == "photorealistic"
    assert character_negatives == ["blue hair", "red hair"]


def test_scene_plan_preserves_two_identical_character_captions():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "2girls, twins",
                "entities": [
                    {
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "school uniform",
                    },
                    {
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "school uniform",
                    },
                ],
            }
        ),
        set(),
    )

    prompt, positions, _, character_negatives = compile_scene_plan(plan, [], 22)

    assert prompt.count(" | girl, school uniform") == 2
    assert positions == [None, None]
    assert character_negatives == ["", ""]


def test_scene_plan_compiles_ascii_visible_text_with_its_carrier():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "menu board",
                "rendered_text": [
                    {
                        "content": "Welcome",
                        "carrier": "menu board",
                        "style": "hand-lettered",
                        "position": "in the center",
                    }
                ],
            }
        ),
        set(),
    )

    prompt, _, _, _ = compile_scene_plan(plan, [], 22)

    assert 'A hand-lettered menu board in the center reads "Welcome"' in prompt
    assert "Text:" not in prompt


def test_scene_plan_keeps_local_visual_facts_in_character_captions():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "2girls, indoors",
                "entities": [
                    {
                        "id": "observer",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "sigrika (wuthering waves)",
                        "pose_tags": ["hands on glass"],
                        "expression_tags": ["looking inside"],
                    },
                    {
                        "id": "sleeping_girl",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "aemeath (wuthering waves)",
                        "pose_tags": ["lying on bed", "sleeping"],
                        "expression_tags": ["closed eyes"],
                    },
                ],
                "rendered_text": [
                    {"content": "中文招牌", "carrier": "sign"},
                ],
            }
        ),
        set(),
    )

    prompt, _, _, _ = compile_scene_plan(plan, [], 22)
    base_prompt, observer_prompt, sleeping_prompt = prompt.split(" | ")

    assert "hands on glass" in observer_prompt
    assert "looking inside" in observer_prompt
    assert "lying on bed" in sleeping_prompt
    assert "closed eyes" in sleeping_prompt
    assert "hands on glass" not in base_prompt
    assert '"中文招牌"' in base_prompt


def test_scene_plan_binds_structured_action_roles_and_character_numbers():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "2girls, cafe, cowboy shot",
                "entities": [
                    {
                        "id": "giver",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "red hair",
                        "pose_tags": ["standing"],
                        "expression_tags": ["smile"],
                        "action_prompt": "giver holds a cup toward receiver",
                        "position": "left",
                        "depth": "foreground",
                        "size": "large",
                    },
                    {
                        "id": "receiver",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "blue hair",
                        "pose_tags": ["sitting"],
                        "action_prompt": "receiver reaches for the cup",
                        "position": "right",
                        "depth": "background",
                        "size": "small",
                    },
                ],
                "relationships": [
                    {
                        "subject": "giver",
                        "object": "receiver",
                        "relation": "interacting-with",
                        "subject_tags": [
                            "source#handing over",
                            "target#invalid direction",
                        ],
                        "object_tags": [
                            "target#receiving",
                            "source#invalid direction",
                        ],
                        "prompt": "giver hands a cup to receiver",
                    }
                ],
            }
        ),
        set(),
    )

    prompt, _, _, _ = compile_scene_plan(plan, [], 22)
    base_prompt, giver_prompt, receiver_prompt = prompt.split(" | ")

    assert "Character 1 hands a cup to Character 2" in base_prompt
    assert "Character 1 holds a cup toward Character 2" in giver_prompt
    assert "standing" in giver_prompt
    assert "smile" in giver_prompt
    assert "source#handing over" in giver_prompt
    assert "target#invalid direction" not in giver_prompt
    assert "target#receiving" in receiver_prompt
    assert "source#invalid direction" not in receiver_prompt
    assert "sitting" in receiver_prompt
    assert "Character 2 reaches for the cup" in receiver_prompt
    assert "left" not in giver_prompt
    assert "foreground" not in giver_prompt
    assert "large" not in giver_prompt


def test_scene_plan_compiles_comic_layout_panels_in_order():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "1girl, comic, chibi",
                "layout": {
                    "prompt": "A two-panel comic with equal panels and white borders",
                    "panels": [
                        {
                            "position": "left",
                            "prompt": "she notices an alarm clock",
                            "rendered_text": [
                                {
                                    "content": "Wake up!",
                                    "carrier": "speech bubble",
                                    "style": "jagged",
                                    "position": "above the girl",
                                }
                            ],
                        },
                        {
                            "position": "right",
                            "prompt": "the chair is empty and toast hangs in mid-air",
                            "no_text": True,
                        },
                    ],
                    "continuity_prompt": "The same room and color palette continue across both panels",
                },
            }
        ),
        set(),
    )

    prompt, _, _, _ = compile_scene_plan(plan, [], 22)

    assert "A two-panel comic with equal panels and white borders" in prompt
    assert "Panel 1 (left): she notices an alarm clock;" in prompt
    assert 'A jagged speech bubble above the girl reads "Wake up!"' in prompt
    assert "Panel 2 (right): the chair is empty and toast hangs in mid-air;" in prompt
    assert "No text in this panel" in prompt
    assert "Text:" not in prompt
    assert "The same room and color palette continue across both panels" in prompt
    assert prompt.index("Panel 1") < prompt.index("Panel 2")


def test_scene_plan_exposes_depth_as_partial_character_coordinates():
    plan = parse_scene_plan(
        json.dumps(
            {
                "global_prompt": "2girls, street",
                "entities": [
                    {
                        "id": "near",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "red coat",
                        "position": "auto",
                        "depth": "foreground",
                    },
                    {
                        "id": "far",
                        "kind": "character",
                        "subject": "girl",
                        "prompt": "blue coat",
                        "position": "auto",
                        "depth": "background",
                    },
                ],
            }
        ),
        set(),
    )

    _, positions, _, _ = compile_scene_plan(plan, [], 22)

    assert positions == [{"y": 0.8}, {"y": 0.2}]


def test_scene_plan_rejects_duplicate_entity_ids():
    raw_plan = {
        "entities": [
            {"id": "same", "kind": "character", "subject": "girl"},
            {"id": "same", "kind": "character", "subject": "girl"},
        ]
    }

    try:
        parse_scene_plan(json.dumps(raw_plan), set())
    except ValueError as exc:
        assert "重复实体 ID" in str(exc)
    else:
        raise AssertionError("Expected duplicate entity IDs to be rejected")


def test_scene_plan_rejects_global_text_when_panels_exist():
    raw_plan = {
        "layout": {
            "panels": [{"position": "left", "prompt": "a quiet room"}],
        },
        "rendered_text": [{"content": "Hello", "carrier": "speech bubble"}],
    }

    try:
        parse_scene_plan(json.dumps(raw_plan), set())
    except ValueError as exc:
        assert "必须绑定到具体分格" in str(exc)
    else:
        raise AssertionError("Expected global comic text to be rejected")


def test_scene_plan_rejects_text_inside_a_no_text_panel():
    raw_plan = {
        "layout": {
            "panels": [
                {
                    "position": "left",
                    "no_text": True,
                    "rendered_text": [{"content": "Hello", "carrier": "speech bubble"}],
                }
            ]
        }
    }

    try:
        parse_scene_plan(json.dumps(raw_plan), set())
    except ValueError as exc:
        assert "不能同时声明" in str(exc)
    else:
        raise AssertionError("Expected conflicting panel text policy to be rejected")


def test_manual_weight_fragments_keep_independent_semantic_gaps_visible():
    prompt = "1.2::artist::, masterpiece, 1.3::style::, 3girls, red hair"

    semantic_prompt, fragments = protect_manual_prompt_fragments(prompt)

    assert semantic_prompt == "masterpiece, 3girls, red hair"
    assert [item["raw"] for item in fragments] == [
        "1.2::artist::",
        "1.3::style::",
    ]


def test_fallback_binding_expansion_does_not_touch_weight_fragments():
    bindings = [
        {
            "ref": "PRESET_1",
            "name": "Alice",
            "content": "girl, alice_(wonderland)",
            "category": "人物",
            "kind": "character",
        }
    ]

    expanded = expand_bindings_for_fallback(
        "1.2::Alice::, Alice in a garden",
        bindings,
    )

    assert expanded == "1.2::Alice::, girl, alice_(wonderland) in a garden"
