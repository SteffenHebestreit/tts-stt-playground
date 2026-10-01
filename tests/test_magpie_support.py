"""Unit tests for magpie-tts-service/magpie_support.py and text_splitting.py (no torch, no NeMo).

Every rule here was measured on nvidia/magpie_tts_multilingual_357m with nemo_toolkit
3.0.0; the module docstring of magpie_support.py says what and why.
"""

import ast
import time

import numpy as np
import pytest

from magpie_loader import CHECKPOINT_TOKENIZERS, LANGUAGE_TOKENIZER_MAP, REPO, SERVICE_DIR, load_support

support = load_support()
splitter = support.text_splitting  # the copy magpie_support itself imported


# --- languages ------------------------------------------------------------------------

def test_the_languages_nemo_can_route_are_those_whose_tokenizer_the_checkpoint_has():
    """NeMo's map names tokenizers; a code whose candidates the model lacks would be read with English rules."""
    assert support.supported_languages(LANGUAGE_TOKENIZER_MAP, CHECKPOINT_TOKENIZERS) == [
        "de", "en", "es", "fr", "ja", "zh"]


@pytest.mark.parametrize("code", ["it", "vi", "hi"])
def test_a_language_the_checkpoint_has_a_tokenizer_for_is_still_unsupported_when_nemos_map_cannot_reach_it(code):
    """The checkpoint holds Italian, Vietnamese and Hindi tokenizers under names the 3.0.0 map does not list."""
    assert code in LANGUAGE_TOKENIZER_MAP
    assert code not in support.supported_languages(LANGUAGE_TOKENIZER_MAP, CHECKPOINT_TOKENIZERS)


@pytest.mark.parametrize("value, expected", [
    ("de", "de"), ("DE", "de"), (" de ", "de"), ("de-DE", "de"), ("de_AT", "de"),
    ("German", "de"), ("deutsch", "de"), ("English", "en"), ("en-US", "en"), ("Español", "es"),
])
def test_a_language_is_accepted_as_a_code_a_locale_or_a_name(value, expected):
    assert support.resolve_language(value, "de", ["de", "en", "es"]) == expected


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO"])
def test_blank_and_auto_mean_the_configured_default(value):
    assert support.resolve_language(value, "de", ["de", "en"]) == "de"


@pytest.mark.parametrize("value", ["nl", "Dutch", "xx", "it", "pt-BR"])
def test_an_unsupported_language_is_refused_and_the_message_lists_the_supported_ones(value):
    """NeMo would answer these with the English tokenizer and an ordinary success."""
    with pytest.raises(support.LanguageNotSupported) as exc:
        support.resolve_language(value, "de", ["de", "en", "fr"])
    assert value in str(exc.value)
    assert "de, en, fr" in str(exc.value)
    assert exc.value.supported == ["de", "en", "fr"]


def test_a_default_the_model_cannot_speak_fails_loudly_instead_of_speaking_english():
    with pytest.raises(support.LanguageNotSupported) as exc:
        support.resolve_language("auto", "nl", ["de", "en"])
    assert "nl" in str(exc.value)


# --- speakers -------------------------------------------------------------------------

LABELS = ["Aria", "Jason", "John", "Leo", "Sofia"]


@pytest.mark.parametrize("value, expected", [
    ("Sofia", 4), ("sofia", 4), ("SOFIA", 4), (" Leo ", 3), ("Aria", 0),
    (2, 2), ("2", 2), (" 4 ", 4), (0, 0),
])
def test_a_speaker_is_a_name_in_any_case_or_an_index(value, expected):
    assert support.resolve_speaker(value, LABELS, default_index=1) == expected


@pytest.mark.parametrize("value", [None, "", "  ", "auto", "Auto"])
def test_a_blank_speaker_is_the_default(value):
    assert support.resolve_speaker(value, LABELS, default_index=3) == 3


@pytest.mark.parametrize("value", [
    "Bob", 5, "5", -1, "-1", 99, True, "Sofia2",
    # str.isdigit() is true for these, but int() refuses them: they must be a 400, not a 500.
    "²", "⑤", "1²",
    # int() refuses more than sys.get_int_max_str_digits() (4300) digits.
    pytest.param("9" * 5000, id="5000-digits"),
])
def test_an_unknown_speaker_is_refused_and_the_message_lists_the_valid_ones(value):
    """An out-of-range index is a ValueError inside the model; the caller should be told what is valid."""
    with pytest.raises(support.SpeakerNotFound) as exc:
        support.resolve_speaker(value, LABELS, default_index=0)
    assert "0 = Aria" in str(exc.value) and "4 = Sofia" in str(exc.value)
    assert len(str(exc.value)) < 200, "the message echoes the whole value back"


def test_a_name_of_digit_like_characters_int_cannot_read_is_looked_up_as_a_name():
    """isdecimal(), not isdigit(): "⑤" is no index, so it can only be the name of a speaker."""
    assert support.resolve_speaker("⑤", ["Aria", "⑤"], default_index=0) == 1


def test_speaker_names_are_parsed_in_order_without_duplicates():
    assert support.parse_speakers("Ann, Bob ,ann, Cy") == ("Ann", "Bob", "Cy")
    assert support.parse_speakers(None) == support.DEFAULT_SPEAKERS
    assert support.parse_speakers("  ") == support.DEFAULT_SPEAKERS
    assert support.parse_speakers(" , ,") == support.DEFAULT_SPEAKERS


def test_speaker_names_may_also_be_one_per_line():
    """A YAML block scalar (`MAGPIE_SPEAKERS: |`) gives one name per line and a trailing line break."""
    assert support.parse_speakers("Aria\nJason\nJohn\nLeo\nSofia\n") == support.DEFAULT_SPEAKERS
    assert support.parse_speakers("Aria\r\nJason\r\n") == ("Aria", "Jason")


def test_a_control_character_is_removed_from_a_name_and_the_name_keeps_its_place():
    """The position is the speaker index: dropping the name would move every later speaker to the wrong voice."""
    assert support.parse_speakers("Aria,Ja\x07son,John\x00,\x1bLeo,Sofia\x7f") == support.DEFAULT_SPEAKERS
    assert support.parse_speakers("A,\tB\t,C") == ("A", "B", "C")


def test_labels_follow_the_models_speaker_count():
    names = ("A", "B", "C")
    assert support.speaker_labels(names, 3) == ["A", "B", "C"]
    assert support.speaker_labels(names, 5) == ["A", "B", "C", "Speaker 3", "Speaker 4"]
    assert support.speaker_labels(names, 2) == ["A", "B"]
    assert support.speaker_labels(names, None) == ["A", "B", "C"]


def test_a_default_speaker_that_names_nobody_falls_back_to_the_first(caplog):
    with caplog.at_level("WARNING"):
        assert support.default_speaker_index("Nobody", LABELS) == 0
    assert "MAGPIE_DEFAULT_SPEAKER" in caplog.text
    assert support.default_speaker_index("Leo", LABELS) == 3


@pytest.mark.parametrize("value", ["²", "⑤", pytest.param("9" * 5000, id="5000-digits")])
def test_a_default_speaker_int_cannot_read_falls_back_to_the_first_instead_of_failing_the_import(value, caplog):
    """app.py resolves MAGPIE_DEFAULT_SPEAKER at import: a ValueError there would stop the service."""
    with caplog.at_level("WARNING"):
        assert support.default_speaker_index(value, LABELS) == 0
    assert "MAGPIE_DEFAULT_SPEAKER" in caplog.text


# --- text grouping --------------------------------------------------------------------

def test_german_ordinals_and_abbreviations_do_not_end_a_group():
    """"am 3. Mai", "ca. 5", "z. B." and "Dr. Müller" are not sentence ends: splitting there changes how it is read."""
    text = "Am 3. Mai kostet es ca. 12 Euro, z. B. für Dr. Müller. Danach gibt es Kaffee."
    groups = support.group_text(text, "de", 400)
    assert groups == [text]
    assert support.group_text(text, "de", 60) == [
        "Am 3. Mai kostet es ca. 12 Euro, z. B. für Dr. Müller.", "Danach gibt es Kaffee."]


def test_groups_never_exceed_the_ceiling_even_without_punctuation():
    text = " ".join(["wort"] * 200)
    groups = support.group_text(text, "de", 120)
    assert len(groups) > 1
    assert all(len(g) <= 120 for g in groups)
    assert " ".join(groups) == text


def test_groups_hold_whole_sentences_and_lose_nothing():
    sentences = [f"Das ist der {n} Satz dieses Textes." for n in ("erste", "zweite", "dritte", "vierte", "fünfte")]
    text = " ".join(sentences)
    groups = support.group_text(text, "de", 80)
    assert " ".join(groups) == text
    assert all(g.rstrip().endswith("Textes.") for g in groups)


def test_blank_text_makes_no_groups_and_other_text_always_makes_one():
    assert support.group_text("   \n ", "de", 200) == []
    assert support.group_text("Hallo", "de", 200) == ["Hallo"]
    assert support.group_text("...", "de", 200) == ["..."]
    assert support.group_text("   \n ", "zh", 200) == []
    assert support.group_text("你好", "zh", 200) == ["你好"]


# --- Chinese and Japanese -------------------------------------------------------------
#
# NeMo 3.0.0 splits them only at 。？！… and only above 100 characters / 80 words, so a
# long sentence joined by commas was one chunk, cut off at the decoder's 500 frames.
# group_text(text, "zh"/"ja", 200) groups them at a third of the ceiling.

CJK_CEILING = 66

# One sentence of 173 characters whose clauses are joined by "，" alone.
LONG_COMMA_SENTENCE = "，".join([
    "随着城市化进程的不断加快", "越来越多的年轻人离开家乡来到大城市工作和生活", "他们在追求更好发展机会的同时",
    "也面临着住房成本高涨和通勤时间过长等诸多现实问题", "而这些问题如果长期得不到有效解决",
    "不仅会影响个人的身心健康和生活质量", "还可能对整个社会的稳定与可持续发展产生深远的负面影响",
    "因此政府和企业以及社会各界都应当共同努力", "积极探索切实可行的解决方案",
]) + "。"


@pytest.mark.parametrize("language", ["zh", "ja"])
@pytest.mark.parametrize("text", [
    pytest.param("这是第一句话。这是第二句话。" * 40, id="full-stops"),
    pytest.param(LONG_COMMA_SENTENCE, id="one-sentence-of-commas"),
])
def test_chinese_and_japanese_are_grouped_at_their_own_sentence_and_clause_marks(text, language):
    groups = support.group_text(text, language, 200)

    assert len(groups) > 1
    assert all(len(g) <= CJK_CEILING for g in groups), [len(g) for g in groups]
    assert all(g[-1] in "。，" for g in groups), "a group ends inside a clause"
    assert "".join(groups) == text, "text without spaces comes back whole"


@pytest.mark.parametrize("language", ["zh", "ja"])
@pytest.mark.parametrize("text, expected", [
    pytest.param("他说：「今天天气很好。」我们出去吧，然后去公园散步！",
                 ["他说：「今天天气很好。」", "我们出去吧，然后去公园散步！"], id="corner-brackets"),
    pytest.param("她问：“你明天来不来？”我说：“一定来，不见不散！”",
                 ["她问：“你明天来不来？”", "我说：“一定来，不见不散！”"], id="curly-quotes"),
])
def test_a_closing_quote_stays_with_the_sentence_mark_before_it(text, expected, language):
    assert support.group_text(text, language, 60) == expected


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_a_decimal_point_is_not_a_sentence_end_but_a_full_stop_before_a_space_is(language):
    assert support.group_text("版本3.5很好。" * 3, language, 60) == ["版本3.5很好。版本3.5很好。", "版本3.5很好。"]
    assert support.group_text("This is the end. 这是另一个很长的句子，它有很多的字。", language, 60) == [
        "This is the end.", "这是另一个很长的句子，它有很多的字。"]


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_a_line_break_ends_a_sentence_and_becomes_a_space_inside_a_group(language):
    assert support.group_text("第一行\n第二行\r\n第三行。", language, 200) == ["第一行 第二行 第三行。"]


@pytest.mark.parametrize("language", ["zh", "ja"])
def test_a_run_without_any_mark_is_cut_hard_and_loses_nothing(language):
    text = "字" * 300

    groups = support.group_text(text, language, 200)

    assert [len(g) for g in groups] == [66, 66, 66, 66, 36]
    assert "".join(groups) == text


def test_an_over_long_sentence_is_cut_at_a_space_rather_than_inside_a_latin_word():
    text = "我们使用 Kubernetes 和 Docker 来部署 microservices architecture 以便快速迭代并且保证系统的稳定性和可扩展性"

    groups = support.group_text(text, "zh", 200)

    assert groups[0].endswith("architecture") and len(groups) == 2
    assert " ".join(groups) == text


def test_a_number_is_not_cut_at_its_separator():
    """A clause cut after the ':' of "14:30" or the ',' of "1,000" would have them read as two numbers."""
    text = "价格是1,000元，然后是2,000元，最后是3,000元的东西"
    assert splitter.split_cjk(text, 15) == ["价格是1,000元，", "然后是2,000元，", "最后是3,000元的东西"]


# Sentences longer than the 66-character ceiling whose last clause mark in the cut window
# is the separator inside the number, ASCII or full-width (a Chinese or Japanese IME types
# "14：30" and "１，０００"). Cut there, NeMo's normalizer read "下午14：" and "30请" as two
# numbers with the pause between groups, where the whole "14：30" is a time.
ZH_PRICE = "根据公司最新发布的价格调整通知，从下个月开始，本店所有普通会员卡和高级会员卡的年费将统一上调到{}元并且不再提供任何形式的折扣或者优惠"
ZH_TIME = "根据公司最新发布的会议安排通知，从下个月开始，本部门所有普通员工和高级员工的例会将统一改在每周三下午{}举行并且不再另行通知任何人"
JA_PRICE = "誠に勝手ながら来月より当店の会員カードの年会費は一律で{}円に改定させていただきますのでご了承ください何卒よろしくお願い申し上げます"


@pytest.mark.parametrize("template, language, number", [
    (ZH_PRICE, "zh", "1,000"), (ZH_PRICE, "zh", "1\uff0c000"), (ZH_PRICE, "zh", "\uff11\uff0c\uff10\uff10\uff10"),
    (ZH_TIME, "zh", "14:30"), (ZH_TIME, "zh", "14\uff1a30"), (ZH_TIME, "zh", "\uff11\uff14\uff1a\uff13\uff10"),
    (JA_PRICE, "ja", "1\uff0c000"), (JA_PRICE, "ja", "\uff11\uff0c\uff10\uff10\uff10"),
], ids=["zh-1,000", "zh-1-fw-comma-000", "zh-fullwidth-1000", "zh-14:30", "zh-14-fw-colon-30", "zh-fullwidth-1430",
        "ja-1-fw-comma-000", "ja-fullwidth-1000"])
def test_a_number_with_a_full_width_separator_is_not_cut_either(template, language, number):
    text = template.format(number)
    assert len(text) > CJK_CEILING

    groups = support.group_text(text, language, 200)

    assert len(groups) > 1 and all(len(g) <= CJK_CEILING for g in groups)
    assert any(number in g for g in groups), f"{number!r} was cut: {groups}"
    assert "".join(groups) == text


def test_without_a_clause_mark_to_cut_at_a_sentence_is_cut_before_a_number_not_inside_a_word():
    """The separator inside the price is no place to cut, and there is no other mark: a hard cut fell inside "申し上げます"."""
    text = JA_PRICE.format("1,000")

    groups = support.group_text(text, "ja", 200)

    assert groups == [text[:text.index("1,000")], text[text.index("1,000"):]]


# A timetable joined by nothing but the colons of its times.
TIMETABLE = ("上午9\uff1a30开会10\uff1a02休息10\uff1a15讨论11\uff1a45午饭13\uff1a00出发14\uff1a20到达"
             "15\uff1a30参观16\uff1a30返回17\uff1a30结束18\uff1a00晚饭19\uff1a30散会")


@pytest.mark.parametrize("cut", [
    pytest.param(lambda: support.group_text(TIMETABLE, "zh", 200), id="grouped"),
    pytest.param(lambda: support.split_cut_off_group(TIMETABLE, "zh"), id="cut-off"),
])
def test_a_timetable_is_cut_between_two_times_not_inside_one(cut):
    groups = cut()

    assert len(groups) > 1 and "".join(groups) == TIMETABLE
    for before, after in zip(groups, groups[1:]):
        assert not before[-1].isdigit() and before[-1] != "\uff1a", f"cut inside a time: {before!r} | {after!r}"
        assert after[:1].isdigit(), f"not cut before a time: {after!r}"


@pytest.mark.parametrize("language", ["zh", "ja"])
@pytest.mark.parametrize("text", [
    pytest.param("。" * 5000, id="5000-stops"),
    pytest.param("，" * 5000, id="5000-commas"),
    pytest.param("字" * 5000, id="5000-unmarked"),
])
def test_grouping_chinese_and_japanese_is_linear(text, language):
    """It runs on the event loop, on caller-controlled text of up to MAX_TEXT_CHARS."""
    started = time.perf_counter()
    groups = support.group_text(text, language, 200)
    assert time.perf_counter() - started < 1.0
    assert all(len(g) <= CJK_CEILING for g in groups) and "".join(groups) == text


# --- a group whose generation was cut off ----------------------------------------------

@pytest.mark.parametrize("language, text, halves", [
    ("de", "Das ist der erste Satz. Das ist der zweite Satz.", ["Das ist der erste Satz.", "Das ist der zweite Satz."]),
    ("en", "one two three four five six seven eight nine ten", ["one two three four five", "six seven eight nine ten"]),
    ("zh", "这是第一句话，这是第二句话。这是第三句话，这是第四句话。", ["这是第一句话，这是第二句话。", "这是第三句话，这是第四句话。"]),
    ("ja", "字" * 30, ["字" * 15, "字" * 15]),
])
def test_a_group_of_prose_is_halved_at_its_best_boundaries(language, text, halves):
    assert support.split_cut_off_group(text, language) == halves


@pytest.mark.parametrize("language, text", [("de", "Hallo Welt."), ("zh", "你好。")])
def test_a_short_group_cannot_be_cut(language, text):
    assert support.split_cut_off_group(text, language) == [text]


def test_a_digit_weighs_like_ten_characters():
    """Normalization writes "1234567" out as 83 characters, spoken in about 4.7 s."""
    assert support.spoken_weight("Hallo Welt.") == 11
    assert support.spoken_weight("1234567") == 70
    assert support.spoken_weight("Am 3. Mai") == 9 + 9


# Measured on the real model: 6 of 7 runs of this group ended at 21.7 s with the last
# number or two missing; halving it by characters cut "8.912.345 | Schiffe." and
# "und im Jahr | 2020", and spoke "Schiffe." on its own.
GERMAN_CAPPED = ("Die Stadt hatte im Jahr 1990 genau 1.234.567 Einwohner, im Jahr 2000 schon 2.345.678 Einwohner, "
                 "im Jahr 2010 dann 3.456.789 Einwohner und im Jahr 2020 schließlich 4.567.891 Einwohner.")
GERMAN_CAPPED3 = ("Es gab 1.234.567 Autos, 2.345.678 Busse, 3.456.789 Räder, 4.567.891 Boote, 5.678.912 Züge, "
                  "6.789.123 Flugzeuge, 7.891.234 Roller und 8.912.345 Schiffe.")


def test_a_cut_off_number_list_is_cut_at_its_commas_and_keeps_its_last_noun():
    assert support.split_cut_off_group(GERMAN_CAPPED3, "de") == [
        "Es gab 1.234.567 Autos, 2.345.678 Busse, 3.456.789 Räder,",
        "4.567.891 Boote, 5.678.912 Züge, 6.789.123 Flugzeuge,",
        "7.891.234 Roller und 8.912.345 Schiffe.",
    ]


def test_a_cut_off_sentence_is_cut_before_its_und_rather_than_inside_a_phrase():
    assert support.split_cut_off_group(GERMAN_CAPPED, "de") == [
        "Die Stadt hatte im Jahr 1990 genau 1.234.567 Einwohner,",
        "im Jahr 2000 schon 2.345.678 Einwohner,",
        "im Jahr 2010 dann 3.456.789 Einwohner",
        "und im Jahr 2020 schließlich 4.567.891 Einwohner.",
    ]


@pytest.mark.parametrize("text", [
    pytest.param(GERMAN_CAPPED, id="years-and-numbers"),
    pytest.param(GERMAN_CAPPED3, id="numbers-and-nouns"),
    pytest.param("Die Messwerte waren " + ", ".join(str(1234567 + 1111111 * i) for i in range(17)) + ".", id="17-numbers"),
    pytest.param("Die Kontonummern lauten " + ", ".join(str(1234567890 + 1111111111 * i) for i in range(7)) + " und 1023456789.",
                 id="10-digit-numbers"),
    pytest.param("Die Termine sind am " + ", ".join(f"{d:02d}.{m:02d}.2024" for d, m in zip(range(3, 30, 3), range(1, 12))) + ".",
                 id="dates"),
    pytest.param("The totals were 1,234,567, 2,345,678, 3,456,789, 4,567,891, 5,678,912, 6,789,123 and 7,891,234 units.",
                 id="english"),
])
def test_every_piece_of_a_cut_off_group_can_be_spoken_in_full(text):
    pieces = support.split_cut_off_group(text, "de")

    assert len(pieces) > 1
    assert " ".join(pieces) == text, "text was lost or changed"
    budget = min(support.RESPLIT_WEIGHT, support.spoken_weight(text) // 2)
    assert max(support.spoken_weight(p) for p in pieces) <= budget, [support.spoken_weight(p) for p in pieces]


def test_a_few_words_a_cut_left_at_the_end_join_the_piece_before_them():
    """A piece of a word or two is spoken as an utterance of its own, after a pause."""
    text = "eins zwei drei vier fuenf sechs sieben acht neun zehn elf"

    assert splitter.split_by_weight(text, 50, len) == [text]
    assert splitter.split_by_weight(text, 30, len) == ["eins zwei drei vier fuenf", "sechs sieben acht neun zehn elf"]


def test_a_cut_between_words_does_not_part_a_number_from_its_neighbours():
    """Only "8.912.345" (weight 72) and "Besucher" may not part: the cut moves before the number."""
    pieces = splitter.split_by_weight("Heute kamen 8.912.345 Besucher in die Stadt", 85, support.spoken_weight)

    assert pieces == ["Heute kamen", "8.912.345 Besucher in die Stadt"]


def test_a_word_too_heavy_for_any_piece_is_cut_into_equal_parts():
    pieces = splitter.split_by_weight("Nummer 123456789012345678901234567890 Ende", 120, support.spoken_weight)

    assert pieces == ["Nummer 1234567890", "1234567890", "1234567890 Ende"]


def test_nothing_to_cut_gives_nothing():
    assert splitter.split_by_weight("  \n ", 10, len) == []


# --- the splitter is chatterbox's, and stays linear ---------------------------------------

# What text_splitting.py shares with chatterbox-tts-service/app.py, by name.
SHARED_SPLITTER = (
    "_CLOSERS", "_OPENERS", "_TERMINAL", "_WHITESPACE", "_ABBREVIATIONS", "_AMBIGUOUS_ABBREVIATIONS",
    "_ORDINAL_FOLLOWERS", "_dot_ends_sentence", "_sentences", "_split_sentences",
)


def _definitions(path):
    """Module-level functions (signature and body, docstring left out) and assigned values, as AST dumps by name."""
    found = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef):
            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
            returns = ast.dump(node.returns) if node.returns is not None else ""
            found[node.name] = ast.dump(node.args) + returns + "".join(ast.dump(statement) for statement in body)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = ast.dump(node.value)
    return found


def test_the_sentence_rules_are_chatterboxs_and_not_a_fork_of_them():
    """test_chatterbox_chunking.py pins magpie's copy of the splitter only while its code is chatterbox's.

    The copy exists because each service builds its image from its own directory, not
    because the two may behave differently: a fix to one (an abbreviation, the linear
    back-scan that keeps a 5000-character text from holding the event loop for 20 s)
    belongs in both. Docstrings may differ.
    """
    magpie = _definitions(SERVICE_DIR / "text_splitting.py")
    chatterbox = _definitions(REPO / "chatterbox-tts-service" / "app.py")

    missing = [name for name in SHARED_SPLITTER if name not in magpie or name not in chatterbox]
    assert not missing, f"not defined in both copies: {missing}"
    diverged = [name for name in SHARED_SPLITTER if magpie[name] != chatterbox[name]]
    assert not diverged, (
        f"{diverged} differ between magpie-tts-service/text_splitting.py and chatterbox-tts-service/app.py: "
        "make the same change in both."
    )


@pytest.mark.parametrize("text", [
    pytest.param("1" * 5000, id="digits"),
    pytest.param("1, " * 1666, id="numbers-and-commas"),
    pytest.param("und " * 1250, id="conjunctions"),
    pytest.param("a " * 2500, id="words"),
    pytest.param("12. " * 1250, id="ordinals"),
])
@pytest.mark.parametrize("budget", [10, 250, 100000])
def test_cutting_by_weight_is_linear_on_hostile_input(text, budget):
    started = time.perf_counter()
    pieces = splitter.split_by_weight(text, budget, support.spoken_weight)
    assert time.perf_counter() - started < 3.0
    assert "".join(pieces).replace(" ", "") == text.replace(" ", "")


@pytest.mark.parametrize("language", ["de", "zh"])
@pytest.mark.parametrize("max_chars", [40, 200])
@pytest.mark.parametrize("text", [
    pytest.param("a" * 2400 + " b. c. d. e. " * 200, id="token-then-periods"),  # 5000 characters
    pytest.param("." * 5000, id="periods"),
    pytest.param(".!?;:" * 1000, id="marks"),
    pytest.param("a." + '"' * 4998, id="quotes"),
    pytest.param("12. " * 1250, id="ordinals"),
    pytest.param("z. B. " * 833, id="abbreviations"),
    pytest.param("。" * 5000, id="cjk-stops"),
    pytest.param("，" * 5000, id="cjk-commas"),
])
def test_grouping_is_linear_on_hostile_input(text, max_chars, language):
    """/tts calls group_text on the event loop, with caller-controlled text of up to MAX_TEXT_CHARS (5000)."""
    ceiling = max(20, max_chars // 3) if language == "zh" else max_chars

    started = time.perf_counter()
    groups = support.group_text(text, language, max_chars)

    assert time.perf_counter() - started < 3.0
    assert groups and max(len(g) for g in groups) <= ceiling


# --- audio ----------------------------------------------------------------------------

def test_parts_are_joined_with_the_requested_silence_between_them_only():
    rate = 1000
    joined = support.join_audio([np.ones(100), np.ones(50) * 2, np.ones(10) * 3], rate, gap_ms=150)
    assert joined.dtype == np.float32
    assert joined.size == 100 + 150 + 50 + 150 + 10
    assert np.all(joined[100:250] == 0) and joined[0] == 1 and joined[-1] == 3


def test_empty_parts_are_skipped_and_no_parts_make_no_audio():
    assert support.join_audio([np.zeros(0), np.ones(5), np.zeros(0)], 1000, 150).size == 5
    assert support.join_audio([], 1000, 150).size == 0
    assert support.join_audio([np.zeros(0)], 1000, 150).size == 0


def test_a_zero_gap_joins_the_parts_directly():
    assert support.join_audio([np.ones(3), np.ones(4)], 22050, 0).size == 7


def test_two_dimensional_model_output_is_flattened():
    assert support.join_audio([np.ones((1, 8))], 1000, 0).shape == (8,)


# --- configuration --------------------------------------------------------------------

def test_a_junk_number_costs_the_knob_not_the_service(monkeypatch, caplog):
    monkeypatch.setenv("X_NUM", "banana")
    with caplog.at_level("WARNING"):
        assert support.env_number("X_NUM", 7, cast=int, minimum=1) == 7
    assert "X_NUM" in caplog.text
    monkeypatch.setenv("X_NUM", "0")
    assert support.env_number("X_NUM", 7, cast=int, minimum=1) == 7
    monkeypatch.setenv("X_NUM", "nan")
    assert support.env_number("X_NUM", 7.5, cast=float) == 7.5
    monkeypatch.setenv("X_NUM", "12")
    assert support.env_number("X_NUM", 7, cast=int, minimum=1) == 12
    monkeypatch.delenv("X_NUM")
    assert support.env_number("X_NUM", 7, cast=int) == 7


def test_a_number_above_its_maximum_falls_back_to_the_default_with_a_warning(monkeypatch, caplog):
    """MAGPIE_MAX_GROUP_CHARS above 250 would let NeMo chunk an English group by force again."""
    monkeypatch.setenv("X_NUM", "1000")
    with caplog.at_level("WARNING"):
        assert support.env_number("X_NUM", 200, cast=int, minimum=40, maximum=250) == 200
    assert "X_NUM" in caplog.text and "250" in caplog.text
    monkeypatch.setenv("X_NUM", "250")
    assert support.env_number("X_NUM", 200, cast=int, minimum=40, maximum=250) == 250
    monkeypatch.setenv("X_NUM", "39")
    assert support.env_number("X_NUM", 200, cast=int, minimum=40, maximum=250) == 200


@pytest.mark.parametrize("raw, expected", [("1", True), ("true", True), (" YES ", True), ("on", True),
                                           ("0", False), ("false", False), ("nope", False)])
def test_flags(monkeypatch, raw, expected):
    monkeypatch.setenv("X_FLAG", raw)
    assert support.env_flag("X_FLAG", default=not expected) is expected


def test_an_unset_or_blank_flag_is_its_default(monkeypatch):
    monkeypatch.delenv("X_FLAG", raising=False)
    assert support.env_flag("X_FLAG", True) is True
    monkeypatch.setenv("X_FLAG", "  ")
    assert support.env_flag("X_FLAG", True) is True and support.env_flag("X_FLAG", False) is False


def test_a_model_path_is_shown_as_its_file_name_only():
    assert support.public_model_name("/data/models/magpie.nemo") == "magpie.nemo"
    assert support.public_model_name("nvidia/magpie_tts_multilingual_357m") == "nvidia/magpie_tts_multilingual_357m"
    assert support.public_model_name("") == ""


def test_cuda_out_of_memory_is_recognised_by_type_name_and_by_message():
    class OutOfMemoryError(RuntimeError):
        pass

    assert support.is_cuda_oom(OutOfMemoryError("x"))
    assert support.is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert not support.is_cuda_oom(RuntimeError("shape mismatch"))
