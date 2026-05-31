from __future__ import annotations

from pathlib import Path

import pytest

from game.board import BLACK, WHITE
from records.metadata import (
    RecordMetadata,
    build_record_filename,
    parse_record_filename,
    parse_record_metadata,
    parse_standard_record,
)


STANDARD_RECORD = (
    "{[C5][先手参赛队 B][后手参赛队 W][先手胜]"
    "[2017.07.29 14:00 重庆][2017 CCGC];"
    "B(J,10)MARK[1];W(L,10);B(J,11);W(L,12);B(H,10);W(H,8);B(K,8)}"
)


def test_parse_record_metadata_from_standard_header():
    metadata = parse_record_metadata(STANDARD_RECORD)

    assert metadata.game_type == "C5"
    assert metadata.black_team == "先手参赛队 B"
    assert metadata.white_team == "后手参赛队 W"
    assert metadata.result == "先手胜"
    assert metadata.datetime_location == "2017.07.29 14:00 重庆"
    assert metadata.event_name == "2017 CCGC"
    assert metadata.raw_items == (
        "C5",
        "先手参赛队 B",
        "后手参赛队 W",
        "先手胜",
        "2017.07.29 14:00 重庆",
        "2017 CCGC",
    )


def test_parse_standard_record_returns_metadata_and_moves():
    parsed = parse_standard_record(STANDARD_RECORD)

    assert parsed.metadata.game_type == "C5"
    assert [move.color for move in parsed.moves[:3]] == [BLACK, WHITE, BLACK]
    assert [move.coord for move in parsed.moves] == [
        "J10",
        "L10",
        "J11",
        "L12",
        "H10",
        "H8",
        "K8",
    ]
    assert parsed.moves[0].mark == "1"


def test_parse_standard_record_enforces_mark_range():
    with pytest.raises(ValueError, match="MARK"):
        parse_standard_record("{[C5];B(J,10)MARK[3];W(H,8)}")


def test_export_standard_record_round_trip():
    from records.exporter import export_standard_record

    parsed = parse_standard_record(STANDARD_RECORD)
    exported = export_standard_record(parsed.moves, parsed.metadata)
    reparsed = parse_standard_record(exported)

    assert reparsed.metadata == parsed.metadata
    assert [(m.color, m.coord, m.mark) for m in reparsed.moves] == [
        (m.color, m.coord, m.mark) for m in parsed.moves
    ]
    assert exported.startswith("{[C5][先手参赛队 B]")


def test_record_filename_build_and_parse():
    metadata = RecordMetadata(
        game_type="C5",
        black_team="先手参赛队 B",
        white_team="后手参赛队 W",
        result="先手胜",
        datetime_location="2017.07.29 14:00 重庆",
        event_name="2017 CCGC",
    )

    filename = build_record_filename(metadata)
    assert filename.endswith(".txt")
    assert filename.startswith("C5-先手参赛队 B vs 后手参赛队 W-先手胜-")

    parsed = parse_record_filename(filename)
    assert parsed.game_type == metadata.game_type
    assert parsed.black_team == metadata.black_team
    assert parsed.white_team == metadata.white_team
    assert parsed.result == metadata.result
    assert parsed.datetime_location == metadata.datetime_location
    assert parsed.event_name == metadata.event_name


def test_record_file_io_supports_gb2312(tmp_path: Path):
    from records.file_io import read_record_file, write_record_file

    path = tmp_path / "record.txt"
    write_record_file(path, STANDARD_RECORD, encoding="gb2312")

    text = read_record_file(path, encoding="gb2312")
    assert text == STANDARD_RECORD
