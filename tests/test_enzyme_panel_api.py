"""酶组合规划接口测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

# 42bp：四种酶各一个位点，其间用 6 个 A 隔开；XbaI(TCTAGA) 在序列上无位点
SEQUENCE = "GGATCC" + "A" * 6 + "GAATTC" + "A" * 6 + "GATATC" + "A" * 6 + "AAGCTT"

ENZYMES = {
    "BamHI": {"name": "BamHI", "pattern": "GGATCC", "cut_top": 1, "cut_bottom": 5},
    "EcoRI": {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5},
    "EcoRV": {"name": "EcoRV", "pattern": "GATATC", "cut_top": 3, "cut_bottom": 3},
    "HindIII": {"name": "HindIII", "pattern": "AAGCTT", "cut_top": 1, "cut_bottom": 5},
    "XbaI": {"name": "XbaI", "pattern": "TCTAGA", "cut_top": 1, "cut_bottom": 5},
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, topology="linear"):
    resp = client.post(
        "/digests",
        json={
            "sequence": SEQUENCE,
            "topology": topology,
            "enzymes": [ENZYMES["BamHI"]],
        },
    )
    assert resp.status_code == 201
    return resp.json()


def _plan(client, digest_id, target, names):
    return client.post(
        f"/digests/{digest_id}/enzyme-panel-plan",
        json={"target_fragments": target, "candidates": [ENZYMES[n] for n in names]},
    )


def test_minimal_combination_linear(client):
    body = _create(client)
    # 单个切割酶即可达到 2 个片段，不应返回两种酶的组合
    resp = _plan(
        client,
        body["id"],
        2,
        ["HindIII", "XbaI", "EcoRV", "EcoRI", "BamHI"],
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["id"] == body["id"]
    assert data["target_fragments"] == 2
    assert data["enzyme_count"] == 1
    assert data["enzymes"] == [ENZYMES["BamHI"]]
    assert data["fragment_count"] == 2
    assert [(c["top_cut"], c["bottom_cut"]) for c in data["cuts"]] == [(1, 5)]


def test_target_three_requires_two_enzymes_and_tie_breaks_by_name(client):
    body = _create(client)
    # 无单酶产生 3 个片段；两酶组合中 (BamHI, EcoRI) 名字典序最小
    resp = _plan(
        client,
        body["id"],
        3,
        ["HindIII", "XbaI", "EcoRV", "EcoRI", "BamHI"],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [e["name"] for e in data["enzymes"]] == ["BamHI", "EcoRI"]
    assert data["enzyme_count"] == 2
    assert data["fragment_count"] == 3
    assert [(c["top_cut"], c["bottom_cut"]) for c in data["cuts"]] == [
        (1, 5),
        (13, 17),
    ]


def test_targets_four_and_five(client):
    body = _create(client)
    resp4 = _plan(
        client, body["id"], 4, ["BamHI", "EcoRI", "EcoRV", "HindIII", "XbaI"]
    )
    assert resp4.status_code == 200
    assert [e["name"] for e in resp4.json()["enzymes"]] == [
        "BamHI",
        "EcoRI",
        "EcoRV",
    ]
    assert len(resp4.json()["cuts"]) == 3

    resp5 = _plan(
        client, body["id"], 5, ["BamHI", "EcoRI", "EcoRV", "HindIII", "XbaI"]
    )
    assert resp5.status_code == 200
    assert [e["name"] for e in resp5.json()["enzymes"]] == [
        "BamHI",
        "EcoRI",
        "EcoRV",
        "HindIII",
    ]
    assert resp5.json()["fragment_count"] == 5


def test_non_cutting_enzyme_satisfies_target_one(client):
    body = _create(client)
    # 无切口 -> 线性分子 1 条完整片段；XbaI 为候选中唯一无位点的酶
    resp = _plan(client, body["id"], 1, ["XbaI", "BamHI"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["enzymes"] == [ENZYMES["XbaI"]]
    assert data["cuts"] == []
    assert data["fragment_count"] == 1


def test_no_solution_returns_422(client):
    body = _create(client)
    # 线性序列最多 5 个片段（4 个内部切点），目标 6 无解
    resp = _plan(
        client, body["id"], 6, ["BamHI", "EcoRI", "EcoRV", "HindIII", "XbaI"]
    )
    assert resp.status_code == 422
    assert "6" in resp.text


def test_circular_topic_uses_record_topology(client):
    # 环形：单个切口开环仍是 1 个片段；无位点的 XbaI 不应胜出
    body = _create(client, topology="circular")
    resp = _plan(
        client, body["id"], 1, ["HindIII", "XbaI", "EcoRV", "EcoRI", "BamHI"]
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [e["name"] for e in data["enzymes"]] == ["BamHI"]
    assert data["fragment_count"] == 1

    resp2 = _plan(
        client,
        body["id"],
        5,
        ["BamHI", "EcoRI", "EcoRV", "HindIII", "XbaI"],
    )
    # 环形 4 个切口至多 4 个片段
    assert resp2.status_code == 422


def test_missing_record_returns_404(client):
    resp = _plan(client, 999, 1, ["XbaI"])
    assert resp.status_code == 404


def test_plan_does_not_persist_or_mutate_record(client):
    body = _create(client)
    _plan(
        client,
        body["id"],
        3,
        ["BamHI", "EcoRI", "EcoRV", "HindIII", "XbaI"],
    )
    # 原记录内容不变，且没有新增任何提交
    stored = client.get(f"/digests/{body['id']}").json()
    assert stored == body
    assert len(client.get("/digests").json()) == 1


def test_result_independent_of_candidate_order(client):
    body = _create(client)
    names = ["HindIII", "XbaI", "EcoRV", "EcoRI", "BamHI"]
    r1 = _plan(client, body["id"], 3, names).json()
    r2 = _plan(client, body["id"], 3, list(reversed(names))).json()
    assert r1 == r2
    assert [e["name"] for e in r1["enzymes"]] == ["BamHI", "EcoRI"]


def test_deduplicated_cut_positions_count_once(client):
    # 两种酶定义不同但物理切口完全相同（同模式同偏移、异名）：
    # 任一单酶都只产生 1 个有效切点 -> 2 个片段；同时选择两者也不会
    # 多产生片段（物理切口去重），因此最优解只保留一个酶。
    body = _create(client)
    alias = {"name": "BamHI-Alias", "pattern": "GGATCC", "cut_top": 1, "cut_bottom": 5}
    resp = client.post(
        f"/digests/{body['id']}/enzyme-panel-plan",
        json={
            "target_fragments": 2,
            "candidates": [ENZYMES["EcoRI"], alias],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # BamHI-Alias 字典序先于 EcoRI，且单酶即可达标
    assert [e["name"] for e in data["enzymes"]] == ["BamHI-Alias"]
    assert len(data["cuts"]) == 1
    assert data["fragment_count"] == 2

    # 显式校验"同物理切口去重"：两种酶的酶切事件合并为一条切口记录
    from app.digest import run_digest
    from app.schemas import EnzymeIn

    dup = run_digest(
        SEQUENCE,
        "linear",
        [EnzymeIn(**ENZYMES["BamHI"]), EnzymeIn(**alias)],
    )
    assert len(dup["cuts"]) == 1
    assert dup["cuts"][0]["enzymes"] == ["BamHI", "BamHI-Alias"]
    assert len(dup["fragments"]) == 2


def test_duplicate_candidate_names_return_422(client):
    body = _create(client)
    resp = client.post(
        f"/digests/{body['id']}/enzyme-panel-plan",
        json={
            "target_fragments": 2,
            "candidates": [ENZYMES["BamHI"], ENZYMES["BamHI"]],
        },
    )
    assert resp.status_code == 422
    assert "duplicate" in resp.text.lower()


@pytest.mark.parametrize(
    "target,candidates",
    [
        (0, ["XbaI"]),          # 目标片段数小于 1
        (-1, ["XbaI"]),
        (2, []),                # 候选为空
    ],
)
def test_invalid_request_returns_422(client, target, candidates):
    body = _create(client)
    resp = client.post(
        f"/digests/{body['id']}/enzyme-panel-plan",
        json={"target_fragments": target, "candidates": candidates},
    )
    assert resp.status_code == 422


def test_more_than_eight_candidates_returns_422(client):
    body = _create(client)
    resp = client.post(
        f"/digests/{body['id']}/enzyme-panel-plan",
        json={
            "target_fragments": 1,
            "candidates": [
                {"name": f"E{i}", "pattern": "AAAA", "cut_top": 2, "cut_bottom": 2}
                for i in range(9)
            ],
        },
    )
    assert resp.status_code == 422


def test_illegal_offsets_and_bases_return_422(client):
    body = _create(client)
    resp = client.post(
        f"/digests/{body['id']}/enzyme-panel-plan",
        json={
            "target_fragments": 1,
            "candidates": [
                {"name": "Bad", "pattern": "GAATTC", "cut_top": 9, "cut_bottom": 5}
            ],
        },
    )
    assert resp.status_code == 422

    resp2 = client.post(
        f"/digests/{body['id']}/enzyme-panel-plan",
        json={
            "target_fragments": 1,
            "candidates": [
                {"name": "Bad", "pattern": "GXXTTC", "cut_top": 1, "cut_bottom": 5}
            ],
        },
    )
    assert resp2.status_code == 422
