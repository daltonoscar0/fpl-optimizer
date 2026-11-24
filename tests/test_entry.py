"""Reading a real manager's team from public data, offline.

Two things have to be reconstructed, because the public endpoints serve a squad
but not its price history: what each player is worth to *this* manager, and how
many free transfers they have. Both are arithmetic over recorded payloads, so
both are tested without a request — the fixtures are shaped like the live
endpoints, verified against a real entry, with invented numbers.
"""

from __future__ import annotations

import pytest

from fpl_optimizer.entry import (
    EntryUnavailable,
    banked_free_transfers,
    load_entry,
    purchase_prices,
    resolve_players,
    selling_prices,
    squad_from_picks,
)
from fpl_optimizer.rules import archive_season_rules

from .fixtures import ENTRY_ID, StubEntryClient, entry_payloads
from .fixtures.handbuilt import hand_built_season


@pytest.fixture
def payloads():
    return entry_payloads()


@pytest.fixture
def client(payloads):
    return StubEntryClient(payloads)


@pytest.fixture(scope="module")
def rules():
    return archive_season_rules("2024-25")


@pytest.fixture(scope="module")
def season():
    return hand_built_season()


# --- free transfers ---------------------------------------------------------


def _history(rows, chips=()):
    return {
        "chips": [{"name": name, "event": gw} for gw, name in chips],
        "current": [
            {"event": gw, "event_transfers": used, "event_transfers_cost": cost}
            for gw, used, cost in rows
        ],
    }


def test_one_free_transfer_accrues_a_gameweek(rules):
    free, notes = banked_free_transfers(
        _history([(1, 0, 0), (2, 0, 0), (3, 0, 0)]), rules
    )
    # Started with one, banked one in each of three gameweeks.
    assert free == 4
    assert notes == []


def test_banking_stops_at_the_cap(rules):
    free, _ = banked_free_transfers(
        _history([(gw, 0, 0) for gw in range(1, 20)]), rules
    )
    assert free == rules.max_free_transfers


def test_transfers_spend_the_bank(rules):
    # Three gameweeks of accrual to three free transfers, then two spent.
    free, notes = banked_free_transfers(
        _history([(1, 0, 0), (2, 0, 0), (3, 2, 0)]), rules
    )
    assert free == 2
    assert notes == []


def test_a_hit_does_not_spend_more_than_the_bank_holds(rules):
    """Four transfers on one free one: three are hits, and the bank empties."""
    free, notes = banked_free_transfers(_history([(1, 4, 12)]), rules)

    assert free == 1  # emptied, then one accrues
    assert notes == []


def test_a_wildcard_gameweeks_transfers_are_free_and_keep_the_bank(rules):
    """Eight transfers on a wildcard cost nothing and spend no free transfer."""
    free, notes = banked_free_transfers(
        _history([(1, 0, 0), (2, 8, 0)], chips=[(2, "wildcard")]), rules
    )

    assert free == 3  # one at the start, one in each gameweek, none spent
    assert any("wildcard" in note for note in notes)


def test_a_free_hit_is_treated_the_same_way(rules):
    free, notes = banked_free_transfers(
        _history([(1, 11, 0)], chips=[(1, "freehit")]), rules
    )
    assert free == 2
    assert any("freehit" in note for note in notes)


def test_a_disagreement_with_the_hits_fpl_charged_is_reported(rules):
    """The replay is checked against what FPL actually charged.

    One transfer on one free transfer should be free. If FPL charged four
    points for it, this reconstruction has the rules wrong — which is worth
    saying rather than quietly advising on a bad number.
    """
    free, notes = banked_free_transfers(_history([(1, 1, 4)]), rules)

    assert any("may be wrong" in note for note in notes)
    assert any("expected 0" in note for note in notes)
    assert free == 1


def test_gameweeks_before_the_team_existed_are_skipped(rules):
    """A team started in gameweek 2 did not bank a transfer in gameweek 1."""
    rows = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
    from_one, _ = banked_free_transfers(_history(rows), rules, started_event=1)
    from_two, _ = banked_free_transfers(_history(rows), rules, started_event=2)

    assert from_one == 4
    assert from_two == 3


def test_the_recorded_entry_has_the_free_transfers_its_history_implies(
    payloads, rules
):
    """Started GW2, two transfers in GW3, no chips, no hits charged."""
    free, notes = banked_free_transfers(payloads["history"], rules, started_event=2)

    # GW2: 1 -> 2. GW3: two spent from two, then one accrues.
    assert free == 1
    assert notes == []


# --- what a player is worth to this manager ---------------------------------


def test_a_transferred_in_player_cost_what_the_transfer_says(payloads, season):
    prices, notes = purchase_prices(
        [7, 12], payloads["transfers"], season, started_event=2
    )

    assert prices[7] == 40
    assert prices[12] == 72
    assert notes == []


def test_an_original_pick_cost_its_price_when_the_team_started(payloads, season):
    """Nothing in the transfer list, so it was bought with the initial squad."""
    prices, _ = purchase_prices([4], payloads["transfers"], season, started_event=2)

    at_start = season.player_gws[
        (season.player_gws["player_id"] == 4) & (season.player_gws["gw"] == 2)
    ]["price"]
    assert prices[4] == int(at_start.iloc[0])


def test_the_most_recent_purchase_wins(season):
    """A player can be sold and bought back at a different price."""
    transfers = [
        {"element_in": 4, "element_in_cost": 100, "element_out": 9,
         "element_out_cost": 60, "event": 2, "time": "2026-09-01T10:00:00Z"},
        {"element_in": 4, "element_in_cost": 120, "element_out": 11,
         "element_out_cost": 95, "event": 5, "time": "2026-09-30T10:00:00Z"},
    ]
    prices, _ = purchase_prices([4], transfers, season, started_event=1)
    assert prices[4] == 120


def test_an_unknown_purchase_price_is_flagged_not_hidden(season):
    prices, notes = purchase_prices([9999], [], season, started_event=2)

    assert prices[9999] == 0
    assert any("no purchase price" in note for note in notes)


@pytest.mark.parametrize(
    "paid, now, expected",
    [
        (70, 70, 70),   # unchanged
        (70, 75, 72),   # rose 0.5, keep half rounded down
        (70, 73, 71),   # rose 0.3, half is 0.15 -> 0.1
        (70, 71, 70),   # rose 0.1, half rounds down to nothing
        (70, 65, 65),   # a fall is taken in full
        (155, 158, 156),  # the real case: Haaland at 15.5 -> 15.8 sells at 15.6
    ],
)
def test_selling_price_is_the_cost_plus_half_the_rise(season, rules, paid, now, expected):
    """FPL's rule, which is why a purchase price has to be reconstructed at all."""
    assert rules.selling_price(paid, now) == expected


def test_selling_prices_use_the_latest_price_up_to_the_gameweek(season, rules):
    prices = selling_prices({4: 100}, season, rules, gw=2)
    at_gw2 = season.player_gws[
        (season.player_gws["player_id"] == 4) & (season.player_gws["gw"] == 2)
    ]["price"]

    assert prices[4] == rules.selling_price(100, int(at_gw2.iloc[0]))


# --- the picks payload ------------------------------------------------------


def test_the_squad_comes_out_in_position_order(payloads):
    squad, captain, vice = squad_from_picks(payloads["picks"])

    assert len(squad) == 15
    assert captain == 4
    assert vice == 2
    assert captain in squad and vice in squad


def test_an_empty_picks_payload_is_an_error():
    with pytest.raises(EntryUnavailable, match="no players"):
        squad_from_picks({"picks": []})


# --- the whole read ---------------------------------------------------------


def test_loading_an_entry_reads_only_the_four_public_endpoints(
    client, season, rules
):
    state = load_entry(client, ENTRY_ID, season, rules)

    assert sorted(c.split("/")[0] for c in client.calls) == [
        "entry",
        "history",
        "picks",
        "transfers",
    ]
    assert state.team_id == ENTRY_ID
    assert state.name == "Hand Built FC"
    assert state.manager == "Test Manager"
    assert state.started_event == 2
    assert state.gw == 3


def test_loading_an_entry_assembles_bank_squad_and_prices(client, season, rules):
    state = load_entry(client, ENTRY_ID, season, rules)

    assert len(state.squad) == 15
    assert state.bank == 8
    assert state.free_transfers == 1
    assert set(state.purchase) == set(state.squad)
    assert set(state.selling) == set(state.squad)
    # The two transferred-in players carry the price the transfer recorded.
    assert state.purchase[7] == 40
    assert state.purchase[12] == 72


def test_the_budget_is_the_bank_plus_what_the_squad_would_fetch(
    client, season, rules
):
    state = load_entry(client, ENTRY_ID, season, rules)
    assert state.budget == state.bank + sum(state.selling.values())


def test_the_read_says_the_squad_may_be_out_of_date(client, season, rules):
    """The public endpoint cannot know about a transfer made since the deadline."""
    state = load_entry(client, ENTRY_ID, season, rules)

    assert any("not necessarily the squad now" in note for note in state.notes)


def test_a_team_value_that_does_not_reconcile_is_reported(client, season, rules):
    """FPL's `value` is the squad at current prices plus the bank.

    Verified against a real entry across ten gameweeks, where it held exactly.
    So it is a free check on this reconstruction: if it does not reconcile,
    either the squad or the prices are wrong and the selling prices built on
    them cannot be trusted.
    """
    payloads = entry_payloads()
    payloads["picks"]["entry_history"]["value"] = 9999
    state = load_entry(StubEntryClient(payloads), ENTRY_ID, season, rules)

    assert any("the prices or the squad disagree" in note for note in state.notes)


# --- naming a squad by hand -------------------------------------------------


def test_names_resolve_to_ids(season):
    ids, _ = resolve_players(["Keeper", "Playmaker"], season)
    assert ids == (1, 2)


def test_ids_pass_straight_through(season):
    ids, _ = resolve_players(["4", "12"], season)
    assert ids == (4, 12)


def test_names_and_ids_can_be_mixed(season):
    ids, _ = resolve_players(["Keeper", "12"], season)
    assert ids == (1, 12)


def test_an_unknown_name_is_an_error(season):
    with pytest.raises(EntryUnavailable, match="no player matching"):
        resolve_players(["Nobody"], season)


def test_an_unknown_id_is_an_error(season):
    with pytest.raises(EntryUnavailable, match="no player with id"):
        resolve_players(["99999"], season)


def test_an_ambiguous_name_is_refused_rather_than_guessed(season):
    """Picking the wrong Silva would advise on someone else's squad.

    The real case: "Wilson" matched three players and "Gomez" two.
    """
    data = hand_built_season()
    frame = data.player_gws.copy()
    frame.loc[frame["player_id"] == 9, "name"] = "Keeper"  # now two Keepers
    data.player_gws = frame

    with pytest.raises(EntryUnavailable, match="matches more than one player"):
        resolve_players(["Keeper"], data)


def test_the_same_player_twice_is_an_error(season):
    with pytest.raises(EntryUnavailable, match="named twice"):
        resolve_players(["Keeper", "1"], season)


def test_a_name_resolves_without_its_accents(season):
    """Typing "Guehi" for "Guéhi" is what a manager will actually do."""
    data = hand_built_season()
    frame = data.player_gws.copy()
    frame.loc[frame["player_id"] == 1, "name"] = "Guéhi"
    data.player_gws = frame

    ids, _ = resolve_players(["Guehi"], data)
    assert ids == (1,)
