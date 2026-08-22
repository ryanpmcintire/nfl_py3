# Registry integrity audit — `weak_signals.json` correlation & duplicate census (2026-08-22)

Scope: entries added to `registry/weak_signals.json` since 2026-08-21T00:00Z (per
`recorded_at`), duplicate/near-duplicate census, population/window correlation map,
and pool-integrity delta. Read-only on both registries; nothing deleted; no code
changes (docs only). Provenance labels per AGENTS.md: **measured** = run this
session (command cited), **read** = file opened this session, **reported** =
unverified claim from owner/doc, **inferred** = reasoning, not evidence.

Method note: all counts below are **measured** via Python over the registry JSON,
the pool command `.tools\uv.exe run nfl-ats weak-signals pool --league nfl
--effect-units accuracy_points`, and a DerSimonian-Laird reimplementation used
only where the tool cannot be run on a historical snapshot (labelled as
approximation where used).

---

## Executive summary — top 5 integrity risks

| # | Risk | Evidence (measured unless noted) | Severity |
|---|------|----------------------------------|----------|
| 1 | **Exact-duplicate measurement inside the NFL pool**: `body_clock_night_dose_ge2000` ≡ `body_clock_night_west_road_ge2000et` — identical effect, interval, P+, n=119, SE, and source artifact. One measurement is double-counted in the sign test and pool. | Full-field identity check, all stat fields + source byte-equal | High |
| 2 | **Two more numerically-duplicated night-screen dose buckets with disagreeing sample sizes**: `dose_1300` re-emits `body_clock_west_road_early`; `dose_1400_1659` re-emits `body_clock_west_road_midday_control`. P+ and SE identical; effects equal to stored 4-dp rounding; but `sample_games` disagree (352 vs 4,317; 270 vs 4,317), suggesting copied stats rather than an independent re-measure (**inferred**: copy bug in the night screen). | Precision comparison of effect/interval/P+/SE/n across pairs | High |
| 3 | **Pool overlap explosion**: `overlap_warnings` grew +25,224 this week (30,600 → 55,824, +82%), dominated by 21,517 new×old pairs because nearly every new entry spans the full 2009–2025 window. The pooled CI materially overstates precision. | Range-overlap rule validated to reproduce the tool's 55,824 exactly | High |
| 4 | **Same-window multi-arm batteries counted as independent sign-test votes**: 4 best-pick follow-up arms on the identical 11,780-game window; 4 overlay compositions on 1,503 games; 5 close-game-luck cells on 8,634; 6 redzone cells on 8,634; 4 motivation cells on 2,768. The 182/346 sign test treats these as separate coins. | Cluster analysis by source+window | Medium |
| 5 | **League misfiling**: the 4 `best_pick_followup_*` entries are tagged `cfb`, silently excluding them from the NFL pool (**inferred**: mislabel — they measure the NFL best-pick ranker on 11,780 games 2007–2025). If re-tagged, they add another same-window arm block. | League/unit counter over registry | Medium |

Bottom line: of 92 additions, **6 entries (3 pairs) are duplicate-candidates**
and must never contribute independent votes; the rest are genuinely distinct
measurements but mostly correlated decompositions sharing windows.

---

## 1. Entries added since 2026-08-21T00:00Z

Measured: 92 additions (87 on 2026-08-21, 5 on 2026-08-22); registry total 417.
All 92 classified `unresolved_below_power` (**measured**, classification
counter). 88 are `nfl`/`accuracy_points` (pool-eligible); 4 are tagged `cfb`
(risk #5).

Cluster codes: ALT altitude · BC body-clock · BPF best-pick follow-up · BYE bye ·
CGL close-game luck · DIV divisional rematch · MOT motivation ladder · NFC
NFL.com Friday · OL O-line continuity · OVR overlay subsets/stacker · PT
primetime · QBA QB age · RZ redzone reversion · SAG Sagarin battery · WXT weather-total interaction.

| Signal | Cl | Lg | Effect | 95% IV | P+ | n | Seasons | Duplicate flag |
|---|---|---|---|---|---|---|---|---|
| `era_weighting_half_life_8_opener_confirmation` | - | nfl | -0.2193 | [-3.39,+2.68] | 0.425 | 456 | 2020-2021 |  |
| `altitude_deficit_4000ft` | ALT | nfl | +0.0230 | [-0.24,+0.29] | 0.566 | 4,317 | 2009-2025 |  |
| `altitude_deficit_4000ft_division` | ALT | nfl | -0.0589 | [-0.21,+0.09] | 0.221 | 4,317 | 2009-2025 |  |
| `altitude_deficit_4000ft_era_2009_2017` | ALT | nfl | -0.0399 | [-0.41,+0.33] | 0.415 | 2,242 | 2009-2017 |  |
| `altitude_deficit_4000ft_era_2018_2025` | ALT | nfl | +0.0905 | [-0.28,+0.47] | 0.687 | 2,075 | 2018-2025 |  |
| `den_home_vs_own_conference` | ALT | nfl | -0.1046 | [-0.33,+0.12] | 0.175 | 4,317 | 2009-2025 |  |
| `mexico_city_neutral` | ALT | nfl | -0.0222 | [-0.05,+0.05] | 0.205 | 4,317 | 2009-2025 |  |
| `body_clock_east_host_west_visitor_early` | BC | nfl | -0.2690 | [-0.62,+0.09] | 0.071 | 4,317 | 2009-2025 |  |
| `body_clock_night_dose_1300` | BC | nfl | -0.1545 | [-0.63,+0.31] | 0.259 | 352 | 2009-2025 | NUMERIC DUP of body_clock_west_road_early |
| `body_clock_night_dose_1400_1659` | BC | nfl | -0.1984 | [-0.58,+0.18] | 0.149 | 270 | 2009-2025 | NUMERIC DUP of body_clock_west_road_midday_control |
| `body_clock_night_dose_1700_1959` | BC | nfl | +0.0124 | [-0.03,+0.04] | 0.689 | 3 | 2009-2025 |  |
| `body_clock_night_dose_ge2000` | BC | nfl | -0.1713 | [-0.41,+0.07] | 0.084 | 119 | 2009-2025 | EXACT DUP of body_clock_night_west_road_ge2000et |
| `body_clock_night_east_road_ge2000et` | BC | nfl | +0.1305 | [-0.31,+0.57] | 0.714 | 417 | 2009-2025 |  |
| `body_clock_night_west_road_ge2000et` | BC | nfl | -0.1713 | [-0.41,+0.07] | 0.084 | 119 | 2009-2025 | primary of exact-dup pair |
| `body_clock_night_west_road_ge2000et_2009_2016` | BC | nfl | -0.1048 | [-0.26,+0.06] | 0.109 | 48 | 2009-2016 |  |
| `body_clock_night_west_road_ge2000et_2017_2025` | BC | nfl | -0.0640 | [-0.25,+0.12] | 0.247 | 71 | 2017-2025 |  |
| `body_clock_night_west_road_true_slots` | BC | nfl | -0.1488 | [-0.39,+0.09] | 0.113 | 113 | 2009-2025 |  |
| `body_clock_west_host_east_visitor_late` | BC | nfl | +0.0265 | [-0.15,+0.20] | 0.616 | 4,317 | 2009-2025 |  |
| `body_clock_west_road_early` | BC | nfl | -0.1545 | [-0.63,+0.31] | 0.259 | 4,317 | 2009-2025 |  |
| `body_clock_west_road_early_2009_2016` | BC | nfl | -0.0308 | [-0.33,+0.27] | 0.422 | 4,317 | 2009-2016 |  |
| `body_clock_west_road_early_2017_2025` | BC | nfl | -0.1175 | [-0.46,+0.23] | 0.249 | 4,317 | 2017-2025 |  |
| `body_clock_west_road_midday_control` | BC | nfl | -0.1984 | [-0.58,+0.18] | 0.149 | 4,317 | 2009-2025 |  |
| `best_pick_followup_alpha2000_distance` | BPF | cfb | -1.7857 | [-7.50,+3.57] | 0.239 | 11,780 | 2007-2025 |  |
| `best_pick_followup_dispersion_gated_smooth_distance` | BPF | cfb | -1.4286 | [-5.36,+2.50] | 0.217 | 11,780 | 2007-2025 |  |
| `best_pick_followup_ensemble_distance` | BPF | cfb | +0.3571 | [-4.64,+5.36] | 0.527 | 11,780 | 2007-2025 |  |
| `best_pick_followup_smooth_cdf_distance` | BPF | cfb | +0.7143 | [-4.29,+5.71] | 0.584 | 11,780 | 2007-2025 |  |
| `bye_overval_both_bye_sanity` | BYE | nfl | -0.0306 | [-0.44,+0.36] | 0.442 | 4,317 | 2009-2025 |  |
| `bye_overval_fade_full_slate_post2011` | BYE | nfl | +0.5680 | [-0.41,+1.55] | 0.870 | 2,454 | 2012-2025 |  |
| `bye_overval_home_edge_post2011` | BYE | nfl | -0.3304 | [-0.76,+0.10] | 0.064 | 3,573 | 2012-2025 | disclosed subset overlap w/ venue_milestone_post_bye_home |
| `bye_overval_home_edge_pre2011` | BYE | nfl | +0.2708 | [-0.74,+1.20] | 0.707 | 744 | 2009-2011 |  |
| `bye_overval_road_fav_post2011` | BYE | nfl | -0.0129 | [-0.28,+0.26] | 0.461 | 3,573 | 2012-2025 |  |
| `close_game_luck_early_season_fade` | CGL | nfl | -0.1828 | [-1.46,+1.08] | 0.388 | 3,988 | 2009-2025 |  |
| `close_game_luck_one_score_over_fade` | CGL | nfl | -0.0076 | [-0.58,+0.55] | 0.482 | 8,634 | 2009-2025 |  |
| `close_game_luck_one_score_under_rebound` | CGL | nfl | -0.0151 | [-0.55,+0.52] | 0.473 | 8,634 | 2009-2025 |  |
| `close_game_luck_takeaway_share_extreme_fade` | CGL | nfl | -0.0305 | [-0.61,+0.54] | 0.448 | 8,634 | 2009-2025 |  |
| `close_game_luck_turnover_over_fade` | CGL | nfl | +0.0076 | [-0.57,+0.57] | 0.501 | 8,634 | 2009-2025 |  |
| `close_game_luck_turnover_under_rebound` | CGL | nfl | +0.4092 | [-0.15,+0.97] | 0.920 | 8,634 | 2009-2025 | disclosed decomposition of turnover trait |
| `divisional_rematch_blowout_winner_fade` | DIV | nfl | -0.0418 | [-0.21,+0.13] | 0.296 | 269 | 2009-2025 |  |
| `divisional_rematch_revenge_early_w1to6` | DIV | nfl | +0.0116 | [-0.02,+0.02] | 0.794 | 4 | 2009-2025 |  |
| `divisional_rematch_revenge_home_loser` | DIV | nfl | +0.1087 | [-0.12,+0.34] | 0.822 | 358 | 2009-2025 |  |
| `divisional_rematch_revenge_late_w12plus` | DIV | nfl | +0.0999 | [-0.17,+0.37] | 0.754 | 628 | 2009-2025 |  |
| `divisional_rematch_revenge_road_loser` | DIV | nfl | +0.0730 | [-0.14,+0.29] | 0.728 | 412 | 2009-2025 |  |
| `motivation_ladder_elim_visitor_alive_host` | MOT | nfl | +0.0198 | [-0.56,+0.60] | 0.509 | 2,768 | 2009-2025 |  |
| `motivation_ladder_fighter_vs_nothing` | MOT | nfl | -0.3341 | [-0.65,-0.02] | 0.015 | 2,768 | 2009-2025 |  |
| `motivation_ladder_locked_seed_wk16_18` | MOT | nfl | +0.1498 | [-0.24,+0.54] | 0.754 | 2,768 | 2009-2025 |  |
| `motivation_ladder_tank_zone_wk14_18` | MOT | nfl | +0.3049 | [-0.08,+0.70] | 0.933 | 2,768 | 2009-2025 |  |
| `nflcom_friday_new_saturday_designation` | NFC | nfl | +0.0000 | [-16.84,+16.65] | 0.463 | 1,574 | 2022-2024 |  |
| `nflcom_friday_out_count_ge2` | NFC | nfl | -2.6899 | [-5.40,+0.00] | 0.024 | 1,574 | 2022-2024 |  |
| `nflcom_friday_q_or_worse_starter_caliber` | NFC | nfl | +0.6667 | [-2.58,+4.02] | 0.642 | 1,574 | 2022-2024 |  |
| `ol_acute_overhaul_fade` | OL | nfl | -0.0227 | [-0.11,+0.06] | 0.260 | 6,644 | 2013-2025 |  |
| `ol_high_continuity_back` | OL | nfl | -0.5799 | [-1.42,+0.26] | 0.085 | 6,644 | 2013-2025 |  |
| `ol_low_continuity_fade` | OL | nfl | -0.0309 | [-0.74,+0.68] | 0.459 | 6,644 | 2013-2025 |  |
| `ol_prior_season_weak_early_fade` | OL | nfl | +0.1274 | [-0.86,+1.06] | 0.598 | 3,076 | 2013-2025 |  |
| `combined_stacker_opener_2022_2023` | OVR | nfl | -0.9728 | [-2.90,+0.95] | 0.133 | 514 | 2022-2023 | stacks overlay-family inputs |
| `overlay_subset_all_seven_joint` | OVR | nfl | -2.8609 | [-7.37,+1.67] | 0.105 | 1,503 | 2020-2025 |  |
| `overlay_subset_holdout_2020_2022_reverse` | OVR | nfl | +2.1307 | [-0.85,+5.11] | 0.912 | 704 | 2020-2022 | partitions same window as comps |
| `overlay_subset_holdout_2023_2025_frozen` | OVR | nfl | +0.8761 | [-2.88,+4.79] | 0.660 | 799 | 2023-2025 | partitions same window as comps |
| `overlay_subset_production_chain_coach_arrest` | OVR | nfl | +0.8649 | [-0.66,+2.40] | 0.859 | 1,503 | 2020-2025 |  |
| `overlay_subset_production_plus_division_revenge` | OVR | nfl | +1.5303 | [-0.54,+3.63] | 0.920 | 1,503 | 2020-2025 |  |
| `overlay_subset_production_plus_spread_gap_zone` | OVR | nfl | +1.8629 | [-0.46,+4.26] | 0.935 | 1,503 | 2020-2025 |  |
| `pt_away_underdog` | PT | nfl | -0.0618 | [-0.33,+0.21] | 0.324 | 8,634 | 2009-2025 |  |
| `pt_divisional_favorite` | PT | nfl | +0.0181 | [-0.21,+0.24] | 0.561 | 8,634 | 2009-2025 |  |
| `pt_off_loss` | PT | nfl | +0.0068 | [-0.25,+0.27] | 0.516 | 8,090 | 2009-2025 |  |
| `pt_off_win` | PT | nfl | +0.0138 | [-0.25,+0.28] | 0.536 | 8,090 | 2009-2025 |  |
| `pt_post_mnf_sunday` | PT | nfl | +0.0666 | [-0.23,+0.36] | 0.670 | 8,090 | 2009-2025 | note: measured 0-row overlap w/ bias_battery_short_week |
| `pt_post_mnf_sunday_era_2009_2017` | PT | nfl | +0.2546 | [-0.16,+0.67] | 0.884 | 4,196 | 2009-2017 | partitions parent |
| `pt_post_mnf_sunday_era_2018_2025` | PT | nfl | -0.1367 | [-0.55,+0.29] | 0.257 | 3,894 | 2018-2025 | partitions parent |
| `qb_age_rookie_late_improvement` | QBA | nfl | -0.4082 | [-0.92,+0.09] | 0.053 | 2,137 | 2009-2025 |  |
| `qb_age_rookie_vs_pressure` | QBA | nfl | -0.0333 | [-0.26,+0.20] | 0.387 | 2,137 | 2009-2025 |  |
| `qb_age_second_year_jump` | QBA | nfl | +0.2354 | [-0.20,+0.67] | 0.855 | 8,634 | 2009-2025 |  |
| `qb_age_veteran_late_fade` | QBA | nfl | -0.0373 | [-0.12,+0.05] | 0.207 | 94 | 2009-2025 |  |
| `redzone_reversion_c1_rz_over_fade` | RZ | nfl | +0.0302 | [-0.54,+0.60] | 0.537 | 8,634 | 2009-2025 |  |
| `redzone_reversion_c2_rz_under_rebound` | RZ | nfl | -0.1062 | [-0.70,+0.50] | 0.361 | 8,634 | 2009-2025 |  |
| `redzone_reversion_c3_third_down_over_fade` | RZ | nfl | +0.3665 | [-0.26,+1.00] | 0.872 | 8,634 | 2009-2025 |  |
| `redzone_reversion_c4_third_down_under_rebound` | RZ | nfl | -0.3564 | [-0.89,+0.18] | 0.099 | 8,634 | 2009-2025 |  |
| `redzone_reversion_c5_hot_offense_vs_stingy_defense` | RZ | nfl | +0.1610 | [-0.21,+0.52] | 0.804 | 4,317 | 2009-2025 |  |
| `redzone_reversion_c6_early_season_extreme_fade` | RZ | nfl | -0.4497 | [-1.60,+0.68] | 0.209 | 3,988 | 2009-2025 |  |
| `sagarin_battery_large_divergence_close` | SAG | nfl | -0.5863 | [-3.56,+2.39] | 0.347 | 1,194 | 2010-2025 |  |
| `sagarin_battery_large_divergence_era_2010_2016` | SAG | nfl | +1.8072 | [-3.00,+6.69] | 0.762 | 498 | 2010-2016 | partitions parent population |
| `sagarin_battery_large_divergence_era_2017_2025` | SAG | nfl | -2.2989 | [-6.05,+1.51] | 0.115 | 696 | 2017-2025 | partitions parent population |
| `sagarin_battery_large_divergence_open` | SAG | nfl | -1.4778 | [-5.98,+2.99] | 0.254 | 406 | 2020-2025 | nested in close population |
| `sagarin_battery_model_agreement_close` | SAG | nfl | -0.9474 | [-6.13,+4.27] | 0.360 | 1,492 | 2018-2025 |  |
| `sagarin_battery_top_decile_close` | SAG | nfl | +3.5354 | [-2.00,+9.15] | 0.891 | 297 | 2010-2025 | nested in close population |
| `sagarin_battery_top_decile_open` | SAG | nfl | -0.9434 | [-9.79,+7.89] | 0.393 | 106 | 2020-2025 | nested in open population |
| `venue_milestone_home_opener` | VEN | nfl | -0.1329 | [-0.67,+0.39] | 0.311 | 4,317 | 2009-2025 |  |
| `venue_milestone_new_stadium_debut` | VEN | nfl | +0.0263 | [-0.07,+0.11] | 0.745 | 4,317 | 2009-2025 |  |
| `venue_milestone_post_bye_home` | VEN | nfl | -0.2319 | [-0.67,+0.20] | 0.145 | 4,317 | 2009-2025 | overlaps bye_overval_home_edge_post2011 |
| `venue_milestone_post_bye_road` | VEN | nfl | -0.0010 | [-0.45,+0.45] | 0.500 | 4,317 | 2009-2025 |  |
| `wxtot_cold35_top_total` | WXT | nfl | -0.0071 | [-0.18,+0.17] | 0.462 | 4,313 | 2009-2025 |  |
| `wxtot_precip60_top_total` | WXT | nfl | +0.2243 | [+0.07,+0.37] | 0.998 | 4,313 | 2009-2025 |  |
| `wxtot_wind15_bottom_total` | WXT | nfl | +0.1366 | [-0.15,+0.42] | 0.828 | 4,313 | 2009-2025 | decomposition of weather_battery_high_wind_outdoor |
| `wxtot_wind15_top_total` | WXT | nfl | +0.0772 | [-0.12,+0.28] | 0.780 | 4,313 | 2009-2025 | decomposition of weather_battery_high_wind_outdoor |

---

## 2. Duplicate / near-duplicate measurements

Three collision groups found among the week's additions (**measured**: exact
match on `(effect, probability_positive)` plus full-field comparison;
cross-day old↔new numeric matches checked and none found):

| Pair | Type | Evidence | Verdict |
|---|---|---|---|
| `body_clock_night_dose_ge2000` ≡ `body_clock_night_west_road_ge2000et` | **Exact** | effect −0.1713, interval [−0.4137,+0.0738], P+ 0.0843, n=119, SE, and source artifact all identical | Same measurement recorded twice under two names, same results file. The dose entry's own note already reads "Exact duplicate of primary numbers; never pool as independent." (**read**) |
| `body_clock_night_dose_1300` ≈ `body_clock_west_road_early` | **Numeric** | P+ and SE identical; effect −0.1545 vs −0.1545255991 (night screen stores 4-dp-rounded values); intervals match at stored precision. But `sample_games` disagree: 352 vs 4,317 | Night screen appears to have re-emitted the early screen's cell rather than measuring its own dose bucket (**inferred**: copy bug — identical stats at different stated n are not plausibly independent) |
| `body_clock_night_dose_1400_1659` ≈ `body_clock_west_road_midday_control` | **Numeric** | effect −0.1984 vs −0.1983984911, same pattern as above; n 270 vs 4,317 | Same verdict as above |

Not duplicates (checked and cleared): `dose_1700_1959` (n=3) matches nothing;
no new entry numerically matches any pre-week entry; `pt_post_mnf_sunday`'s
note records a **measured** 0-row overlap with `bias_battery_short_week`.

---

## 3. Correlation map — shared populations/windows

Entries within each row share the same game population and season window and
**must never be pooled as independent** (**measured**: clusters by source
artifact + `sample_games` + seasons; cross-family links **read** from notes).

| Cluster | Entries | Shared window / population | Cross-family links |
|---|---|---|---|
| ALT | altitude_deficit_4000ft (+_division, ×2 era splits), den_home_vs_own_conference, mexico_city_neutral | Same artifact; 4,317-game 2009–2025 base; era splits partition the base | Era splits partition parent |
| BC | 15 entries across early screen (18:21Z artifact) and night screen (22:25Z artifact) | Early cells on 4,317; night cells on small subsets (3–417); night era splits partition their parent | Night dose buckets duplicate early-screen cells (§2) |
| BPF | 4 follow-up arms | Identical 11,780-game 2007–2025 window | Tagged `cfb` (risk #5); would collide with best_pick family if re-tagged nfl |
| BYE | 5 cells | post-2011 pair shares n=3,573; pre/post split the slate | `bye_overval_home_edge_post2011` discloses subset overlap with `venue_milestone_post_bye_home` and correlation with `travel_rest_home` (**read**) |
| CGL | 6 cells | 5 on identical 8,634; early_season_fade on 3,988 subset | `turnover_under_rebound` notes itself a "correlated decomposition … not independent confirmation" (**read**) |
| DIV | 5 cells | Nested revenge populations (269–628 games) inside the same mining frame | Overlaps bias_battery_division_revenge_game population |
| MOT | 4 cells | Identical 2,768-game population | — |
| NFC | 3 cells | Identical 1,574-game 2022–2024 window | — |
| OL | 4 cells | 3 on identical 6,644; prior_season_weak on 3,076 subset | — |
| OVR | 4 compositions + 2 holdouts + combined_stacker | Compositions share 1,503; holdouts partition it (704+799); stacker consumes overlay inputs on 2022–2023 | Stacker is downstream of overlay family inputs |
| PT | 7 cells | off_loss/win/post_mnf on 8,090; away_underdog/divisional on 8,634; era splits partition | — |
| QBA | rookie pair on 2,137; second_year on 8,634; veteran on 94 | Partially nested | — |
| RZ | c1–c4 on identical 8,634; c5/c6 subsets | Same trait decomposed | — |
| SAG | 7 entries | Nested divergence populations (106–1,492); era splits partition | — |
| VEN | 4 cells | Identical 4,317-game base | post_bye_home ↔ bye_overval link above |
| WXT | 4 interaction cells | Identical 4,313-game base | Decompositions of `weather_battery_high_wind_outdoor` (**read** from notes) |
| singleton | era_weighting_half_life_8_opener_confirmation | Reuses the spent [2020,2021] mod07 window (456 games) also touched by injury_value_lost family and weak_stack_v2/v3 (**read** from note + window match) | Reused window carries a stated discount, per AGENTS.md rule 6 |

---

## 4. Census: duplicate candidates vs genuinely distinct

| Category | Count | Notes |
|---|---|---|
| Week's additions total | 92 | 87 on 08-21 + 5 on 08-22 (**measured**) |
| Exact-duplicate candidates | **6 entries / 3 pairs** | §2; all six sit inside the BC family |
| Genuinely distinct measurements | **86** | Distinct constructs/cells; many remain correlated decompositions of shared windows (§3) and so are distinct-but-not-independent for pooling |
| Pool-eligible (nfl/accuracy_points) | 88 | 4 excluded by `cfb` tag (risk #5) |
| Classifications | 92 × `unresolved_below_power` | No terminal classifications; nothing to record via `weak-signals record` beyond what exists (**measured**) |

---

## 5. Recommended actions (nothing deleted)

Supersede via `--replace` with pointer notes — keep the original annotated and
pointing forward, following the established precedent
(`opener_error_mining_movement_agreement_agrees_corrected`: "Supersedes-by-relabeling
… annotated, not deleted" — **read**):

| Entry to supersede | Superseded by | Rationale |
|---|---|---|
| `body_clock_night_dose_ge2000` | `body_clock_night_west_road_ge2000et` | Exact duplicate; pointer note already half-written |
| `body_clock_night_dose_1300` | `body_clock_west_road_early` | Numeric re-emission of the early-screen cell |
| `body_clock_night_dose_1400_1659` | `body_clock_west_road_midday_control` | Numeric re-emission of the early-screen cell |

Annotate only (do not replace — these are real, distinct measurements that are
merely non-independent):

- Era-split entries (altitude ×2, pt_post_mnf ×2, body_clock west_road_early ×2, sagarin ×2, night ge2000et ×2): add "partition of `<parent>`; never pool with parent as independent".
- BPF arms: add shared-window annotation ("same 11,780-game window; dependence-aware testing only").
- OVR compositions and combined_stacker: annotate input overlap.
- CGL/RZ/MOT/NFC/VEN/BYE/SAG multi-cell batteries: one shared-window line per battery.
- Already-disclosed entries (`wxtot_wind15_top_total`, `close_game_luck_turnover_under_rebound`, `bye_overval_home_edge_post2011`) need no further action (**read**).

Investigate before any promotion claim:

1. Open `artifacts/body_clock_night_screen/20260821T222542Z/results.json` and determine why dose buckets carry stats identical to early-screen cells at disagreeing `sample_games` — if confirmed a copy bug, every number sourced from that artifact needs re-derivation before use (**inferred** priority).
2. Confirm or fix the `cfb` tags on the four `best_pick_followup_*` entries before the next pool run; if re-tagged `nfl`, apply the shared-window annotation simultaneously.
3. For any future battery recording >1 cell per population, declare the family and commensurability before signs are seen (existing rule; restated here because risk #4 shows the cost of skipping it).

---

## 6. Pool delta — overlap warnings and sign test

Current pool (**measured**, `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points`, run this session):

| Metric | Value |
|---|---|
| Eligible signals | 346 |
| Pooled effect | +0.004901 accuracy points, 95% [−0.011146, +0.020949] |
| `excludes_zero` | false |
| τ² (heterogeneity) | 0.0013706 |
| Sharpening vs best single | +0.0030998 |
| Sign test | **182 favouring candidate / 164 favouring baseline of 346**, p = 0.3608 — "directions are consistent with a coin flip" |
| `overlap_warnings` | **55,824** |

Overlap-warning growth (**measured**: I recomputed warnings under the tool's
rule — pairs whose season ranges `[min,max]` intersect — which reproduces the
tool's 55,824 exactly, then applied the same rule to the pre-week subset):

| Quantity | Count |
|---|---|
| Pre-week baseline warnings (old-only eligible, n=258) | 30,600 |
| Current warnings (n=346) | 55,824 |
| **Growth this week** | **+25,224 (+82%)** |
| …new×new pairs | 3,707 |
| …new×old pairs | 21,517 |

Sign-test movement: against the owner-cited baseline of 175-of-326 candidate
(**reported** by owner, unverified — I could not reconstruct it: the current
registry's pre-week eligible set is 258, so 175/326 was most likely a mid-batch
snapshot taken after ~68 of the week's entries were already recorded —
**inferred**), the count moved to 182-of-346, i.e. **+7 net candidate-favouring
signs on ~20 additional votes**, still statistically indistinguishable from a
coin flip (p = 0.361). Per the standing rule, an interval crossing zero is not
grounds for rejection; the pooled estimate remains unresolved.

Pooled-effect movement (approximation): my DerSimonian-Laird reimplementation
reproduces the tool's current pooled effect to ~2×10⁻⁴ and τ² to ~2×10⁻⁵;
applied to the pre-week subset it gives ≈ **+0.0133 [−0.0071, +0.0338]**, so
this week's batch moved the pooled estimate by roughly **−0.0086 accuracy
points** while narrowing the interval (~30% tighter SE). The batch leans
net-negative point-wise but does not resolve anything.

Interpretation guardrails: the pooled figure assumes independence its own
`overlap_warnings` field denies (note on the output: "Check overlap_warnings
before believing this interval"); the sign test counts correlated arms as
separate coins (risk #4). Treat both as upper bounds on precision until
families are pooled dependence-aware.

---

*Audit performed 2026-08-22. Registries untouched; only this document written.*
