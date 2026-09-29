# LEAD-70: preserved-model disagreement

**Verification:** `.tools/uv.exe run --no-sync python -B scripts/lead70_unit1.py`
ran once and exited 0 with `UV_NO_CACHE=1` and BLAS/OMP thread limits of 2.
Scoped Ruff passed. Pending record commands passed read-only closure validation and
PowerShell parsing; no record command ran. Subsequent edits corrected reporting only,
without changing fits, source selection, predictions, or metrics.

**Measured:** 1503 opener-graded non-push games across 2020-2025; 15 distinct challenger streams.
**Measured decisive-game record:** 8-15 on 23 changed-side games; 95% Wilson accuracy interval [0.188113, 0.551097], exact two-sided equal-success null p=0.210040. The four-term record on those same games is 15-8.

## Protocol and interpretation

The preregistration is in `docs/lanes/lead70.md`; source inventory is in `docs/lead70_inventory.md`.
The two fitted forms are the existing four-term logistic model and that same model with
`model_logit * z(disagreement)` added. This is `a*logit*(1+d*z)` with `d=interaction/a`.
All coefficients, feature standardisers, and disagreement tercile cuts are fitted on the other
five seasons. The full-population fit supplies descriptive IS metrics only. The fixed existing
ridge is 0.001; no parameter search occurred. Disagreement is the population standard deviation
of challenger predicted margins in points; it is never an independent side-selection rule.
The existing discrete opener probability supplies the model logit; no pooled-residual mapping is fitted.
**Look accounting:** 2 predeclared fitted forms, 12 LOSO fits plus 2 descriptive full fits,
3 predeclared disagreement bands, 12 arm-by-band diagnostic cells, 9 paired arm-by-metric contrasts.
The broader reporting surface is explicit; the nominal two looks is not a claim of two independent tests.
The market/model-only references are fixed and evaluated on exactly the same games.
Intervals are 95% percentile intervals from 4000 season-cluster draws (seed 20260929);
`probability_positive` counts positive draws plus half the zero draws. Six clusters limit precision.
Positive paired effects mean lower loss or greater accuracy; accuracy effects are percentage points.
These are reused research seasons and previously developed challengers, not an untouched outer test.
Different fitting configurations share training data and are not independent statistical replicates.
A training cutoff before game day establishes the declared fold condition, not the provenance of
every historical feature or an exact Tuesday-noon forecast capture; those remain source limitations.
**Measured:** incremental log loss, Brier and accuracy improvement intervals are wholly negative.
**Inferred disposition, pending serial recording:** the tested panel and modifier satisfy
`wrong_sign_resolved` under AGENTS.md; this does not close every disagreement mechanism.
The interaction coefficient changes sign across folds (four negative, two positive).
Positive market/model-only comparisons do not isolate this modifier; their records remain
`unresolved_below_power`. No serving or stable-edge claim follows from those comparisons.

**Measured:** four-term OOS probability difference from the saved served fit on these games: 2.22044604925e-16.
A nonzero difference can reflect common-game restriction; exact agreement is expected on the full population.
**Read source:** `artifacts/pick_probability/20260929T192747Z/per_game.parquet`, SHA256 `0490c806caf9e9707abe28f3e3e42f85df312084ae52d9d1588d173bba5b01e5`.

## IS, OOS and gap

Gap is OOS minus IS (lower is favorable for losses, higher for accuracy). Accuracy is a fraction here.

| Arm | Metric | Full-fit IS | LOSO OOS | OOS minus IS | OOS CI low | OOS CI high |
| --- | --- | --- | --- | --- | --- | --- |
| market | log_loss | NA | 0.69314718056 | NA | 0.69314718056 | 0.69314718056 |
| market | brier | NA | 0.25 | NA | 0.25 | 0.25 |
| market | accuracy | NA | 0.496340652029 | NA | 0.485211267606 | 0.506329113924 |
| model_only | log_loss | NA | 0.696917400657 | NA | 0.692593561147 | 0.702124572441 |
| model_only | brier | NA | 0.251715273298 | NA | 0.24963640222 | 0.254180108892 |
| model_only | accuracy | NA | 0.533599467731 | NA | 0.516441005803 | 0.55 |
| four_term | log_loss | 0.681816021924 | 0.682774250356 | 0.000958228431974 | 0.680372089984 | 0.685131124278 |
| four_term | brier | 0.244388880327 | 0.244838386553 | 0.000449506225497 | 0.24357932325 | 0.246015966498 |
| four_term | accuracy | 0.575515635396 | 0.574184963407 | -0.00133067198935 | 0.562629757785 | 0.587719298246 |
| disagreement | log_loss | 0.681778641854 | 0.683376983614 | 0.00159834176001 | 0.680794743022 | 0.685715485849 |
| disagreement | brier | 0.24436875122 | 0.245132861655 | 0.000764110434612 | 0.243812987543 | 0.246360783378 |
| disagreement | accuracy | 0.580172987359 | 0.569527611444 | -0.0106453759148 | 0.559861591696 | 0.580687830688 |

## Paired OOS improvement

| Reference | Metric | Effect | 95% low | 95% high | SE | probability_positive |
| --- | --- | --- | --- | --- | --- | --- |
| four_term | log_loss | -0.000602733258864 | -0.000921038437556 | -0.000305375027671 | 0.000160787789008 | 0 |
| four_term | brier | -0.000294475101834 | -0.000452993339221 | -0.000143756666829 | 8.06009395718e-05 | 0 |
| four_term | accuracy | -0.465735196274 | -0.816326530612 | -0.13698630137 | 0.171681920725 | 0.0005 |
| model_only | log_loss | 0.0135404170427 | 0.0083365606235 | 0.0191947554204 | 0.00281723370354 | 1 |
| model_only | brier | 0.00658241164341 | 0.00407627755964 | 0.00931194706033 | 0.0013827145613 | 1 |
| model_only | accuracy | 3.59281437126 | 2.08269032922 | 4.95785303958 | 0.747357624538 | 1 |
| market | log_loss | 0.00977019694553 | 0.00743169471083 | 0.0123524375377 | 0.00128206454984 | 1 |
| market | brier | 0.00486713834543 | 0.00363921662154 | 0.00618701245663 | 0.000663757060289 | 1 |
| market | accuracy | 7.31869594145 | 5.92782273166 | 8.86243386243 | 0.76277409343 | 1 |

## Coefficients and stability

Coefficients are on natural feature scales, except the interaction uses training-standardised disagreement.
`d` is unstable when the fitted model-logit coefficient is near zero; the interaction coefficient is primary.

| Held season | Arm | Train n | Test n | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | model_logit_x_disagreement | d | disagreement_mean | disagreement_std | Tercile q1 | Tercile q2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | four_term | 1283 | 220 | -0.045251389477 | 0.341245670369 | 0.244788877029 | 0.220729892837 | 0.0281125857047 | NA | NA | 0.884370103922 | 0.548546109845 | NA | NA |
| 2021 | four_term | 1267 | 236 | -0.0623316714161 | 0.17791560301 | 0.271867045317 | 0.223715901482 | 0.0441826334916 | NA | NA | 0.894530720611 | 0.562859056609 | NA | NA |
| 2022 | four_term | 1255 | 248 | -0.0556982863393 | 0.249646980307 | 0.238476202553 | 0.22290342563 | 0.0396741936399 | NA | NA | 0.894671606871 | 0.550098412606 | NA | NA |
| 2023 | four_term | 1237 | 266 | -0.0380639602385 | 0.341479690675 | 0.246113773766 | 0.230271820108 | 0.00228557855336 | NA | NA | 0.900862418544 | 0.569151930079 | NA | NA |
| 2024 | four_term | 1237 | 266 | -0.0561775905904 | 0.256115554052 | 0.262262006796 | 0.246065978687 | 0.0391458442704 | NA | NA | 0.914019095589 | 0.573572672432 | NA | NA |
| 2025 | four_term | 1236 | 267 | -0.0540985739437 | 0.292328163528 | 0.29599663969 | 0.188739248752 | 0.047462415826 | NA | NA | 0.91210221856 | 0.555365538655 | NA | NA |
| full IS | four_term | 1503 | 0 | -0.0520649470078 | 0.27650961063 | 0.259973644449 | 0.221777836843 | 0.0341744224812 | NA | NA | 0.899959668409 | 0.560026187401 | NA | NA |
| 2020 | disagreement | 1283 | 220 | -0.041778814066 | 0.360422665399 | 0.245245107798 | 0.218849562478 | 0.0236476219648 | -0.112204548297 | -0.311313796463 | 0.884370103922 | 0.548546109845 | 0.584405112769 | 0.964501089801 |
| 2021 | disagreement | 1267 | 236 | -0.0623509910476 | 0.177469923981 | 0.271870657963 | 0.223777552701 | 0.0442328174339 | 0.00377978859265 | 0.0212981924365 | 0.894530720611 | 0.562859056609 | 0.596202186949 | 0.975709815956 |
| 2022 | disagreement | 1255 | 248 | -0.0531506200442 | 0.263254383906 | 0.238115603664 | 0.221473437342 | 0.0364274850921 | -0.0847312783382 | -0.321860844561 | 0.894671606871 | 0.550098412606 | 0.596141452686 | 0.979306768228 |
| 2023 | disagreement | 1237 | 266 | -0.0355104517597 | 0.356321658212 | 0.245711554033 | 0.229255780894 | -0.00138568771254 | -0.0841458242747 | -0.236151304125 | 0.900862418544 | 0.569151930079 | 0.602548615245 | 0.987124919505 |
| 2024 | disagreement | 1237 | 266 | -0.0583266129871 | 0.245148088916 | 0.262274170233 | 0.247801570406 | 0.0418463541426 | 0.0666589097231 | 0.271912826316 | 0.914019095589 | 0.573572672432 | 0.606267516537 | 0.994079078355 |
| 2025 | disagreement | 1236 | 267 | -0.0511456607272 | 0.30770140415 | 0.295730675222 | 0.187452726731 | 0.0440905584536 | -0.102117384376 | -0.331871687937 | 0.91210221856 | 0.555365538655 | 0.612043007779 | 1.00030589798 |
| full IS | disagreement | 1503 | 0 | -0.0506974188725 | 0.284758788358 | 0.259897371876 | 0.220878949282 | 0.0323638112749 | -0.0525130744181 | -0.184412480194 | 0.899959668409 | 0.560026187401 | NA | NA |

## Season stability and fold IS/OOS

| Season | Arm | Held n | Train LL | Train Brier | Train accuracy | OOS LL | OOS Brier | OOS accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | four_term | 220 | 0.681198801546 | 0.244073824172 | 0.579890880748 | 0.685840339152 | 0.246385670458 | 0.554545454545 |
| 2021 | four_term | 236 | 0.681082822866 | 0.24404032882 | 0.572217837411 | 0.686447158157 | 0.246677034389 | 0.563559322034 |
| 2022 | four_term | 248 | 0.682291503892 | 0.244640730541 | 0.568924302789 | 0.679695770788 | 0.243306877693 | 0.604838709677 |
| 2023 | four_term | 266 | 0.682740788167 | 0.244865451506 | 0.582053354891 | 0.677979750425 | 0.242324443786 | 0.563909774436 |
| 2024 | four_term | 266 | 0.681472801991 | 0.2442589462 | 0.572352465643 | 0.683667466736 | 0.245123684942 | 0.582706766917 |
| 2025 | four_term | 267 | 0.681633664883 | 0.244234108632 | 0.575242718447 | 0.683747503467 | 0.245581121583 | 0.573033707865 |
| 2020 | disagreement | 220 | 0.681022776943 | 0.243979821837 | 0.581449727202 | 0.686935250461 | 0.246932862166 | 0.55 |
| 2021 | disagreement | 236 | 0.681082651816 | 0.244040196186 | 0.573007103394 | 0.686481562614 | 0.246694524429 | 0.563559322034 |
| 2022 | disagreement | 248 | 0.682191863543 | 0.244590113464 | 0.572111553785 | 0.680075056382 | 0.243486263676 | 0.592741935484 |
| 2023 | disagreement | 266 | 0.682638702484 | 0.24481599207 | 0.578819725141 | 0.678339143353 | 0.242474894953 | 0.563909774436 |
| 2024 | disagreement | 266 | 0.681411796093 | 0.244232233715 | 0.569118835893 | 0.684813635547 | 0.24568005782 | 0.578947368421 |
| 2025 | disagreement | 267 | 0.681496247022 | 0.24415741885 | 0.580097087379 | 0.684355615446 | 0.245901658411 | 0.565543071161 |

## Reliability by training-defined disagreement tercile

| Tercile | Arm | n | Mean home p | Home cover rate | Mean pick p | Accuracy | Brier | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | market | 500 | 0.5 | 0.498 | 0.5 | 0.498 | 0.25 | 0.69314718056 |
| 1 | model_only | 500 | 0.485601956682 | 0.498 | 0.555479603155 | 0.544 | 0.251875472789 | 0.697212612784 |
| 1 | four_term | 500 | 0.499921243517 | 0.498 | 0.55843951132 | 0.556 | 0.249005858926 | 0.691607683281 |
| 1 | disagreement | 500 | 0.499492381242 | 0.498 | 0.559819562672 | 0.554 | 0.249462005025 | 0.692579265623 |
| 2 | market | 501 | 0.5 | 0.483033932136 | 0.5 | 0.483033932136 | 0.25 | 0.69314718056 |
| 2 | model_only | 501 | 0.475370881245 | 0.483033932136 | 0.562030343788 | 0.538922155689 | 0.250957260843 | 0.695493764801 |
| 2 | four_term | 501 | 0.494269120973 | 0.483033932136 | 0.558269303864 | 0.588822355289 | 0.243062050105 | 0.679199974742 |
| 2 | disagreement | 501 | 0.493945412825 | 0.483033932136 | 0.558859319361 | 0.578842315369 | 0.242891286625 | 0.678855286797 |
| 3 | market | 502 | 0.5 | 0.50796812749 | 0.5 | 0.50796812749 | 0.25 | 0.69314718056 |
| 3 | model_only | 502 | 0.478641964944 | 0.50796812749 | 0.562594459553 | 0.517928286853 | 0.252312214522 | 0.698044164603 |
| 3 | four_term | 502 | 0.493844746836 | 0.50796812749 | 0.556972633031 | 0.577689243028 | 0.242460315584 | 0.677543165933 |
| 3 | disagreement | 502 | 0.495148272494 | 0.50796812749 | 0.555392071947 | 0.575697211155 | 0.243058075609 | 0.678724053537 |
