# PFSCE v8.2 Broad Prospective Cohort Extension

Cohort extension ID: `PFSCE-GEN-20260820-v1-B`
Freeze time: **2026-08-20T08:53Z**
Parent cohort: `PFSCE-GEN-20260820-v1`

All questions below were unresolved at freeze time. Baseline snapshots and PFSCE probabilities are frozen now; later runs may append evidence and probability trajectories but must not rewrite these originals.

## GEN-20260820-005 — US federal government shutdown by October 1, 2026

**Resolution contract.** YES iff the United States federal government enters a shutdown caused by a lapse in appropriations by 2026-10-01 23:59 ET, including a partial shutdown where non-excepted operations are suspended. Resolve primarily from official OMB/OPM/U.S. government information; credible reporting only if official evidence is ambiguous.

**Strong baseline snapshot:** Polymarket `Government shutdown by October 1?`, **15% YES** displayed at freeze-window retrieval (page also showed 14% in its FAQ/order-book text; store the visible top-line 15% as the frozen baseline and flag the 1 pp intrapage discrepancy as baseline measurement uncertainty). Source: https://polymarket.com/event/government-shutdown-by-october-1-20260610162414910

**PFSCE raw probability:** **10% YES**.

**Probability provenance:** reference-market anchored, evidence-adjusted raw model judgment; not locally calibrated; no PFSCE production authority.

**Authority:** F1 exploratory.

**Why adjusted below baseline:** the U.S. Senate's active-legislation page lists FY2027 continuing-resolution vehicles through Dec. 4 and Dec. 11, and an official House page states a continuing-resolution bill through Dec. 4 passed the House. This materially reduces the path to an Oct. 1 lapse, while Senate/presidential completion risk remains. Official evidence snapshot: https://www.senate.gov/legislative/active_leg_page.htm ; https://cloud.house.gov/positions/making-continuing-appropriations-for-fiscal-year-2027-and-for-other-purposes

**Driver cluster:** US fiscal/appropriations politics. Keep separate from monetary-policy questions but not necessarily fully independent of broad U.S. political-regime shocks.

## GEN-20260820-006 — Russia–Ukraine ceasefire agreement by October 31, 2026

**Resolution contract.** YES iff Russia and Ukraine reach a mutually agreed general suspension of direct military engagement by 2026-10-31 23:59 ET, officially announced by both countries or confirmed by a consensus of credible reporting. Localized pauses, unilateral de-escalation, or issue-specific arrangements do not count. Primary sources: governments of Russia/Ukraine plus consensus reporting.

**Strong baseline snapshot:** Polymarket `Russia x Ukraine ceasefire agreement by...?`, October 31 contract, **8% YES** at freeze-window retrieval. Source: https://polymarket.com/event/russia-x-ukraine-ceasefire-agreement-by

**PFSCE raw probability:** **6% YES**.

**Probability provenance:** reference-market anchored, evidence-adjusted raw model judgment; not calibrated for war/ceasefire targets.

**Authority:** F1 exploratory.

**Why adjusted below baseline:** Reuters on Aug. 19 reported a POW exchange but continuing division over peace negotiations; Reuters on Aug. 14 reported Russia dismissing a Black Sea ceasefire, and on Aug. 19 reported major Russian missile strikes on Kyiv while diplomacy remained stalled. This is evidence of humanitarian channels without a near-term general ceasefire pathway. Sources: https://www.reuters.com/world/russia-ukraine-swap-103-prisoners-war-each-2026-08-19/ ; https://www.reuters.com/world/europe/russia-dismisses-idea-black-sea-ceasefire-2026-08-14/ ; https://www.reuters.com/world/ukrainian-capital-kyiv-under-attack-by-russian-ballistic-missiles-mayor-says-2026-08-19/

**Driver cluster:** Russia–Ukraine war/diplomacy.

## GEN-20260820-007 — US government removes public access to another major AI model by December 31, 2026

**Resolution contract.** YES iff a formal U.S. federal action after the market's creation causes ordinary public access within the United States to at least one qualifying flagship general-purpose AI model to be generally removed for any period by 2026-12-31 23:59 ET. Removal from one channel only does not count. Resolve from official U.S. government and relevant AI-company announcements, with credible reporting if needed.

**Strong baseline snapshot:** Polymarket `US Government removes public access to another major AI model in 2026?`, **22% YES** at freeze-window retrieval. Source: https://polymarket.com/event/us-government-removes-public-access-to-another-major-ai-model-in-2026-20260703202936862

**PFSCE raw probability:** **18% YES**.

**Probability provenance:** reference-market anchored, evidence-adjusted raw model judgment; not calibrated for strategic regulatory targets.

**Authority:** F1 exploratory.

**Why adjusted below baseline:** a June U.S. directive did force Anthropic to suspend Fable 5/Mythos 5 access, proving mechanism feasibility, but access to Fable 5 was restored after the controls were lifted. The White House's June 2 frontier-model order explicitly says its voluntary framework does not authorize mandatory licensing/preclearance for model release, while the national-security memorandum emphasizes rapid access to advanced models. This leaves a nontrivial recurrence risk but weakens the case for another broad domestic removal before year-end. Sources: https://www.anthropic.com/news/fable-mythos-access ; https://www.anthropic.com/news/redeploying-fable-5 ; https://www.whitehouse.gov/presidential-actions/2026/06/promoting-advanced-artificial-intelligence-innovation-and-security/ ; https://www.whitehouse.gov/presidential-actions/2026/06/national-security-presidential-memorandum-nspm-11/

**Driver cluster:** US AI national-security/regulatory policy. Correlated with other U.S.-AI-policy questions but distinct from Anthropic IPO timing.

## GEN-20260820-008 — WHO declares any new pandemic in 2026

**Resolution contract.** YES iff the World Health Organization explicitly declares or characterizes any disease as a pandemic between 2026-01-01 and 2026-12-31 23:59 ET. A PHEIC alone does not satisfy this contract. Primary resolution source: official WHO announcements.

**Strong baseline snapshot:** Polymarket `New pandemic in 2026?`, **6% YES** at freeze-window retrieval. Source: https://polymarket.com/event/new-pandemic-in-2026

**PFSCE raw probability:** **9% YES**.

**Probability provenance:** reference-market anchored, evidence-adjusted raw model judgment; not calibrated for pandemic-declaration targets.

**Authority:** F1 exploratory.

**Why adjusted above baseline:** WHO's Aug. 14 update says the Bundibugyo Ebola outbreak is in intense transmission, is the largest Ebola outbreak ever reported in DRC, is expanding faster than prior Ebola outbreaks there, has spread across six provinces, and remains a PHEIC. WHO still assesses risk outside the African region and globally as low, so this is not evidence that a pandemic declaration is likely; it is a measurable tail-risk increase relative to an ordinary quiet-year reference class. Source: https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON615

**Driver cluster:** global infectious-disease emergency. Keep separate from NECF/weather and from economic/political clusters.

## Cohort-level rules

- These four questions add four nominal observations but not necessarily four independent observations; effective-N must account for shared U.S. policy/regime drivers where relevant.
- Baseline prices are frozen snapshots, not later closing prices.
- PFSCE probabilities are raw evidence-adjusted judgments with explicit F1 authority; they are not F2 and must not be presented as calibrated probabilities.
- Updates are append-only. The original 10%, 6%, 18%, and 9% values remain immutable for first-issue scoring.
- Score binary questions with Brier and log loss against the frozen baseline. Report per-question differences before any aggregate.
- No question may be removed because its forecast looks bad or its baseline later moves.
