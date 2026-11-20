# Grouped dataset splits with PingRoom

Create a train/holdout partition while keeping related observations together. A named dataset owner selects the grouping column through PingRoom. Code searches every supported partition for the closest class-count and row-count targets, then exports row IDs and diagnostics.

The example contains repeated photographs of practice cards. The owner decides whether card identity or capture batch defines the evaluation groups. The program does not discover entities or train a model; selecting a grouping column does not rule out every source of leakage.

## Try it offline

Choose **Use this template**, clone your copy, and use Node.js 22 or newer:

```sh
npm ci
npm test
npm run demo
```

The demos compare two grouping columns and show Unknown withholding output. They use `@pingroom/sdk` **0.8.0** through local mock transports. After installation, tests and demos need no credentials or network. The lockfile pins dependencies; `.npmrc` requests a 14-day minimum release age where supported by npm. Live phone delivery is not tested.

## Prepare dataset metadata

```sh
cp examples/dataset.json dataset.json
cp config.example.json config.json
npm run preview
```

Export metadata from one frozen dataset version. Keep features, source images and other training data in your existing pipeline. Supply only these fields:

| Field | Meaning |
| --- | --- |
| `datasetId`, `title`, `context` | Stable dataset ID and short descriptions identifying the source and grouping question. |
| `holdoutPercent` | Integer from 1 to 99. This targets the fraction of rows and each observed class assigned to holdout. |
| `groupingColumns` | 1–3 objects with unique `id`, `label` and `meaning`. Each describes a candidate entity or dependency grouping. |
| `rows` | 2–128 objects with unique `id`, `classId` and `groups`. There may be 1–8 observed classes. |
| Row `groups` | An object containing exactly one nonempty group ID for every candidate column. Each column must contain 2–16 distinct group IDs. |

All row, class, column and group IDs use 1–12 characters: a lowercase letter followed by lowercase letters, digits, `_` or `-`. `unknown`, `constructor`, `prototype` and `__proto__` are reserved. `datasetId` uses the same alphabet with a 40-character limit. Text limits in UTF-16 units are 40 for the title, 80 for context, 26 for column labels and 48 for meanings. The complete Question must fit 500 units.

IDs are exact and case-sensitive. Extra fields, incomplete group maps, duplicate row/column IDs, whitespace padding, malformed Unicode and control/format characters fail validation. Rows and columns canonicalize by ID, so reordering metadata does not change its identity. Changing any value does. Multiple physical observations of one entity must use the same declared group ID; the program cannot infer that relationship.

## Connect the dataset owner

Install [PingRoom](https://pingroom.io/i), then run:

```sh
npm run connect
npm run inspect
```

Approve the pairing link and grant a **standard private room** containing the owner, with access to list rooms and create/read Questions. Verify the intended person in the app, then enter their exact member UUID and a descriptive label in `config.json`. Names alone do not establish identity.

For an existing granted connection, copy `.env.example` to `.env` and set **both** `PINGROOM_TOKEN` and `PINGROOM_ROOM`. Blank or partial pairs fail; credential sources cannot be combined.

```sh
npm run preview
npm start
# After the owner answers:
npm start
npm run export
```

`preview` and `validate` print the complete local metadata, policy, recipient and visible Question without network access. The Question sends the dataset ID, title/context, column labels/meanings, owner UUID and input/column/policy hashes. Individual row IDs, class IDs and group values stay local. Only nonsecret descriptions belong in the Question.

Each `start` invocation creates at most one direct, 15-minute Question or reads the same Question once. The owner selects a documented column or **Unknown**. Unknown, missing, expired, cancelled, invalid or misattributed answers withhold the partition. There is no background polling or automatic replacement Question.

## Partition rule and output

```sh
npm run status
npm run report
npm run export
```

These commands are local and need no credentials. `report` writes `.split/report.json`; after a known selection, `export` writes `.split/split.json`. Both include source metadata, policy and attributed answer evidence. Repeated exports send nothing and preserve the source.

For the selected column, every group moves intact to train or holdout. Both sides must be nonempty. Code enumerates all `2^G − 2` possible holdout subsets, where `G` is the number of groups. For each subset it calculates:

```text
classDeviationScaled = sum over classes c of abs(100 * holdoutCount[c] - percent * totalCount[c])
totalSizeDeviationScaled = abs(100 * holdoutRows - percent * totalRows)
```

It minimizes class deviation first, then total-size deviation. These integer scores are absolute count deviations multiplied by 100, not class-frequency percentages or accuracy estimates. Larger classes can contribute more count error. Exact class targets and the requested overall fraction may be impossible with indivisible groups.

Among equal best scores, code selects the lexicographically first **sorted array of group IDs**; comparisons use exact ASCII order, with a proper prefix first. `optimumCount` reports how many subsets share the best two scores. Renaming IDs can change this tie choice. The partition is deterministic, not a random sample.

The `analysis` output contains `trainRowIds`, `holdoutRowIds`, both group lists, per-class train/holdout counts, target deviations, optimum count and enumerated candidate count. `missingClasses` reports classes absent from either side. Such a result remains complete and exportable; your downstream evaluation must decide whether it is usable. Coverage checks account for every source row exactly once and report overlap only for the selected declared group IDs.

Grouping is domain-specific: [scikit-learn's grouped-data guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data) explains why related samples may need to stay on one side. This starter implements its own bounded exhaustive objective; it does not depend on or reproduce scikit-learn's splitters. It does not establish statistical representativeness, independent observations, chronological validity or model performance.

## Adapt and resume

Map your export to `dataset.json`, choose grouping candidates from dataset documentation, and join the exported row-ID lists back to the same frozen feature table. `analyzeSplit(rawInput, columnId)` in `src/model.js` is the pure calculation helper. CLI export also requires the attributed PingRoom choice. Add training or preprocessing in your application code, fitting transformations only on the intended training data.

[Randomized pilot](https://github.com/pingroom/starter-randomized-pilot) assigns interventions before observing outcomes. This starter partitions already-labeled observations into training and holdout sets.

Keep `.split/` and `.pingroom/` private and intact. Credentials, real inputs, configuration, state and outputs are ignored by Git. Input/configuration reads are limited to 64/16 KiB. Private writes are atomic under a local lock. Changed inputs, owner, room or credentials cannot reuse a session.

A lost creation response leaves `create_unknown`. Inspect and recover the original Question without resending:

```sh
npm run status
npm run find-id
npm run recover -- EXACT-QUESTION-UUID
```

An empty bounded search does not prove creation failed. For uncertain or expired pairing, inspect Connected Agents and resolve or revoke the original registration before archiving pending pairing state. Remove a stale `.lock` only after confirming its process stopped. Restore original source/session files rather than deleting state to retry.

Run commands serially. Exit codes: `0` complete/local result; `1` invalid setup/evidence or refused export; `2` incomplete/Unknown/failed answer; `3` ready/pending/interrupted; `4` uncertain creation.

Tests cover an independent recursive partition oracle, the maximum 16-group case, invalid metadata, input permutation, durable recovery, pairing and CLI subprocesses. GitHub Actions checks Node.js 22 and 24. MIT licensed.
