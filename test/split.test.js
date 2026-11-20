import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { analyzeSplit, validateInput, requestFor, previewInput, splitPlan, hash } from '../src/model.js';
import { step } from '../src/engine.js';
import { mockApi, CONFIG, ROOM_CODE, id } from '../src/mock.js';

const example = JSON.parse(await readFile(new URL('../examples/dataset.json', import.meta.url), 'utf8'));
const fixture = () => structuredClone(example);
const dataset = (rows, holdoutPercent = 25) => ({ datasetId: 'test-data', title: 'Test data', context: 'Choose the documented entity.', holdoutPercent,
  groupingColumns: [{ id: 'entity', label: 'Entity', meaning: 'Same source object' }], rows });
const row = (id, classId, group) => ({ id, classId, groups: { entity: group } });

// Separate recursive selection and row-scanning oracle; no production aggregation or bitmask helper.
function oracle(input, column) {
  const groupIds = [...new Set(input.rows.map(item => item.groups[column]))].sort();
  const classes = [...new Set(input.rows.map(item => item.classId))].sort();
  const candidates = [];
  function visit(index, chosen) {
    if (index === groupIds.length) {
      if (chosen.length === 0 || chosen.length === groupIds.length) return;
      const held = input.rows.filter(item => chosen.includes(item.groups[column]));
      let score = 0;
      for (const category of classes) {
        const all = input.rows.filter(item => item.classId === category).length;
        const part = held.filter(item => item.classId === category).length;
        score += Math.abs(part * 100 - input.holdoutPercent * all);
      }
      candidates.push({ chosen: [...chosen], score, total: Math.abs(held.length * 100 - input.rows.length * input.holdoutPercent) });
      return;
    }
    visit(index + 1, chosen);
    visit(index + 1, [...chosen, groupIds[index]]);
  }
  visit(0, []);
  candidates.sort((a, b) => a.score - b.score || a.total - b.total || compareLex(a.chosen, b.chosen));
  const best = candidates[0];
  return { best, count: candidates.filter(item => item.score === best.score && item.total === best.total).length };
}
function compareLex(a, b) {
  for (let i = 0; i < a.length && i < b.length; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return a.length - b.length;
}
function assertPartition(input, column, result) {
  const joined = [...result.trainRowIds, ...result.holdoutRowIds];
  assert.deepEqual([...joined].sort(), input.rows.map(item => item.id).sort());
  assert.equal(new Set(joined).size, joined.length);
  const train = new Set(input.rows.filter(item => result.trainRowIds.includes(item.id)).map(item => item.groups[column]));
  const held = new Set(input.rows.filter(item => result.holdoutRowIds.includes(item.id)).map(item => item.groups[column]));
  assert.ok([...train].every(value => !held.has(value)));
  assert.ok(train.size > 0 && held.size > 0);
  for (const category of result.classCounts) assert.equal(category.total, category.train + category.holdout);
}

test('example keeps one complete card together and uses its exact class targets', () => {
  const result = analyzeSplit(fixture(), 'a');
  assert.deepEqual(result.holdoutGroupIds, ['doc1']); assert.deepEqual(result.holdoutRowIds, ['r01', 'r02']);
  assert.deepEqual(result.objective, { classDeviationScaled: 0, totalSizeDeviationScaled: 0, scale: 100 });
  assert.equal(result.optimumCount, 4); assert.equal(result.candidatesEvaluated, 14);
  assert.deepEqual(result.missingClasses, { train: [], holdout: [] }); assertPartition(fixture(), 'a', result);
});

test('choosing capture batches changes indivisible units without losing rows', () => {
  const result = analyzeSplit(fixture(), 'b');
  assert.deepEqual(result.holdoutGroupIds, ['batch3']); assert.equal(result.optimumCount, 1);
  assertPartition(fixture(), 'b', result);
});

test('enumeration matches an independent recursive row-count oracle across 372 labeled group models', () => {
  let checked = 0;
  for (let n = 2; n <= 6; n++) for (let labels = 0; labels < 2 ** n; labels++) for (const percent of [20, 50, 80]) {
    const input = dataset(Array.from({ length: n }, (_, i) => row(`r${i}`, labels & 2 ** i ? 'yes' : 'no', `g${i}`)), percent);
    const expected = oracle(input, 'entity'), result = analyzeSplit(input, 'entity');
    assert.deepEqual(result.holdoutGroupIds, expected.best.chosen);
    assert.equal(result.objective.classDeviationScaled, expected.best.score);
    assert.equal(result.objective.totalSizeDeviationScaled, expected.best.total);
    assert.equal(result.optimumCount, expected.count); assertPartition(input, 'entity', result); checked++;
  }
  assert.equal(checked, 372);
});

test('uneven groups and mixed labels match row-based objective instead of group counts', () => {
  for (const percent of [1, 17, 33, 50, 66, 99]) {
    const input = dataset([
      row('r1', 'cat', 'a'), row('r2', 'dog', 'a'), row('r3', 'cat', 'a'), row('r4', 'cat', 'b'),
      row('r5', 'dog', 'c'), row('r6', 'dog', 'c'), row('r7', 'cat', 'd'),
    ], percent);
    const result = analyzeSplit(input, 'entity'), expected = oracle(input, 'entity');
    assert.deepEqual(result.holdoutGroupIds, expected.best.chosen); assert.equal(result.optimumCount, expected.count);
    assert.equal(result.objective.totalSizeDeviationScaled, expected.best.total); assertPartition(input, 'entity', result);
  }
});

test('maximum 16 groups and 128 rows enumerate every candidate and count all 12870 optima', () => {
  const input = dataset(Array.from({ length: 128 }, (_, i) => row(`r${String(i).padStart(3, '0')}`, i % 2 ? 'dog' : 'cat', `g${String(Math.floor(i / 8)).padStart(2, '0')}`)), 50);
  const result = analyzeSplit(input, 'entity');
  assert.equal(result.candidatesEvaluated, 65534); assert.equal(result.optimumCount, 12870);
  assert.deepEqual(result.holdoutGroupIds, ['g00', 'g01', 'g02', 'g03', 'g04', 'g05', 'g06', 'g07']);
  assert.equal(result.holdoutRowIds.length, 64); assert.equal(result.objective.classDeviationScaled, 0); assertPartition(input, 'entity', result);
});

test('rare classes absent from either side remain explicit diagnostics in a complete partition', () => {
  const input = dataset([row('r1', 'cat', 'a'), row('r2', 'dog', 'b')]);
  const result = analyzeSplit(input, 'entity');
  assert.deepEqual(result.holdoutGroupIds, ['a']); assert.equal(result.optimumCount, 2);
  assert.deepEqual(result.missingClasses, { train: ['cat'], holdout: ['dog'] });
  assert.equal(result.coverage.eachRowExactlyOnce, true); assert.deepEqual(result.coverage.overlappingSelectedGroupIds, []);
});

test('rows, grouping columns and object key order canonicalize without changing the partition or hash', () => {
  const input = fixture(), reordered = fixture();
  reordered.rows.reverse(); reordered.groupingColumns.reverse();
  reordered.rows = reordered.rows.map(item => ({ groups: Object.fromEntries(Object.entries(item.groups).reverse()), classId: item.classId, id: item.id }));
  assert.deepEqual(validateInput(input), validateInput(reordered));
  assert.deepEqual(analyzeSplit(input, 'a'), analyzeSplit(reordered, 'a'));
  assert.equal(hash(validateInput(input)), hash(validateInput(reordered)));
});

test('source evidence includes exact immutable policy and only metadata is sent in the Question', () => {
  const input = fixture(), request = requestFor(input, CONFIG, 'grouping', id(77));
  assert.equal(request.data.input_hash, hash(validateInput(input))); assert.ok(request.prompt.length <= 500);
  assert.equal(request.target_user_id, CONFIG.participant.id); assert.equal(request.options.at(-1).value, 'unknown');
  assert.equal(JSON.stringify(request).includes('r01'), false); assert.equal(JSON.stringify(request).includes('doc1'), false);
  const preview = previewInput(input, CONFIG); assert.deepEqual(preview.source, validateInput(input));
  assert.equal(request.data.policy_hash, preview.policyHash); assert.equal(request.data.columns_hash, hash(validateInput(input).groupingColumns));
});

test('Unknown and missing choices withhold partitions and unlisted values are refused', () => {
  assert.deepEqual(splitPlan(fixture(), 'unknown'), { status: 'unknown', analysis: null });
  assert.deepEqual(splitPlan(fixture(), null), { status: 'incomplete', analysis: null });
  for (const choice of [undefined, '', 'other', 1, {}, 'a\n']) assert.throws(() => splitPlan(fixture(), choice));
  assert.throws(() => analyzeSplit(fixture(), 'unknown'));
});

test('strict metadata validation rejects incomplete, malformed, excessive or impossible groups', () => {
  const mutations = [
    x => { x.extra = true; }, x => { delete x.rows; }, x => { x.holdoutPercent = -0; }, x => { x.holdoutPercent = 100; },
    x => { x.holdoutPercent = 1.5; }, x => { x.holdoutPercent = '25'; }, x => { x.rows = [x.rows[0]]; },
    x => { x.rows = Array(2); }, x => { x.rows[1].id = x.rows[0].id; }, x => { x.rows[0].id = 'constructor'; },
    x => { x.rows[0].classId = 'unknown'; }, x => { x.rows[0].classId = 'cat\n'; }, x => { x.rows[0].groups.a = ''; },
    x => { x.rows[0].groups.a = 'a/b'; }, x => { x.rows[0].groups.a = 3; }, x => { x.rows[0].groups.extra = 'x'; },
    x => { delete x.rows[0].groups.b; }, x => { x.rows[0].feature = 0.5; }, x => { x.groupingColumns = []; },
    x => { x.groupingColumns = Array(2); }, x => { x.groupingColumns[1].id = 'a'; }, x => { x.groupingColumns[0].meaning = ' leading'; },
    x => { x.title = '\ud800'; }, x => { x.context = 'hidden\u200btext'; }, x => { x.datasetId = 'x\n'; },
    x => { x.rows.forEach(item => { item.groups.a = 'one'; }); }, x => { x.rows = Array.from({ length: 129 }, (_, i) => ({ ...x.rows[0], id: `r${i}` })); },
  ];
  for (const mutate of mutations) { const input = fixture(); mutate(input); assert.throws(() => validateInput(input)); }
  assert.throws(() => validateInput(dataset(Array.from({ length: 17 }, (_, i) => row(`r${i}`, 'cat', `g${i}`)))));
  assert.throws(() => validateInput(dataset(Array.from({ length: 9 }, (_, i) => row(`r${i}`, `c${i}`, `g${i}`)))));
});

test('full prompt limit is checked before state files and any mock API operation', async t => {
  const directory = await mkdtemp(join(tmpdir(), 'split-preflight-')); t.after(() => rm(directory, { recursive: true, force: true }));
  const input = fixture(); input.datasetId = 'd'.repeat(40); input.title = 'T'.repeat(40); input.context = 'C'.repeat(80);
  input.groupingColumns.push({ id: 'c', label: 'L'.repeat(26), meaning: 'M'.repeat(48) });
  input.rows.forEach(item => { item.groups.c = item.groups.a; });
  input.groupingColumns.forEach(item => { item.label = 'L'.repeat(26); item.meaning = 'M'.repeat(48); });
  const api = mockApi(() => '2026-01-01T09:00:00.000Z');
  await assert.rejects(step({ client: api.client, directory, input, config: CONFIG, token: 'fixture', room: ROOM_CODE }), /prompt exceeds/);
  assert.equal(api.calls.length, 0); await assert.rejects(access(join(directory, 'state.json')), { code: 'ENOENT' });
});
