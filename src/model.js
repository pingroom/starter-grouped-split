import { createHash } from 'node:crypto';

export const TTL = 900;
export const CLOCK_SKEW_MS = 60000;
export const TOKEN = /^[^\s\x00-\x1f\x7f]{1,8192}(?![\s\S])/;
export const ROOM = /^[A-Z0-9]{12}$(?![\s\S])/;
export const isId = (value) => typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$(?![\s\S])/.test(value);
const record = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
const keys = (value, expected) => {
  if (!record(value)) return false;
  const actual = Object.keys(value).sort(), wanted = [...expected].sort();
  return actual.length === wanted.length && actual.every((key, index) => key === wanted[index]);
};
const word = (value) => typeof value === 'string' && /^[a-z][a-z0-9_-]{0,39}$(?![\s\S])/.test(value);
const line = (value, max) => typeof value === 'string' && value.length > 0 && value.length <= max && value.isWellFormed()
  && !/[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]/u.test(value) && value.trim() === value;
const canonical = (value) => Array.isArray(value) ? value.map(canonical) : record(value) ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
export const hash = (value) => createHash('sha256').update(JSON.stringify(canonical(value))).digest('hex');

export function iso(raw) {
  if (typeof raw !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)$(?![\s\S])/.test(raw)) throw new Error('Expected a UTC timestamp.');
  const time = Date.parse(raw);
  if (!Number.isFinite(time) || new Date(time).toISOString().slice(0, 19) !== raw.slice(0, 19)) throw new Error('Invalid UTC timestamp.');
  return new Date(time).toISOString();
}



const safeId = value => word(value) && value.length <= 12 && !['constructor', 'prototype', '__proto__', 'unknown'].includes(value);
const order = (a, b) => a < b ? -1 : a > b ? 1 : 0;
export const POLICY = Object.freeze({ name: 'whole-group-holdout-v1',
  bounds: '2..128 rows, 1..8 classes, 1..3 candidate grouping columns, 2..16 groups per column',
  candidates: 'every nonempty proper subset of groups in the selected column',
  objective: 'lexicographically minimize [sum_c abs(100*holdoutClassCount[c] - percent*classCount[c]), abs(100*holdoutRows - percent*rows)]',
  tie: 'choose the lexicographically first sorted array of holdout group IDs; proper prefix sorts first; report all optimum count',
  identifiers: 'case-sensitive exact declared IDs; every row occurs on exactly one side; selected groups never cross sides',
  selection: 'one owner-declared grouping column; Unknown withholds the partition',
  excluded: 'entity discovery, random sampling, representativeness, independence certification, chronological splitting and model training' });
export const POLICY_HASH = hash(POLICY);
const promptFor = input => `${input.title}\n${input.context}\nWhich grouping column keeps related rows together for ${input.datasetId}?\n${input.groupingColumns.map(column => `${column.id}: ${column.label} — ${column.meaning}`).join('\n')}\nChoose from the dataset documentation, or Unknown. Row metadata stays local.`;
export function validateInput(input) {
  if (!keys(input, ['datasetId', 'title', 'context', 'holdoutPercent', 'groupingColumns', 'rows']) || !word(input.datasetId)
    || !line(input.title, 40) || !line(input.context, 80) || !Number.isSafeInteger(input.holdoutPercent) || input.holdoutPercent < 1 || input.holdoutPercent > 99
    || !Array.isArray(input.rows) || input.rows.length < 2 || input.rows.length > 128
    || !Array.isArray(input.groupingColumns) || input.groupingColumns.length < 1 || input.groupingColumns.length > 3) throw new Error('Supply dataset metadata, a holdout percentage from 1 to 99, two to 128 rows and one to three grouping columns.');
  const ids = new Set();
  const groupingColumns = Array.from(input.groupingColumns, column => {
    if (!keys(column, ['id', 'label', 'meaning']) || !safeId(column.id) || ids.has(column.id) || !line(column.label, 26) || !line(column.meaning, 48)) throw new Error('Grouping columns need unique safe IDs, labels and meanings.');
    ids.add(column.id); return { ...column };
  }).sort((a, b) => order(a.id, b.id));
  const rowIds = new Set(), classes = new Set(), columnIds = groupingColumns.map(column => column.id);
  const rows = Array.from(input.rows, row => {
    if (!keys(row, ['id', 'classId', 'groups']) || !safeId(row.id) || rowIds.has(row.id) || !safeId(row.classId)
      || !keys(row.groups, columnIds) || columnIds.some(id => !safeId(row.groups[id]))) throw new Error('Rows need unique safe IDs, class IDs and one exact group value for every declared column.');
    rowIds.add(row.id); classes.add(row.classId);
    return { id: row.id, classId: row.classId, groups: Object.fromEntries(columnIds.map(id => [id, row.groups[id]])) };
  }).sort((a, b) => order(a.id, b.id));
  if (classes.size > 8) throw new Error('Use one to eight observed classes.');
  for (const column of groupingColumns) {
    const count = new Set(rows.map(row => row.groups[column.id])).size;
    if (count < 2 || count > 16) throw new Error('Each candidate column must contain two to 16 distinct groups.');
  }
  const result = { datasetId: input.datasetId, title: input.title, context: input.context, holdoutPercent: input.holdoutPercent, groupingColumns, rows };
  if (promptFor(result).length > 500) throw new Error('Grouping-column selection prompt exceeds 500 UTF-16 units.');
  return result;
}
export function validateConfig(config) {
  if (!keys(config, ['participant']) || !keys(config.participant, ['id', 'label']) || !isId(config.participant.id) || !line(config.participant.label, 60)) throw new Error('Config requires an exact dataset owner UUID and label.');
  return structuredClone(config);
}
export const itemsFor = rawInput => { validateInput(rawInput); return [{ id: 'grouping' }]; };
export function requestFor(rawInput, config, itemId, attemptId) {
  const input = validateInput(rawInput); validateConfig(config);
  if (itemId !== 'grouping' || !isId(attemptId)) throw new Error('Invalid grouping-column selection item or attempt.');
  const key = `split-${attemptId}`;
  return { prompt: promptFor(input), context: `To ${config.participant.id}`,
    options: [...input.groupingColumns.map(column => ({ value: column.id, label: `${column.id}: ${column.label}`, style: 'default' })), { value: 'unknown', label: 'Unknown', style: 'default' }],
    responder_scope: 'direct', target_user_id: config.participant.id, ttl: TTL, correlation_id: key,
    data: { input_hash: hash(input), columns_hash: hash(input.groupingColumns), policy_hash: POLICY_HASH, item_id: itemId, attempt_id: attemptId }, idempotencyKey: key };
}
export function previewInput(rawInput, config) {
  const input = validateInput(rawInput), request = requestFor(input, config, 'grouping', '01990000-0000-7000-8000-000000000001');
  const { correlation_id, idempotencyKey, data, ...question } = request;
  return { source: input, sourceHash: hash(input), policy: POLICY, policyHash: POLICY_HASH, question,
    note: 'The attempt UUID, correlation ID and input/column/policy hashes are bound when a session creates its Question.' };
}
const compareArrays = (a, b) => {
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return order(a[i], b[i]);
  return a.length - b.length;
};

/** Pure bounded partition calculation; CLI export additionally requires attributed owner evidence. */
export function analyzeSplit(rawInput, columnId) {
  const input = validateInput(rawInput), column = input.groupingColumns.find(item => item.id === columnId);
  if (!column) throw new Error('Select a declared grouping column before partitioning.');
  const classes = [...new Set(input.rows.map(row => row.classId))].sort(order);
  const groupIds = [...new Set(input.rows.map(row => row.groups[columnId]))].sort(order);
  const groups = groupIds.map(id => ({ id, rows: input.rows.filter(row => row.groups[columnId] === id) }));
  const totalCounts = classes.map(id => input.rows.filter(row => row.classId === id).length);
  const counts = groups.map(group => classes.map(id => group.rows.filter(row => row.classId === id).length));
  const limit = 2 ** groups.length - 1;
  let bestScore = null, bestGroups = null, bestCounts = null, optimumCount = 0;
  for (let mask = 1; mask < limit; mask++) {
    const heldGroups = [], heldCounts = classes.map(() => 0);
    for (let i = 0; i < groups.length; i++) if (mask & (1 << i)) {
      heldGroups.push(groups[i].id);
      for (let c = 0; c < classes.length; c++) heldCounts[c] += counts[i][c];
    }
    const heldTotal = heldCounts.reduce((sum, count) => sum + count, 0);
    const score = [heldCounts.reduce((sum, count, c) => sum + Math.abs(100 * count - input.holdoutPercent * totalCounts[c]), 0),
      Math.abs(100 * heldTotal - input.holdoutPercent * input.rows.length)];
    const comparison = bestScore === null ? -1 : compareArrays(score, bestScore);
    if (comparison < 0) { bestScore = score; bestGroups = heldGroups; bestCounts = heldCounts; optimumCount = 1; }
    else if (comparison === 0) {
      optimumCount++;
      if (compareArrays(heldGroups, bestGroups) < 0) { bestGroups = heldGroups; bestCounts = heldCounts; }
    }
  }
  const holdoutSet = new Set(bestGroups);
  const holdoutRowIds = input.rows.filter(row => holdoutSet.has(row.groups[columnId])).map(row => row.id);
  const trainRowIds = input.rows.filter(row => !holdoutSet.has(row.groups[columnId])).map(row => row.id);
  const trainGroupIds = groupIds.filter(id => !holdoutSet.has(id));
  const classCounts = classes.map((classId, c) => ({ classId, total: totalCounts[c], holdout: bestCounts[c], train: totalCounts[c] - bestCounts[c],
    holdoutTargetScaled: input.holdoutPercent * totalCounts[c], absoluteDeviationScaled: Math.abs(100 * bestCounts[c] - input.holdoutPercent * totalCounts[c]) }));
  const coverage = { sourceRows: input.rows.length, trainRows: trainRowIds.length, holdoutRows: holdoutRowIds.length,
    eachRowExactlyOnce: new Set([...trainRowIds, ...holdoutRowIds]).size === input.rows.length && trainRowIds.length + holdoutRowIds.length === input.rows.length,
    overlappingSelectedGroupIds: trainGroupIds.filter(id => holdoutSet.has(id)) };
  if (!coverage.eachRowExactlyOnce || coverage.overlappingSelectedGroupIds.length || !trainRowIds.length || !holdoutRowIds.length) throw new Error('Partition invariants failed.');
  return { columnId, columnLabel: column.label, columnMeaning: column.meaning, holdoutPercent: input.holdoutPercent,
    trainRowIds, holdoutRowIds, trainGroupIds, holdoutGroupIds: bestGroups, classCounts,
    objective: { classDeviationScaled: bestScore[0], totalSizeDeviationScaled: bestScore[1], scale: 100 },
    optimumCount, candidatesEvaluated: limit - 1, coverage,
    missingClasses: { train: classCounts.filter(item => item.train === 0).map(item => item.classId), holdout: classCounts.filter(item => item.holdout === 0).map(item => item.classId) },
    sourceHash: hash(input), policyHash: POLICY_HASH, randomSample: false, modelTrained: false,
    note: 'Disjointness covers only exact IDs in the selected column; it does not establish independence or rule out other leakage.' };
}
export function splitPlan(rawInput, choice) {
  const input = validateInput(rawInput);
  if (![null, 'unknown', ...input.groupingColumns.map(column => column.id)].includes(choice)) throw new Error('Supply a listed grouping column, Unknown or a missing choice.');
  return { status: choice === null ? 'incomplete' : choice === 'unknown' ? 'unknown' : 'complete', analysis: choice === null || choice === 'unknown' ? null : analyzeSplit(input, choice) };
}

export function validateState(state, rawInput) {
  const input = validateInput(rawInput);
  if (!keys(state, ['version', 'inputHash', 'config', 'room', 'credentialHash', 'startedAt', 'attempts']) || state.version !== 1
    || state.inputHash !== hash(input) || typeof state.room !== 'string' || !ROOM.test(state.room)
    || typeof state.credentialHash !== 'string' || !/^[0-9a-f]{64}$(?![\s\S])/.test(state.credentialHash)
    || !Array.isArray(state.attempts) || state.attempts.length > itemsFor(input).length || iso(state.startedAt) !== state.startedAt) throw new Error('Invalid state or changed dataset metadata. Restore the original input and configuration.');
  validateConfig(state.config);
  const items = itemsFor(input), ids = new Set(), questionIds = new Set(); let previous = state.startedAt, askerId;
  for (const [index, attempt] of state.attempts.entries()) {
    if (!keys(attempt, ['id', 'itemId', 'createdAt', 'request', 'questionId', 'status', 'observation']) || !isId(attempt.id) || ids.has(attempt.id)
      || attempt.itemId !== items[index].id || iso(attempt.createdAt) !== attempt.createdAt || attempt.createdAt < previous
      || hash(attempt.request) !== hash(requestFor(input, state.config, attempt.itemId, attempt.id))
      || !['creating', 'pending', 'answered', 'expired', 'cancelled', 'invalid_answer'].includes(attempt.status)) throw new Error('Invalid saved group-column selection or order.');
    ids.add(attempt.id); previous = attempt.createdAt;
    if (attempt.status === 'creating') {
      if (attempt.questionId !== null || attempt.observation !== null) throw new Error('Invalid uncertain creation state.');
    } else {
      if (!isId(attempt.questionId) || questionIds.has(attempt.questionId)) throw new Error('Invalid or repeated Question ID.');
      questionIds.add(attempt.questionId); validateObservation(attempt.observation, attempt.status, attempt.request);
      if (attempt.observation.checkedAt < attempt.createdAt) throw new Error('Observation predates its saved creation intent.');
      previous = attempt.observation.checkedAt;
      if (Date.parse(attempt.observation.createdAt) < Date.parse(attempt.createdAt) - CLOCK_SKEW_MS) throw new Error('Question predates its saved creation intent.');
      if (askerId !== undefined && attempt.observation.askerId !== askerId) throw new Error('Questions belong to different asking registrations.');
      askerId = attempt.observation.askerId;
    }
    if (['creating', 'pending'].includes(attempt.status) && index !== state.attempts.length - 1) throw new Error('Only the last confirmation may have an active Question.');
  }
  return state;
}




export function report(state, rawInput) {
  const input = validateInput(rawInput); validateState(state, input);
  const attempt = state.attempts[0], answer = attempt?.observation?.answer;
  const answers = [{ itemId: 'grouping', status: attempt?.status === 'creating' ? 'create_unknown' : attempt?.status ?? 'unasked',
    choice: answer?.value ?? null, label: answer?.label ?? null, questionId: attempt?.questionId ?? null, attemptId: attempt?.id ?? null,
    responderId: answer?.responderId ?? null, answeredAt: answer?.answeredAt ?? null, askerId: attempt?.observation?.askerId ?? null, observedAt: attempt?.observation?.checkedAt ?? null }];
  const calculation = splitPlan(input, answers[0].choice);
  const status = ['create_unknown', 'pending'].includes(answers[0].status) ? answers[0].status : answers[0].status === 'unasked' ? 'ready' : calculation.status;
  return { version: 1, datasetId: input.datasetId, sourceHash: hash(input), source: input, policy: POLICY, policyHash: POLICY_HASH,
    participant: state.config.participant, room: state.room, configHash: hash(state.config), answers, ...calculation, status };
}

function validateObservation(observation, status, request) {
  if (!keys(observation, ['createdAt', 'expiresAt', 'checkedAt', 'askerId', 'resolvedAt', 'answer']) || !isId(observation.askerId)) throw new Error('Invalid saved observation.');
  for (const key of ['createdAt', 'expiresAt', 'checkedAt']) if (iso(observation[key]) !== observation[key]) throw new Error('Invalid saved timestamp.');
  const duration = Date.parse(observation.expiresAt) - Date.parse(observation.createdAt);
  if (duration < TTL * 1000 - 1000 || duration > TTL * 1000 + 1000) throw new Error('Question lifetime differs from the saved request.');
  const latest = Date.parse(observation.checkedAt) + CLOCK_SKEW_MS;
  if (Date.parse(observation.createdAt) > latest) throw new Error('Question creation is implausibly in the future. Check the local clock.');
  if (status === 'pending') {
    if (observation.resolvedAt !== null) throw new Error('Pending Question has a resolution timestamp.');
  } else if (iso(observation.resolvedAt) !== observation.resolvedAt || observation.resolvedAt < observation.createdAt
    || Date.parse(observation.resolvedAt) > latest
    || (status === 'expired' ? observation.resolvedAt < observation.expiresAt : observation.resolvedAt > observation.expiresAt)) throw new Error('Invalid Question resolution time.');
  if (status === 'answered') {
    const answer = observation.answer;
    if (!keys(answer, ['value', 'label', 'responderId', 'answeredAt']) || answer.responderId !== request.target_user_id
      || !request.options.some((option) => option.value === answer.value && option.label === answer.label)
      || iso(answer.answeredAt) !== answer.answeredAt || answer.answeredAt !== observation.resolvedAt
      || answer.answeredAt < observation.createdAt || answer.answeredAt > observation.expiresAt) throw new Error('Invalid saved group-column selection evidence.');
  } else if (observation.answer !== null) throw new Error('Unanswered group-column selections must not contain group-column selection evidence.');
}

export function observeQuestion(q, request, questionId, room, checkedAt) {
  if (!isId(questionId) || q?.id !== questionId || q.kind !== 'question' || (q.is_handoff !== undefined && q.is_handoff !== false)
    || q.asker?.type !== 'agent' || !isId(q.asker?.id) || q.prompt !== request.prompt || q.context !== request.context
    || q.responder_scope !== 'direct' || q.target_user_id !== request.target_user_id || q.resolution_policy !== 'first'
    || (q.results !== undefined && q.results !== null) || q.text_input !== null || q.correlation_id !== request.correlation_id || q.reply_to !== null || hash(q.data) !== hash(request.data)
    || (q.room_code ?? q.room?.code) !== room || q.room_code !== undefined && q.room_code !== room
    || q.room !== undefined && q.room?.code !== room || !['pending', 'answered', 'expired', 'cancelled'].includes(q.state)
    || !Array.isArray(q.options) || q.options.length !== request.options.length
    || Array.from(q.options).some((option, index) => option?.value !== request.options[index].value || option?.label !== request.options[index].label
      || (option?.primary !== undefined && option.primary !== false) || (option?.destructive !== undefined && option.destructive !== false))) throw new Error('Question does not match the saved configuration, room, or reviewer.');
  const observation = { createdAt: iso(q.created_at), expiresAt: iso(q.expires_at), checkedAt: iso(checkedAt), askerId: q.asker.id,
    resolvedAt: q.resolved_at === null ? null : iso(q.resolved_at), answer: null };
  let status = q.state === 'answered' ? 'invalid_answer' : q.state;
  if (q.state !== 'answered' && q.answer !== null) throw new Error('Unresolved Question includes an unexpected answer.');
  if (q.state === 'answered') {
    const answer = q.answer;
    try {
      if (!answer || answer.text !== null || answer.responder?.id !== request.target_user_id
        || !request.options.some((option) => option.value === answer.value && option.label === answer.label)) throw new Error('Invalid answer.');
      observation.answer = { value: answer.value, label: answer.label, responderId: answer.responder.id, answeredAt: iso(answer.answered_at) };
      validateObservation(observation, 'answered', request); status = 'answered';
    } catch { observation.answer = null; }
  }
  validateObservation(observation, status, request);
  return { status, observation };
}

export function assertParticipantRoom(rooms, room, participantId) {
  if (!Array.isArray(rooms) || rooms.length > 1000) throw new Error('Invalid room inventory.');
  const matches = rooms.filter((item) => item?.invite_code === room), selected = matches[0];
  if (matches.length !== 1 || selected.type !== 'standard' || selected.is_public !== false || !Array.isArray(selected.members)
    || selected.members.length > 1000 || !selected.members.some((member) => member?.user_id === participantId)) throw new Error('Use a granted standard private room containing the named participant.');
}
