/* Edge Security Camera - runs fully in the browser (face-api.js / TensorFlow.js).
   Pipeline: webcam -> face detect (TinyFaceDetector) -> 128-d embedding -> match vs enrolled
   -> tracker + zone rules (green/red polygons) + persistence (N consecutive violating frames) -> alert.
   Nothing is uploaded. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const STORE = 'edge-security-people-v1', ZSTORE = 'edge-security-zones-v1';
  const COL = { ok: '#3fb950', bad: '#f85149', warn: '#f0a020', neutral: '#9aa5b1' };
  const UNKNOWN = '\u0000unknown';

  const video = $('video'), overlay = $('overlay'), octx = overlay.getContext('2d');
  const zoneCv = $('zones'), zctx = zoneCv.getContext('2d');
  const state = {
    running: false, mode: 'rec', armed: true, thr: 0.5, minFrames: 6, cooldown: 8000,
    people: {}, tracks: [], nextId: 1, roster: new Map(), alerts: [], alertTotal: 0,
    lastAlert: -1e9, enroll: null, fps: 0, ms: 0, ready: false,
    zones: [], allowRed: false, drawing: null, // zones: [{type:'green'|'red', pts:[[nx,ny]...]}] in on-screen (mirrored) coords, 0..1
  };
  window.__edge = { state }; // handy for debugging / automated tests

  // ---------- persistence ----------
  function loadPeople() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORE) || '{}');
      for (const [n, arr] of Object.entries(raw)) state.people[n] = arr.map((a) => Float32Array.from(a));
    } catch (e) { state.people = {}; }
  }
  function savePeople() {
    try {
      const out = {};
      for (const [n, arr] of Object.entries(state.people)) out[n] = arr.map((f) => Array.from(f));
      localStorage.setItem(STORE, JSON.stringify(out));
    } catch (e) { setStatus('Could not save to browser storage (private mode?). Use Export to keep faces.'); }
  }

  function loadZones() {
    try {
      const z = JSON.parse(localStorage.getItem(ZSTORE) || '{}');
      if (Array.isArray(z.zones)) state.zones = z.zones.filter((q) => (q.type === 'green' || q.type === 'red') && Array.isArray(q.pts) && q.pts.length >= 3);
      state.allowRed = !!z.allowRed;
    } catch (e) { state.zones = []; }
    $('zAllowRed').checked = state.allowRed;
  }
  function saveZones() {
    try { localStorage.setItem(ZSTORE, JSON.stringify({ zones: state.zones, allowRed: state.allowRed })); } catch (e) { /* optional */ }
  }

  // ---------- zones: geometry + rules ----------
  function pointInPoly(x, y, pts) { // ray casting
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const [xi, yi] = pts[i], [xj, yj] = pts[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }
  // Which zone is the person in? Uses the centre of the face box, in on-screen (mirrored) coordinates.
  function zoneAt(t) {
    if (!state.zones.length) return 'none';
    const nx = (overlay.width - t.cx) / overlay.width, ny = t.cy / overlay.height;
    let hit = null;
    for (const z of state.zones) if (pointInPoly(nx, ny, z.pts)) { if (z.type === 'red') return 'red'; hit = 'green'; }
    return hit || 'out'; // red wins where zones overlap
  }
  // The rules: green = enrolled people are authorised; red = restricted; not enrolled = unauthorised in both;
  // outside both zones nobody is flagged; no zones drawn = whole frame behaves like a green zone.
  function verdictOf(t) {
    const known = t.name != null, z = t.zone;
    if (state.mode === 'det') return { kind: 'warn', label: 'person' + (z === 'red' || z === 'green' ? ` \u00b7 ${z} zone` : ''), badge: 'Detected', violation: false };
    if (z === 'out') return { kind: 'neutral', label: `${known ? t.name : 'Unknown'} \u00b7 outside zones`, badge: 'Outside zones', violation: false };
    if (z === 'red') {
      if (!known) return { kind: 'bad', label: 'UNAUTHORISED', badge: 'Unauthorised', violation: true, reason: 'Unauthorised person in RED zone' };
      if (state.allowRed) return { kind: 'warn', label: `AUTHORISED (RED ZONE): ${t.name}`, badge: 'Authorised \u00b7 red zone', violation: false };
      return { kind: 'bad', label: `NOT ALLOWED IN RED ZONE: ${t.name}`, badge: 'Not allowed here', violation: true, reason: `${t.name} entered RED zone (not allowed)` };
    }
    if (known) return { kind: 'ok', label: `AUTHORISED: ${t.name}`, badge: 'Authorised', violation: false };
    return { kind: 'bad', label: 'UNAUTHORISED', badge: 'Unauthorised', violation: true, reason: 'Unauthorised person' + (z === 'green' ? ' in GREEN zone' : '') };
  }
  function evaluate(tracks) {
    for (const t of tracks) {
      t.zone = zoneAt(t); t.verdict = verdictOf(t);
      t.viol = t.verdict.violation ? (t.viol || 0) + 1 : 0;
      if (!t.viol) t.alerted = false; // re-arm once the person stops violating
    }
  }

  // ---------- tiny helpers ----------
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  const clock = (t) => new Date(t).toLocaleTimeString();
  function setStatus(msg) { $('statusMsg').textContent = msg || ''; }
  function setChip(id, text, kind) { const c = $(id); c.textContent = text; c.className = 'chip' + (kind ? ' ' + kind : ''); }

  // ---------- audio ----------
  let audio;
  function beep() {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      [880, 660, 880].forEach((f, i) => {
        const o = audio.createOscillator(), g = audio.createGain();
        o.frequency.value = f; o.type = 'square'; g.gain.value = 0.08;
        o.connect(g); g.connect(audio.destination);
        o.start(audio.currentTime + i * 0.18); o.stop(audio.currentTime + i * 0.18 + 0.12);
      });
    } catch (e) { /* audio is optional */ }
  }

  // ---------- recognition ----------
  const detOpts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

  function identify(desc) {
    let best = null, bestD = Infinity;
    for (const [name, feats] of Object.entries(state.people)) {
      for (const f of feats) {
        const d = faceapi.euclideanDistance(desc, f);
        if (d < bestD) { bestD = d; best = name; }
      }
    }
    return { name: bestD <= state.thr ? best : null, dist: bestD };
  }

  function updateTracks(dets, now) {
    for (const t of state.tracks) t.matched = false;
    for (const d of dets) {
      const cx = d.box.x + d.box.width / 2, cy = d.box.y + d.box.height / 2;
      let best = null, bestDist = d.box.width * 1.5;
      for (const t of state.tracks) {
        const dist = Math.hypot(t.cx - cx, t.cy - cy);
        if (!t.matched && dist < bestDist) { best = t; bestDist = dist; }
      }
      if (!best) {
        best = { id: state.nextId++, hist: [], unk: 0, alerted: false };
        state.tracks.push(best);
      }
      Object.assign(best, { cx, cy, box: d.box, dist: d.dist, seen: now, matched: true });
      best.hist.push(d.name); if (best.hist.length > 7) best.hist.shift();
      best.unk = d.name == null ? best.unk + 1 : 0;
      const votes = new Map();
      for (const h of best.hist) votes.set(h ?? UNKNOWN, (votes.get(h ?? UNKNOWN) || 0) + 1);
      const top = [...votes.entries()].sort((a, b) => b[1] - a[1])[0][0];
      best.name = top === UNKNOWN ? null : top;
    }
    state.tracks = state.tracks.filter((t) => now - t.seen < 1000);
    return state.tracks.filter((t) => t.matched);
  }

  // ---------- drawing ----------
  function drawTracks(tracks) {
    const W = overlay.width, H = overlay.height;
    octx.clearRect(0, 0, W, H);
    octx.lineWidth = Math.max(2, W / 320);
    octx.font = `600 ${Math.round(Math.max(13, W / 38))}px system-ui, sans-serif`;
    octx.textBaseline = 'middle';
    for (const t of tracks) {
      const color = COL[t.verdict.kind];
      const b = t.box, x = W - (b.x + b.width); // mirrored to match the flipped video
      octx.strokeStyle = color;
      octx.strokeRect(x, b.y, b.width, b.height);
      const label = t.verdict.label;
      const th = parseInt(octx.font.match(/(\d+)px/)[1], 10) + 8, tw = octx.measureText(label).width + 12;
      const ly = b.y > th + 4 ? b.y - th : Math.min(b.y + b.height, H - th);
      octx.fillStyle = 'rgba(0,0,0,.75)'; octx.fillRect(x, ly, tw, th);
      octx.fillStyle = color; octx.fillText(label, x + 6, ly + th / 2 + 1);
    }
  }

  function snapshot() {
    const c = document.createElement('canvas');
    c.width = overlay.width; c.height = overlay.height;
    const g = c.getContext('2d');
    g.save(); g.translate(c.width, 0); g.scale(-1, 1); g.drawImage(video, 0, 0, c.width, c.height); g.restore();
    g.drawImage(zoneCv, 0, 0); g.drawImage(overlay, 0, 0);
    return c.toDataURL('image/jpeg', 0.7);
  }

  // ---------- alerts ----------
  function raiseAlert(track, now) {
    state.lastAlert = now; track.alerted = true; state.alertTotal++;
    const reason = track.verdict.reason || 'Unauthorised person';
    const entry = { id: state.alertTotal, time: Date.now(), dist: track.dist, reason, img: snapshot() };
    state.alerts.unshift(entry); state.alerts.length = Math.min(state.alerts.length, 20);
    beep(); renderAlerts();
    $('banner').textContent = 'INTRUDER ALERT \u2014 ' + reason.toUpperCase();
    $('banner').hidden = false; $('stage').classList.add('alarm');
    clearTimeout(raiseAlert.t);
    raiseAlert.t = setTimeout(() => { $('banner').hidden = true; $('stage').classList.remove('alarm'); }, 3000);
  }

  // ---------- enrollment ----------
  function startEnroll(name) {
    state.enroll = { name, left: 8, next: 0 };
    $('btnEnroll').disabled = true;
    $('enrollMsg').textContent = `Enrolling "${name}": look at the camera (8 samples left)`;
  }
  function enrollStep(dets, now) {
    const e = state.enroll; if (!e || now < e.next) return;
    if (dets.length === 1 && dets[0].score > 0.6 && dets[0].desc) {
      (state.people[e.name] = state.people[e.name] || []).push(Float32Array.from(dets[0].desc));
      e.left--; e.next = now + 400;
      if (e.left > 0) $('enrollMsg').textContent = `Enrolling "${e.name}": ${e.left} sample${e.left > 1 ? 's' : ''} left`;
      else {
        $('enrollMsg').textContent = `Enrolled "${e.name}". Try it: step away, then have someone else step in.`;
        state.enroll = null; $('btnEnroll').disabled = !state.running || state.mode !== 'rec';
      }
      savePeople(); renderPeople();
    } else if (dets.length > 1) $('enrollMsg').textContent = 'Only one face at a time please.';
  }

  // ---------- rendering of lists ----------
  function renderPeople() {
    const list = $('peopleList'); list.replaceChildren();
    const names = Object.keys(state.people).sort();
    for (const n of names) {
      const li = el('li'); li.append(el('span', 'name', n), el('span', 'meta', `${state.people[n].length} samples`));
      const rm = el('button', null, 'Remove'); rm.setAttribute('aria-label', `Remove ${n}`);
      rm.onclick = () => { delete state.people[n]; savePeople(); renderPeople(); };
      li.append(rm); list.append(li);
    }
    $('peopleCount').textContent = names.length; $('peopleEmpty').hidden = names.length > 0;
  }
  function renderAlerts() {
    const list = $('alertList'); list.replaceChildren();
    for (const a of state.alerts) {
      const li = el('li'), img = el('img'); img.src = a.img; img.alt = `Snapshot of alert ${a.id}`;
      const info = el('span', 'name', a.reason);
      const meta = el('span', 'meta', `${clock(a.time)} · ${Number.isFinite(a.dist) ? 'd=' + a.dist.toFixed(2) : 'no enrolled match'}`);
      const dl = el('a', null, 'Save'); dl.href = a.img; dl.download = `intruder_${a.time}.jpg`;
      li.append(img, info, meta, dl); list.append(li);
    }
    $('alertCount').textContent = state.alertTotal; $('alertEmpty').hidden = state.alerts.length > 0;
    setChip('chipAlerts', `Alerts ${state.alertTotal}`, state.alertTotal ? 'bad' : '');
  }
  const label = (t) => (state.mode === 'det' ? `Person #${t.id}` : t.name != null ? t.name : `Unknown #${t.id}`);
  const statusOf = (t) => [t.verdict.badge, t.verdict.kind];

  let lastSig = '';
  function renderLive(tracks, now) {
    const sig = state.mode + tracks.map((t) => `${t.id}:${t.name}:${t.verdict.badge}`).join('|');
    for (const t of tracks) {
      if (t.hist.length < 3) continue;
      const key = state.mode === 'det' ? `det-${t.id}` : t.name != null ? `name-${t.name}` : `unk-${t.id}`;
      if (t.name != null) state.roster.delete(`unk-${t.id}`);
      const [st, kind] = statusOf(t);
      const r = state.roster.get(key) || { first: Date.now() };
      Object.assign(r, { label: label(t), st, kind, last: Date.now(), dist: t.dist });
      state.roster.set(key, r);
    }
    $('chipAuth').textContent = `Authorised ${tracks.filter((t) => t.name != null).length}`;
    $('chipUnk').textContent = `Unknown ${tracks.filter((t) => t.name == null).length}`;
    if (sig === lastSig && now - renderLive.t < 1000) return;
    lastSig = sig; renderLive.t = now;
    const live = $('liveList'); live.replaceChildren();
    for (const t of tracks) {
      const [st, kind] = statusOf(t), li = el('li');
      li.append(el('span', 'name', label(t)), el('span', 'badge ' + kind, st));
      const zn = t.zone === 'green' || t.zone === 'red' ? `${t.zone} zone` : '';
      const dd = state.mode === 'rec' && Number.isFinite(t.dist) ? `d=${t.dist.toFixed(2)}` : '';
      if (zn || dd) li.append(el('span', 'meta', [zn, dd].filter(Boolean).join(' \u00b7 ')));
      live.append(li);
    }
    $('liveCount').textContent = tracks.length; $('liveEmpty').hidden = tracks.length > 0;
    const ros = $('rosterList'); ros.replaceChildren();
    const rows = [...state.roster.values()].sort((a, b) => b.last - a.last).slice(0, 50);
    for (const r of rows) {
      const li = el('li'); li.append(el('span', 'name', r.label), el('span', 'badge ' + r.kind, r.st),
        el('span', 'meta', `${clock(r.first)}–${clock(r.last)}`)); ros.append(li);
    }
    $('rosterCount').textContent = state.roster.size; $('rosterEmpty').hidden = state.roster.size > 0;
  }
  renderLive.t = 0;

  // ---------- main loop ----------
  async function loop() {
    if (!state.running) return;
    const t0 = performance.now();
    let dets = [];
    try {
      if (state.mode === 'rec') {
        const r = await faceapi.detectAllFaces(video, detOpts).withFaceLandmarks().withFaceDescriptors();
        dets = r.map((x) => ({ box: x.detection.box, score: x.detection.score, desc: x.descriptor }));
        enrollStep(dets, t0);
        for (const d of dets) Object.assign(d, identify(d.desc));
      } else {
        const r = await faceapi.detectAllFaces(video, detOpts);
        dets = r.map((x) => ({ box: x.box, score: x.score, name: null, dist: NaN }));
      }
    } catch (e) { setStatus('Detection error: ' + e.message); }

    const now = performance.now();
    const tracks = updateTracks(dets, now);
    evaluate(tracks);
    drawTracks(tracks);
    if (state.mode === 'rec' && state.armed && now - state.lastAlert > state.cooldown) {
      const t = tracks.find((x) => x.viol >= state.minFrames && !x.alerted);
      if (t) raiseAlert(t, now);
    }
    renderLive(tracks, now);
    state.ms = now - t0; state.fps = state.fps ? 0.85 * state.fps + 0.15 * (1000 / Math.max(state.ms, 1)) : 1000 / Math.max(state.ms, 1);
    setChip('chipFps', `FPS ${state.fps.toFixed(1)} · ${state.ms.toFixed(0)} ms`);
    requestAnimationFrame(loop);
  }

  // ---------- camera ----------
  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus(window.isSecureContext ? 'Camera API unavailable in this browser.'
        : `Camera is blocked on http://${location.host} (not a secure origin). Open http://localhost:${location.port || 80}` +
          ' instead (not 0.0.0.0 or a LAN IP), or use HTTPS.');
      return;
    }
    $('btnStart').disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
      video.srcObject = stream; await video.play();
      overlay.width = zoneCv.width = video.videoWidth; overlay.height = zoneCv.height = video.videoHeight;
      $('stage').style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      $('placeholder').classList.add('hidden');
      state.running = true; setStatus(''); $('btnEnroll').disabled = state.mode !== 'rec';
      for (const id of ['zGreen', 'zRed', 'zSplit', 'zClear']) $(id).disabled = false;
      drawZones();
      try { audio = audio || new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { /* optional */ }
      requestAnimationFrame(loop);
    } catch (e) {
      $('btnStart').disabled = false;
      setStatus(e.name === 'NotAllowedError' ? 'Camera permission denied. Allow it in the browser address bar and retry.'
        : e.name === 'NotFoundError' ? 'No camera found.' : 'Camera error: ' + e.message);
    }
  }

  // ---------- UI wiring ----------
  function setMode(m) {
    state.mode = m; state.tracks = []; lastSig = '';
    $('modeRec').classList.toggle('on', m === 'rec'); $('modeDet').classList.toggle('on', m === 'det');
    setChip('chipMode', m === 'rec' ? 'Recognition' : 'Detection only – no identity', m === 'det' ? 'warn' : '');
    $('btnEnroll').disabled = !state.running || m !== 'rec' || !!state.enroll;
    if (m === 'det') setStatus('Detection only: the system sees a face but cannot tell who it is, so nobody is flagged.'); else setStatus('');
    octx.clearRect(0, 0, overlay.width, overlay.height);
  }
  $('modeRec').onclick = () => setMode('rec');
  $('modeDet').onclick = () => setMode('det');
  $('btnArm').onclick = () => {
    state.armed = !state.armed;
    $('btnArm').classList.toggle('on', state.armed); $('btnArm').textContent = state.armed ? 'Armed' : 'Disarmed';
    setChip('chipArm', state.armed ? 'ARMED' : 'DISARMED', state.armed ? 'ok' : 'warn');
  };
  $('thr').oninput = (e) => { state.thr = +e.target.value; $('thrOut').textContent = state.thr.toFixed(2); };
  $('minFrames').oninput = (e) => { state.minFrames = +e.target.value; $('minOut').textContent = state.minFrames; };
  $('btnStart').onclick = startCamera;
  $('enrollForm').onsubmit = (e) => {
    e.preventDefault();
    const name = $('enrollName').value.trim();
    if (name && state.running && state.mode === 'rec') startEnroll(name);
  };
  $('btnExport').onclick = () => {
    const out = {}; for (const [n, a] of Object.entries(state.people)) out[n] = a.map((f) => Array.from(f));
    const a = el('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out)], { type: 'application/json' }));
    a.download = 'authorised-persons.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $('btnImport').onclick = () => $('fileImport').click();
  $('fileImport').onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try {
      const data = JSON.parse(await f.text()); let n = 0;
      for (const [name, arr] of Object.entries(data)) {
        if (!Array.isArray(arr) || !arr.every((v) => Array.isArray(v) && v.length === 128 && v.every(Number.isFinite))) throw new Error('bad format');
        state.people[name] = (state.people[name] || []).concat(arr.map((v) => Float32Array.from(v))); n++;
      }
      savePeople(); renderPeople(); $('enrollMsg').textContent = `Imported ${n} person(s).`;
    } catch (err) { $('enrollMsg').textContent = 'Import failed: not a valid export file.'; }
    e.target.value = '';
  };
  $('btnClear').onclick = () => {
    if (Object.keys(state.people).length && confirm('Remove all enrolled persons from this browser?')) { state.people = {}; savePeople(); renderPeople(); }
  };

  // ---------- zones: drawing tool ----------
  const ZC = { green: '63,185,80', red: '248,81,73' };
  function drawZones(cursor) {
    const W = zoneCv.width, H = zoneCv.height;
    zctx.clearRect(0, 0, W, H);
    zctx.lineWidth = Math.max(2, W / 300);
    zctx.font = `700 ${Math.round(Math.max(12, W / 46))}px system-ui, sans-serif`; zctx.textBaseline = 'middle';
    const path = (pts) => { zctx.beginPath(); pts.forEach(([x, y], i) => (i ? zctx.lineTo(x * W, y * H) : zctx.moveTo(x * W, y * H))); };
    for (const z of state.zones) {
      path(z.pts); zctx.closePath();
      zctx.fillStyle = `rgba(${ZC[z.type]},.22)`; zctx.fill();
      zctx.setLineDash([10, 6]); zctx.strokeStyle = `rgb(${ZC[z.type]})`; zctx.stroke(); zctx.setLineDash([]);
      // tag sits at the zone's bottom-left corner so it never covers a face or a person's label
      const maxY = Math.max(...z.pts.map((p) => p[1])), lx = Math.min(...z.pts.filter((p) => maxY - p[1] < 0.08).map((p) => p[0])) * W;
      const txt = z.type === 'green' ? 'GREEN ZONE' : 'RED ZONE', tw = zctx.measureText(txt).width + 12, th = parseInt(zctx.font.match(/(\d+)px/)[1], 10) + 8;
      const ly = Math.max(maxY * H - th - 6, 2);
      zctx.fillStyle = 'rgba(0,0,0,.6)'; zctx.fillRect(lx + 6, ly, tw, th);
      zctx.fillStyle = `rgb(${ZC[z.type]})`; zctx.fillText(txt, lx + 12, ly + th / 2 + 1);
    }
    const d = state.drawing;
    if (d && d.pts.length) {
      const col = `rgb(${ZC[d.type]})`, pts = cursor ? [...d.pts, cursor] : d.pts;
      path(pts); zctx.strokeStyle = col; zctx.stroke();
      d.pts.forEach(([x, y], i) => { zctx.beginPath(); zctx.arc(x * W, y * H, i === 0 ? 7 : 4, 0, 7); zctx.fillStyle = col; zctx.fill(); });
    }
  }
  function startDrawing(type) {
    if (!state.running) return;
    state.drawing = { type, pts: [] };
    $('stage').classList.add('drawing');
    $('zGreen').classList.toggle('active', type === 'green'); $('zRed').classList.toggle('active', type === 'red');
    $('zUndo').disabled = false;
    $('zoneHint').textContent = `Drawing a ${type.toUpperCase()} zone: click to add corners. Click the first point, double-click or press Enter to finish (3+ points). Backspace = undo point, Esc = cancel.`;
    drawZones();
  }
  function stopDrawing() {
    state.drawing = null; $('stage').classList.remove('drawing');
    $('zGreen').classList.remove('active'); $('zRed').classList.remove('active'); $('zUndo').disabled = true;
    $('zoneHint').innerHTML = 'Optional. <b>Green zone</b>: enrolled people are authorised. <b>Red zone</b>: restricted area. Anyone not enrolled is unauthorised in either zone. Outside the zones nobody is flagged. No zones = the whole frame is monitored.';
    drawZones();
  }
  function finishDrawing() {
    const d = state.drawing; if (!d) return;
    if (d.pts.length >= 3) { state.zones.push({ type: d.type, pts: d.pts }); saveZones(); }
    stopDrawing();
  }
  const stagePoint = (ev) => {
    const r = $('stage').getBoundingClientRect();
    return [Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height))];
  };
  $('stage').addEventListener('click', (ev) => {
    const d = state.drawing; if (!d) return;
    const p = stagePoint(ev), r = $('stage').getBoundingClientRect();
    if (d.pts.length >= 3 && Math.hypot((p[0] - d.pts[0][0]) * r.width, (p[1] - d.pts[0][1]) * r.height) < 14) { finishDrawing(); return; }
    d.pts.push(p); drawZones(p);
  });
  $('stage').addEventListener('dblclick', () => { // the two clicks of a double-click each added a point: keep one
    const d = state.drawing; if (!d) return;
    d.pts.pop(); finishDrawing();
  });
  $('stage').addEventListener('mousemove', (ev) => { if (state.drawing && state.drawing.pts.length) drawZones(stagePoint(ev)); });
  document.addEventListener('keydown', (ev) => {
    if (!state.drawing || /^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) return;
    if (ev.key === 'Enter') finishDrawing();
    else if (ev.key === 'Escape') stopDrawing();
    else if (ev.key === 'Backspace') { state.drawing.pts.pop(); drawZones(); ev.preventDefault(); }
  });
  $('zGreen').onclick = () => (state.drawing && state.drawing.type === 'green' ? stopDrawing() : startDrawing('green'));
  $('zRed').onclick = () => (state.drawing && state.drawing.type === 'red' ? stopDrawing() : startDrawing('red'));
  $('zUndo').onclick = () => { if (state.drawing) { state.drawing.pts.pop(); drawZones(); } };
  $('zSplit').onclick = () => { // quick demo layout: left half green, right half red (as seen on screen)
    stopDrawing();
    state.zones = [{ type: 'green', pts: [[0.02, 0.04], [0.49, 0.04], [0.49, 0.96], [0.02, 0.96]] },
      { type: 'red', pts: [[0.51, 0.04], [0.98, 0.04], [0.98, 0.96], [0.51, 0.96]] }];
    saveZones(); drawZones();
  };
  $('zClear').onclick = () => { stopDrawing(); state.zones = []; saveZones(); drawZones(); };
  $('zAllowRed').onchange = (e) => { state.allowRed = e.target.checked; saveZones(); };

  // ---------- boot ----------
  (async function init() {
    loadPeople(); loadZones(); renderPeople(); renderAlerts(); setChip('chipArm', 'ARMED', 'ok');
    try {
      for (const b of ['webgl', 'cpu']) { // GPU if available, else CPU (skips the unshipped wasm backend)
        try { if (await faceapi.tf.setBackend(b)) break; } catch (e) { /* try next */ }
      }
      await faceapi.tf.ready();
      await Promise.all([faceapi.nets.tinyFaceDetector.loadFromUri('models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('models'), faceapi.nets.faceRecognitionNet.loadFromUri('models')]);
      $('chipBackend').textContent = 'backend: ' + faceapi.tf.getBackend();
      state.ready = true; $('btnStart').disabled = false; $('btnStart').textContent = 'Start camera';
    } catch (e) {
      $('btnStart').textContent = 'Model load failed';
      setStatus('Could not load models (' + e.message + '). Serve this folder over http(s), not file://.');
    }
  })();
})();
