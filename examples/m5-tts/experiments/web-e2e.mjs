import { spawn } from 'node:child_process';
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  `--remote-debugging-port=9336`, '--no-first-run', '--no-default-browser-check',
  '--no-sandbox', '--user-data-dir=/tmp/chrome-e2e-profile2', '--window-size=900,760',
  '--autoplay-policy=no-user-gesture-required', 'about:blank',
], { stdio: 'ignore' });
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function main() {
  let target;
  for (let i = 0; i < 20; i++) {
    try {
      const list = await (await fetch('http://127.0.0.1:9336/json/list')).json();
      target = list.find(t => t.type === 'page');
      if (target) break;
    } catch {}
    await sleep(500);
  }
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  let id = 0; const pending = new Map();
  ws.onmessage = e => { const m = JSON.parse(e.data); if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } };
  await new Promise(r => ws.onopen = r);
  const send = (m, p = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method: m, params: p })); });
  await send('Runtime.enable'); await send('Page.enable');
  const errors = [];
  ws.addEventListener('message', e => {
    const m = JSON.parse(e.data);
    if (m.method === 'Runtime.exceptionThrown') errors.push(JSON.stringify(m.params.exceptionDetails?.text || m.params));
    if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error')
      errors.push(m.params.args.map(a => a.value || a.description).join(' '));
  });
  await send('Page.navigate', { url: 'http://127.0.0.1:18080' });
  await sleep(3000);
  const ev = (expr, t = 120000) => send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true, timeout: t })
    .then(m => m.result?.result?.value ?? JSON.stringify(m.result ?? m));

  console.log('== voices after load ==');
  console.log(await ev(`document.getElementById('voice').innerHTML`));

  console.log('== synthesize with server voice (stats check) ==');
  console.log(await ev(`(async () => {
    document.getElementById('goBtn').click();
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const hidden = document.getElementById('stats').hidden;
      if (!hidden) break;
    }
    const g = x => document.getElementById(x).textContent;
    return JSON.stringify({ rtf: g('sRtf'), audio: g('sAudio'), first: g('sFirst'), frames: g('sFrames'), wall: g('sWall') });
  })()`));

  console.log('== clone via synthetic wav file ==');
  console.log(await ev(`(async () => {
    const sr = 48000, n = sr * 3, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
    const ascii = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    ascii(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); ascii(8, 'WAVE');
    ascii(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    ascii(36, 'data'); v.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.round(Math.sin(2*Math.PI*220*i/sr) * 12000), true);
    const dt = new DataTransfer();
    dt.items.add(new File([buf], 'tone.wav', { type: 'audio/wav' }));
    const input = document.getElementById('recFile');
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
    await new Promise(r => setTimeout(r, 1200));
    document.getElementById('recName').value = '本机测试';
    document.getElementById('recText').value = '这是一段测试。';
    document.getElementById('recCloneBtn').click();
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 2000));
      const t = document.getElementById('log').textContent;
      if (t.includes('克隆成功') || t.includes('克隆失败')) break;
    }
    return document.getElementById('voice').innerHTML;
  })()`));

  console.log('== synthesize with local voice (inline codes) ==');
  console.log(await ev(`(async () => {
    document.getElementById('voice').value = 'loc:本机测试';
    document.getElementById('goBtn').click();
    for (let i = 0; i < 40; i++) {
      await new Promise(r => setTimeout(r, 2000));
      if (!document.getElementById('stats').hidden) break;
    }
    const g = x => document.getElementById(x).textContent;
    return JSON.stringify({ rtf: g('sRtf'), audio: g('sAudio'), first: g('sFirst'), frames: g('sFrames') })
      + ' | localStorage=' + (localStorage.getItem('m5-voices-v2') || '').length + ' bytes';
  })()`));

  console.log('== js errors ==');
  console.log(errors.length ? errors.join('\n') : '(none)');
  ws.close(); chrome.kill();
}
main().catch(e => { console.error('E2E FAILED:', e.message); chrome.kill(); process.exit(1); });
