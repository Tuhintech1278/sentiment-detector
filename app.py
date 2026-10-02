"""
Sentiment website: type a sentence, get POSITIVE / NEGATIVE / NEUTRAL.

Run:   python app.py
Open:  http://127.0.0.1:5000

Needs sentiment_classifier.py in the same folder. Trains a model automatically
on first start if sentiment_model.joblib does not exist.
"""
import __main__

import joblib
from flask import Flask, jsonify, request
from sklearn.linear_model import LogisticRegression

import sentiment_classifier as sc

# lets the saved model load (it was pickled while running as a script)
__main__.TextPreprocessor = sc.TextPreprocessor
__main__._FallbackLemmatizer = sc._FallbackLemmatizer


def get_model():
    if sc.MODEL_PATH.exists():
        print("Loading saved model...")
        return joblib.load(sc.MODEL_PATH)
    print("No saved model found - training one now (about 10 seconds)...")
    df = sc.make_demo_dataset()
    model = sc.build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced"))
    model.fit(df.text, df.label)
    joblib.dump(model, sc.MODEL_PATH)
    return model


model = get_model()
app = Flask(__name__)


@app.post("/api/classify")
def api_classify():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Please type a sentence."), 400
    return jsonify(label=model.predict([text[:2000]])[0])


@app.get("/")
def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&display=swap" rel="stylesheet">
<style>
  :root { --ink:#1a1a3d; --paper:#ffffff; --bg:#dde1ff; }
  body[data-mood="positive"] { --bg:#ffe066; }
  body[data-mood="negative"] { --bg:#ff8a7a; }
  body[data-mood="neutral"]  { --bg:#bfd0e0; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; background:var(--bg); color:var(--ink);
         font-family:'Bricolage Grotesque', system-ui, 'Segoe UI', Roboto, sans-serif;
         transition:background .6s ease; display:flex; align-items:center; }
  main { width:100%; max-width:1080px; margin:0 auto; padding:40px 28px;
         display:grid; grid-template-columns:1.1fr .9fr; gap:56px; align-items:center; }
  h1 { font-size:clamp(40px,6vw,72px); line-height:.98; font-weight:800; margin:0 0 16px; letter-spacing:-.025em; }
  .lead { font-size:19px; line-height:1.5; margin:0 0 28px; max-width:34em; }
  .field { background:var(--paper); border:3px solid var(--ink); border-radius:22px; padding:6px;
           box-shadow:6px 6px 0 var(--ink); }
  textarea { display:block; width:100%; min-height:128px; border:0; outline:0; resize:none; background:transparent;
             font:inherit; font-size:20px; line-height:1.4; color:var(--ink); padding:14px 16px; }
  textarea::placeholder { color:#1a1a3d80; }
  .bar { display:flex; justify-content:flex-end; padding:4px 6px 6px; }
  button.go { font:inherit; font-weight:600; font-size:18px; color:#fff; background:var(--ink); border:0;
              border-radius:14px; padding:12px 24px; cursor:pointer; transition:transform .12s; }
  button.go:hover { transform:translateY(-2px); } button.go:active { transform:translateY(1px); }
  button.go:disabled { opacity:.6; cursor:default; transform:none; }
  :focus-visible { outline:3px solid var(--ink); outline-offset:3px; }
  .try { margin-top:26px; font-size:15px; }
  .chips { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
  .chip { font:inherit; font-size:15px; color:var(--ink); background:transparent; border:2px solid var(--ink);
          border-radius:99px; padding:7px 15px; cursor:pointer; transition:background .15s, color .15s; }
  .chip:hover { background:var(--ink); color:#fff; }
  .stage { text-align:center; }
  svg { width:min(100%, 340px); height:auto; overflow:visible; }
  .face { fill:var(--paper); stroke:var(--ink); stroke-width:4; }
  .feat { fill:var(--ink); } .line { stroke:var(--ink); stroke-width:4.5; stroke-linecap:round; fill:none; }
  .eye { transform-box:fill-box; transform-origin:center; transition:transform .4s ease; }
  .brow { transform-box:fill-box; transform-origin:center; transition:transform .4s ease; }
  .cheek { fill:#ff8a7a; opacity:0; transition:opacity .4s; }
  body[data-mood="positive"] .cheek { opacity:.55; }
  body[data-mood="positive"] .eye { transform:scaleY(.3); }
  body[data-mood="positive"] .brow { transform:translateY(-4px); }
  body[data-mood="negative"] .brow.l { transform:rotate(20deg) translateY(2px); }
  body[data-mood="negative"] .brow.r { transform:rotate(-20deg) translateY(2px); }
  body[data-mood="neutral"] .eye { transform:scaleY(.8); }
  @keyframes blink { 0%,92%,100% { transform:scaleY(1); } 96% { transform:scaleY(.1); } }
  body[data-mood="idle"] .eye { animation:blink 4.5s infinite; }
  #answer { margin-top:6px; min-height:104px; }
  #word { font-size:clamp(40px,5.5vw,60px); font-weight:800; letter-spacing:-.02em; line-height:1; margin:0; }
  #note { margin:8px 0 0; font-size:18px; }
  .err { font-weight:600; }
  @media (max-width:820px) {
    main { grid-template-columns:1fr; gap:20px; padding:28px 20px; }
    .stage { order:-1; } svg { width:190px; } #answer { min-height:86px; }
  }
  @media (prefers-reduced-motion:reduce) { * { transition:none !important; animation:none !important; } }
</style></head>
<body data-mood="idle">
<main>
  <section>
    <h1>How does that sentence feel?</h1>
    <p class="lead">Type a sentence and the model will tell you whether it sounds positive, negative or neutral.</p>
    <div class="field">
      <textarea id="t" placeholder="Write a sentence here..." aria-label="Sentence to analyze" autofocus></textarea>
      <div class="bar"><button class="go" id="b">Detect sentiment</button></div>
    </div>
    <div class="try">Not sure what to write? Try one of these.
      <div class="chips">
        <button class="chip">I love this so much!</button>
        <button class="chip">That was a waste of money.</button>
        <button class="chip">The train leaves at 6 pm.</button>
      </div>
    </div>
  </section>
  <section class="stage" aria-live="polite">
    <svg viewBox="0 0 120 120" role="img" aria-label="A face that reacts to the sentence">
      <circle class="face" cx="60" cy="60" r="54"/>
      <circle class="cheek" cx="30" cy="70" r="9"/><circle class="cheek" cx="90" cy="70" r="9"/>
      <line class="line brow l" x1="34" y1="36" x2="52" y2="36"/>
      <line class="line brow r" x1="68" y1="36" x2="86" y2="36"/>
      <ellipse class="feat eye" cx="43" cy="52" rx="6" ry="7"/>
      <ellipse class="feat eye" cx="77" cy="52" rx="6" ry="7"/>
      <path id="mouth" class="line" d="M 40 80 Q 60 84 80 80"/>
    </svg>
    <div id="answer"><p id="word"></p><p id="note"></p></div>
  </section>
</main>
<script>
const t = document.getElementById('t'), b = document.getElementById('b'),
      word = document.getElementById('word'), note = document.getElementById('note'),
      mouth = document.getElementById('mouth');
const MOOD = {
  positive: {cy: 112, ey: 76, label: 'Positive', note: 'That sentence sounds happy.'},
  negative: {cy: 54,  ey: 86, label: 'Negative', note: 'That sentence sounds unhappy.'},
  neutral:  {cy: 80,  ey: 80, label: 'Neutral',  note: 'That sentence is matter-of-fact.'},
  idle:     {cy: 86,  ey: 80}
};
let cur = {cy: 86, ey: 80}, raf;
function drawMouth(m) { mouth.setAttribute('d', `M 40 ${m.ey} Q 60 ${m.cy} 80 ${m.ey}`); }
function tweenMouth(to) {
  cancelAnimationFrame(raf);
  const from = {...cur}, start = performance.now(), dur = 420;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  (function step(now) {
    let p = reduce ? 1 : Math.min((now - start) / dur, 1);
    p = 1 - Math.pow(1 - p, 3);
    cur = {cy: from.cy + (to.cy - from.cy) * p, ey: from.ey + (to.ey - from.ey) * p};
    drawMouth(cur);
    if (p < 1) raf = requestAnimationFrame(step);
  })(start);
}
function show(mood) {
  document.body.dataset.mood = mood;
  tweenMouth(MOOD[mood]);
  word.textContent = MOOD[mood].label || '';
  note.textContent = MOOD[mood].note || '';
}
async function run() {
  const text = t.value.trim();
  if (!text) { word.innerHTML = ''; note.innerHTML = '<span class="err">Type a sentence first, then press the button.</span>'; t.focus(); return; }
  b.disabled = true; b.textContent = 'Reading...';
  try {
    const res = await fetch('/api/classify', {method:'POST', headers:{'Content-Type':'application/json'},
                                              body: JSON.stringify({text})});
    const d = await res.json();
    if (res.ok) show(d.label); else { word.textContent = ''; note.innerHTML = '<span class="err">' + d.error + '</span>'; }
  } catch (e) {
    word.textContent = ''; note.innerHTML = '<span class="err">Could not reach the server. Make sure app.py is still running.</span>';
  }
  b.disabled = false; b.textContent = 'Detect sentiment';
}
b.addEventListener('click', run);
t.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(); } });
document.querySelectorAll('.chip').forEach(c => c.addEventListener('click', () => { t.value = c.textContent; run(); }));
drawMouth(cur);
</script></body></html>"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
