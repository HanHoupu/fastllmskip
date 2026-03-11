#!/usr/bin/env python3
"""
Generate static HTML visualization of diffusion LLM denoising process.

Reads a full_sequence.pkl file (from collect_full_sequence=True),
converts token IDs to readable text via tokenizer, and outputs a
self-contained HTML file that can be opened directly in any browser.

Usage:
    python viz_static.py data/full_sequence.pkl --sample_id 0 -o viz_sample0.html
    python viz_static.py data/full_sequence.pkl --all -o viz_dir/

Requirements:
    pip install transformers numpy
"""

import pickle
import json
import argparse
import os
import logging
from typing import List, Dict, Any, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_pkl(pkl_path: str) -> list:
    """Load full_sequence.pkl → list of samples."""
    log.info("Loading %s ...", pkl_path)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    log.info("Loaded %d sample(s)", len(data))
    return data


def load_tokenizer(model_path: str = "GSAI-ML/LLaDA-8B-Instruct"):
    """Load HuggingFace tokenizer. Returns None on failure."""
    try:
        from transformers import AutoTokenizer
        log.info("Loading tokenizer from %s ...", model_path)
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        log.info("Tokenizer loaded (%d vocab)", tok.vocab_size)
        return tok
    except Exception as exc:
        log.warning("Could not load tokenizer (%s). Will use token-ID mode.", exc)
        return None


def _decode(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"<{token_id}>"


def process_sample(
    sample_blocks: list,
    tokenizer,
    batch_idx: int = 0,
    question: Optional[str] = None,
    gen_answer: Optional[str] = None,
    ref_answer: Optional[str] = None,
) -> dict:
    """
    Convert one sample (list of block dicts) to a JSON-serializable structure.

    Each block dict has:
        'block_id': int,
        'steps': [ { 'step', 'x', 'x0', 'mask', 'confidence', 'current_block_range' }, ... ]

    Returns a dict ready for JSON embedding into the HTML template.
    """
    first_step = sample_blocks[0]["steps"][0]
    gen_start = first_step["current_block_range"][0]

    prompt_ids = first_step["x"][batch_idx, :gen_start]
    prompt_text = (
        tokenizer.decode(prompt_ids, skip_special_tokens=False)
        if tokenizer
        else " ".join(str(int(t)) for t in prompt_ids)
    )

    block_ranges = [
        list(b["steps"][0]["current_block_range"]) for b in sample_blocks
    ]

    all_steps: List[dict] = []
    for block in sample_blocks:
        default_br = list(block["steps"][0]["current_block_range"])
        for sd in block["steps"]:
            # Keep per-step window range so expanded windows are rendered correctly.
            br = list(sd.get("current_block_range", default_br))
            x = sd["x"][batch_idx]
            x0 = sd.get("x0")
            if x0 is not None:
                x0 = x0[batch_idx] if x0.ndim > 1 else x0
            mask = sd["mask"][batch_idx]
            conf = sd["confidence"][batch_idx]

            tokens: List[dict] = []
            for pos in range(gen_start, len(x)):
                tid = int(x[pos])
                is_m = bool(mask[pos])
                c = float(conf[pos])
                pid = int(x0[pos]) if x0 is not None else None

                display_id = (int(x0[pos]) if is_m and x0 is not None else tid)
                text = _decode(tokenizer, display_id)
                ptxt = (
                    _decode(tokenizer, pid)
                    if pid is not None and pid != tid
                    else None
                )

                tokens.append({
                    "p": pos,
                    "t": text,
                    "m": is_m,
                    "c": round(c, 5),
                    "d": tid,
                    "x": pid,
                    "xt": ptxt,
                })

            all_steps.append({"s": sd["step"], "br": br, "tk": tokens})

    result = {
        "prompt": prompt_text,
        "num_blocks": len(sample_blocks),
        "block_ranges": block_ranges,
        "gen_start": gen_start,
        "steps": all_steps,
    }
    if question is not None:
        result["question"] = question
    if gen_answer is not None:
        result["gen_answer"] = str(gen_answer)
    if ref_answer is not None:
        result["ref_answer"] = str(ref_answer)
    return result


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#eee;display:flex;min-height:100vh}
.main{flex:1;margin-right:280px;padding:20px;overflow-y:auto}
.sidebar{position:fixed;right:0;top:0;width:270px;height:100vh;background:#16213e;padding:18px;overflow-y:auto;box-shadow:-2px 0 12px rgba(0,0,0,.4)}
h1{font-size:22px;margin-bottom:6px}
.meta{color:#8892b0;font-size:13px;margin-bottom:18px}
.section{margin-bottom:22px}
.section-title{font-size:15px;font-weight:600;color:#ccd6f6;margin-bottom:8px;border-bottom:1px solid #233554;padding-bottom:4px}
.prompt-box{background:#0a192f;padding:14px;border-radius:6px;border-left:3px solid #64ffda;font-family:'Courier New',monospace;font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;color:#a8b2d1}
.tokens-wrap{background:#0a192f;padding:14px;border-radius:6px;line-height:2.1;font-family:'Courier New',monospace;font-size:14px;word-wrap:break-word;white-space:pre-wrap}
.output-box{background:#0a192f;padding:14px;border-radius:6px;font-family:'Courier New',monospace;font-size:13px;white-space:pre-wrap;word-break:break-all;max-height:280px;overflow-y:auto;color:#a8b2d1;line-height:1.6}
.tk{display:inline-block;padding:3px 5px;margin:2px;border-radius:4px;cursor:default;position:relative;transition:transform .15s}
.tk:hover{transform:translateY(-2px);box-shadow:0 4px 10px rgba(0,0,0,.4);z-index:99}
.tk.masked{border:2px dashed #546e7a}
.tk.decoded{background:#233554!important;color:#8892b0}
.tk.forced{background:#4b5563!important;color:#fff;border:2px solid #374151;font-weight:600}
.tk.regret-s{background:#d97706!important;color:#451a03;border:2px solid #b45309}
.tk.regret-d{background:#c026d3!important;color:#4a044e;border:2px solid #a21caf}
.tk.saber-f4{background:#0ea5e9!important;color:#082f49;border:2px solid #0284c7;font-weight:700}
.tk.berm-rm{border:2px solid #f43f5e!important;box-shadow:0 0 0 1px #f43f5e inset}
.tk.saber-f4{background:#0ea5e9!important;color:#082f49;border:2px solid #0284c7;font-weight:700}
.tk.berm-rm{border:2px solid #f43f5e!important;box-shadow:0 0 0 1px #f43f5e inset}
.tip{visibility:hidden;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:rgba(0,0,0,.92);color:#fff;padding:6px 10px;border-radius:5px;font-size:12px;white-space:nowrap;z-index:999;pointer-events:none}
.tk:hover .tip{visibility:visible}
.blk-sep{display:inline-block;width:3px;height:26px;background:linear-gradient(#64ffda,#7c3aed);margin:0 8px;vertical-align:middle;border-radius:2px}
.blk-label{display:inline-block;background:linear-gradient(135deg,#7c3aed,#4f46e5);color:#fff;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600;margin:3px 6px}
/* sidebar controls */
.step-info{text-align:center;font-size:15px;font-weight:600;padding:8px;background:#0a192f;border-radius:6px;margin-bottom:12px}
.slider{width:100%;margin:8px 0;-webkit-appearance:none;height:6px;border-radius:3px;background:#233554;outline:none}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#64ffda;cursor:pointer}
.btns{display:flex;gap:6px;margin-bottom:10px}
.btns button{flex:1;padding:8px;border:none;border-radius:5px;font-size:14px;cursor:pointer;font-weight:500;transition:background .2s}
.btn-prev,.btn-next{background:#233554;color:#ccd6f6}
.btn-prev:hover,.btn-next:hover{background:#2d4a6f}
.btn-play{background:#064e3b;color:#6ee7b7}
.btn-play.on{background:#7f1d1d;color:#fca5a5}
.btn-speed{background:#233554;color:#e2e8f0}
.legend{margin-top:16px;font-size:12px;color:#8892b0}
.legend-title{font-weight:600;color:#ccd6f6;margin-bottom:6px}
.legend-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.legend-swatch{width:32px;height:16px;border-radius:3px;flex-shrink:0}
</style>
</head>
<body>
<div class="main">
  <h1>Denoising Visualization</h1>
  <p class="meta" id="metaInfo"></p>
  <div class="section"><div class="section-title">Prompt</div><div class="prompt-box" id="promptBox"></div></div>
  <div class="section"><div class="section-title">Token Visualization</div><div class="tokens-wrap" id="tokensDisplay"></div></div>
  <div class="section"><div class="section-title">Text Output</div><div class="output-box" id="outputBox"></div></div>
</div>
<div class="sidebar">
  <div class="step-info" id="stepInfo">Loading...</div>
  <input type="range" class="slider" id="slider" min="0" max="0" value="0">
  <div class="btns">
    <button class="btn-prev" id="prevBtn">&#9664; Prev</button>
    <button class="btn-play" id="playBtn">&#9654; Play</button>
    <button class="btn-next" id="nextBtn">Next &#9654;</button>
  </div>
  <div class="btns">
    <button class="btn-speed" id="speedBtn">Speed: 1x</button>
  </div>
  <div class="legend">
    <div class="legend-title">Legend</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#233554"></div>Decoded (normal)</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#4b5563;border:2px solid #374151"></div>Forced decode (&lt;0.9)</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#0ea5e9;border:2px solid #0284c7"></div>Saber floor forced (n)</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#0ea5e9;border:2px solid #0284c7"></div>Saber floor forced (n)</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#d97706;border:2px solid #b45309"></div>Confidence dropped</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#c026d3;border:2px solid #a21caf"></div>Model changed mind</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#334155;border:2px solid #f43f5e"></div>BERM remask</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#334155;border:2px solid #f43f5e"></div>BERM remask</div>
    <div class="legend-row"><div class="legend-swatch" style="background:linear-gradient(90deg,#dc2626,#eab308,#22c55e)"></div>Masked (by confidence)</div>
  </div>
</div>
<script>
const D=__DATA_JSON__;
const BR=D.block_ranges;
document.getElementById('promptBox').textContent=D.prompt;
document.getElementById('metaInfo').textContent=`${D.num_blocks} blocks | ${D.steps.length} total steps | gen_start=${D.gen_start}`;

const slider=document.getElementById('slider');
slider.max=D.steps.length-1;
let cur=0,playing=false,timer=null;
const speeds=[0.5,1,2,4,8];let si=1;

function blockOf(p){for(let i=0;i<BR.length;i++){if(p>=BR[i][0]&&p<BR[i][1])return i;}return BR.length-1;}

function confColor(c){
  let r,g,b;
  if(c<0.5){const t=c/0.5;r=220+(234-220)*t;g=38+(179-38)*t;b=38+(8-38)*t;}
  else{const t=(c-0.5)/0.5;r=234+(34-234)*t;g=179+(197-179)*t;b=8+(94-8)*t;}
  return`rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

function render(idx){
  cur=idx;slider.value=idx;
  const st=D.steps[idx];
  const [bs,be]=st.br;
  document.getElementById('stepInfo').textContent=`Block ${blockOf(bs)} | Step ${idx+1}/${D.steps.length}`;
  const prev=idx>0?D.steps[idx-1]:null;
  const pm=new Map();if(prev)prev.tk.forEach(t=>pm.set(t.p,t));
  const container=document.getElementById('tokensDisplay');
  const outBox=document.getElementById('outputBox');
  container.innerHTML='';let out='';
  let lastBlk=-1;
  const THRESH=0.9;
  st.tk.forEach(tk=>{
    const blk=blockOf(tk.p);
    if(blk!==lastBlk){
      if(lastBlk!==-1){const sep=document.createElement('span');sep.className='blk-sep';container.appendChild(sep);out+=' | ';}
      const lb=document.createElement('span');lb.className='blk-label';lb.textContent='B'+blk;container.appendChild(lb);
      lastBlk=blk;
    }
    const sp=document.createElement('span');sp.className='tk';
    const decoded=!tk.m;
    const pt=pm.get(tk.p);
    const justDecoded=pt&&pt.m&&decoded;
    const inCur=tk.p>=bs&&tk.p<be;
    let cls='',status='';
    if(decoded&&inCur){
      if(justDecoded&&tk.f4){cls='saber-f4';status='Saber floor forced';}
      else if(justDecoded&&tk.c<THRESH){cls='forced';status='Forced';}
      else if(!justDecoded&&tk.x!==null){
        if(tk.x!==tk.d){cls='regret-d';status='Regret (diff)';}
        else if(tk.c<THRESH){cls='regret-s';status='Regret (conf)';}
      }
    }
    const justRemasked = pt && !pt.m && tk.m;
    if(cls){sp.classList.add(cls);}
    else if(decoded){sp.classList.add('decoded');status='Decoded';out+=tk.t;}
    else{
      sp.classList.add('masked');sp.style.background=confColor(tk.c);status='Masked';out+=' ';
      if(justRemasked||tk.brm){sp.classList.add('berm-rm');status='BERM remask';}
    }
    if(cls)out+=tk.t;
    sp.textContent=tk.m?`[${tk.t}]`:tk.t;
    const tip=document.createElement('span');tip.className='tip';
    let tipTxt=`Pos:${tk.p} | Conf:${tk.c.toFixed(4)} | ${status}`;
    if(cls==='regret-d'&&tk.xt)tipTxt+=` → wants "${tk.xt}"`;
    tip.textContent=tipTxt;sp.appendChild(tip);
    container.appendChild(sp);
  });
  outBox.textContent=out;
}

slider.addEventListener('input',e=>render(+e.target.value));
document.getElementById('prevBtn').addEventListener('click',()=>{if(cur>0)render(cur-1);});
document.getElementById('nextBtn').addEventListener('click',()=>{if(cur<D.steps.length-1)render(cur+1);});
document.getElementById('playBtn').addEventListener('click',()=>{
  playing=!playing;const b=document.getElementById('playBtn');
  if(playing){b.textContent='⏸ Pause';b.classList.add('on');play();}
  else{b.textContent='▶ Play';b.classList.remove('on');clearInterval(timer);}
});
function play(){timer=setInterval(()=>{if(cur<D.steps.length-1)render(cur+1);else{playing=false;document.getElementById('playBtn').textContent='▶ Play';document.getElementById('playBtn').classList.remove('on');clearInterval(timer);}},500/speeds[si]);}
document.getElementById('speedBtn').addEventListener('click',()=>{si=(si+1)%speeds.length;document.getElementById('speedBtn').textContent='Speed: '+speeds[si]+'x';if(playing){clearInterval(timer);play();}});
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&cur>0)render(cur-1);if(e.key==='ArrowRight'&&cur<D.steps.length-1)render(cur+1);if(e.key===' '){e.preventDefault();document.getElementById('playBtn').click();}});
render(0);
</script>
</body>
</html>"""


def generate_html(sample_json: dict, title: str = "Denoising Visualization") -> str:
    """Embed processed sample data into the HTML template."""
    data_str = json.dumps(sample_json, ensure_ascii=False, separators=(",", ":"))
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", data_str).replace("__TITLE__", title)
    return html


# ---------------------------------------------------------------------------
# Multi-sample HTML (all samples in one file, with sample navigation)
# ---------------------------------------------------------------------------

_MULTI_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#1a1a2e;color:#eee;display:flex;min-height:100vh}
.main{flex:1;margin-right:280px;padding:20px;overflow-y:auto}
.sidebar{position:fixed;right:0;top:0;width:270px;height:100vh;background:#16213e;padding:18px;overflow-y:auto;box-shadow:-2px 0 12px rgba(0,0,0,.4)}
h1{font-size:22px;margin-bottom:6px}
.meta{color:#8892b0;font-size:13px;margin-bottom:18px}
.section{margin-bottom:22px}
.section-title{font-size:15px;font-weight:600;color:#ccd6f6;margin-bottom:8px;border-bottom:1px solid #233554;padding-bottom:4px}
.prompt-box{background:#0a192f;padding:14px;border-radius:6px;border-left:3px solid #64ffda;font-family:'Courier New',monospace;font-size:13px;line-height:1.6;white-space:pre-wrap;word-break:break-all;max-height:200px;overflow-y:auto;color:#a8b2d1}
.tokens-wrap{background:#0a192f;padding:14px;border-radius:6px;line-height:2.1;font-family:'Courier New',monospace;font-size:14px;word-wrap:break-word;white-space:pre-wrap}
.output-box{background:#0a192f;padding:14px;border-radius:6px;font-family:'Courier New',monospace;font-size:13px;white-space:pre-wrap;word-break:break-all;max-height:280px;overflow-y:auto;color:#a8b2d1;line-height:1.6}
.tk{display:inline-block;padding:3px 5px;margin:2px;border-radius:4px;cursor:default;position:relative;transition:transform .15s}
.tk:hover{transform:translateY(-2px);box-shadow:0 4px 10px rgba(0,0,0,.4);z-index:99}
.tk.masked{border:2px dashed #546e7a}
.tk.decoded{background:#233554!important;color:#8892b0}
.tk.forced{background:#4b5563!important;color:#fff;border:2px solid #374151;font-weight:600}
.tk.regret-s{background:#d97706!important;color:#451a03;border:2px solid #b45309}
.tk.regret-d{background:#c026d3!important;color:#4a044e;border:2px solid #a21caf}
.tip{visibility:hidden;position:absolute;bottom:calc(100% + 6px);left:50%;transform:translateX(-50%);background:rgba(0,0,0,.92);color:#fff;padding:6px 10px;border-radius:5px;font-size:12px;white-space:nowrap;z-index:999;pointer-events:none}
.tk:hover .tip{visibility:visible}
.blk-sep{display:inline-block;width:3px;height:26px;background:linear-gradient(#64ffda,#7c3aed);margin:0 8px;vertical-align:middle;border-radius:2px}
.blk-label{display:inline-block;background:linear-gradient(135deg,#7c3aed,#4f46e5);color:#fff;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600;margin:3px 6px}
/* sidebar controls */
.step-info{text-align:center;font-size:15px;font-weight:600;padding:8px;background:#0a192f;border-radius:6px;margin-bottom:12px}
.slider{width:100%;margin:8px 0;-webkit-appearance:none;height:6px;border-radius:3px;background:#233554;outline:none}
.slider::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#64ffda;cursor:pointer}
.btns{display:flex;gap:6px;margin-bottom:10px}
.btns button{flex:1;padding:8px;border:none;border-radius:5px;font-size:14px;cursor:pointer;font-weight:500;transition:background .2s}
.btn-prev,.btn-next{background:#233554;color:#ccd6f6}
.btn-prev:hover,.btn-next:hover{background:#2d4a6f}
.btn-play{background:#064e3b;color:#6ee7b7}
.btn-play.on{background:#7f1d1d;color:#fca5a5}
.btn-speed{background:#233554;color:#e2e8f0}
.sample-nav{margin-bottom:16px;padding:10px;background:#0a192f;border-radius:6px}
.sample-nav select{width:100%;padding:6px;border-radius:4px;border:1px solid #233554;background:#16213e;color:#ccd6f6;font-size:13px}
.sample-nav .nav-btns{display:flex;gap:6px;margin-top:8px}
.sample-nav .nav-btns button{flex:1;padding:6px;border:none;border-radius:4px;font-size:13px;cursor:pointer;background:#233554;color:#ccd6f6}
.sample-nav .nav-btns button:hover{background:#2d4a6f}
.answer-box{background:#0a192f;padding:10px;border-radius:6px;font-size:13px;margin-bottom:12px;border-left:3px solid #7c3aed;color:#a8b2d1}
.answer-box .correct{color:#22c55e;font-weight:600}
.answer-box .wrong{color:#ef4444;font-weight:600}
.legend{margin-top:16px;font-size:12px;color:#8892b0}
.legend-title{font-weight:600;color:#ccd6f6;margin-bottom:6px}
.legend-row{display:flex;align-items:center;gap:8px;margin-bottom:5px}
.legend-swatch{width:32px;height:16px;border-radius:3px;flex-shrink:0}
</style>
</head>
<body>
<div class="main">
  <h1>__TITLE__</h1>
  <p class="meta" id="metaInfo"></p>
  <div class="section" id="questionSection" style="display:none"><div class="section-title">Question</div><div class="prompt-box" id="questionBox"></div></div>
  <div class="section"><div class="section-title">Prompt</div><div class="prompt-box" id="promptBox"></div></div>
  <div class="section"><div class="section-title">Token Visualization</div><div class="tokens-wrap" id="tokensDisplay"></div></div>
  <div class="section"><div class="section-title">Text Output</div><div class="output-box" id="outputBox"></div></div>
</div>
<div class="sidebar">
  <div class="sample-nav">
    <div style="font-weight:600;margin-bottom:6px;color:#ccd6f6">Sample</div>
    <select id="sampleSelect"></select>
    <div class="nav-btns">
      <button id="prevSample">&#9664; Prev Sample</button>
      <button id="nextSample">Next Sample &#9654;</button>
    </div>
  </div>
  <div class="answer-box" id="answerBox" style="display:none"></div>
  <div class="step-info" id="stepInfo">Loading...</div>
  <input type="range" class="slider" id="slider" min="0" max="0" value="0">
  <div class="btns">
    <button class="btn-prev" id="prevBtn">&#9664; Prev</button>
    <button class="btn-play" id="playBtn">&#9654; Play</button>
    <button class="btn-next" id="nextBtn">Next &#9654;</button>
  </div>
  <div class="btns">
    <button class="btn-speed" id="speedBtn">Speed: 1x</button>
  </div>
  <div class="legend">
    <div class="legend-title">Legend</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#233554"></div>Decoded (normal)</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#4b5563;border:2px solid #374151"></div>Forced decode (&lt;0.9)</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#d97706;border:2px solid #b45309"></div>Confidence dropped</div>
    <div class="legend-row"><div class="legend-swatch" style="background:#c026d3;border:2px solid #a21caf"></div>Model changed mind</div>
    <div class="legend-row"><div class="legend-swatch" style="background:linear-gradient(90deg,#dc2626,#eab308,#22c55e)"></div>Masked (by confidence)</div>
  </div>
</div>
<script>
const SAMPLES=__SAMPLES_JSON__;
let sIdx=0;
const sel=document.getElementById('sampleSelect');
SAMPLES.forEach((s,i)=>{const o=document.createElement('option');o.value=i;const tag=s.gen_answer?(s.gen_answer.includes('\u2713')?'\u2713':'\u2717'):'';o.textContent=`#${i} ${tag} (${s.steps.length} steps)`;sel.appendChild(o);});

function loadSample(si){
  sIdx=si;sel.value=si;
  const D=SAMPLES[si];
  window._D=D;window._BR=D.block_ranges;
  document.getElementById('promptBox').textContent=D.prompt;
  document.getElementById('metaInfo').textContent=`Sample ${si+1}/${SAMPLES.length} | ${D.num_blocks} blocks | ${D.steps.length} steps | gen_start=${D.gen_start}`;
  const qSec=document.getElementById('questionSection');
  if(D.question){qSec.style.display='';document.getElementById('questionBox').textContent=D.question;}else{qSec.style.display='none';}
  const aBox=document.getElementById('answerBox');
  if(D.gen_answer){aBox.style.display='';const isC=D.gen_answer.includes('\u2713');aBox.innerHTML=`<span class="${isC?'correct':'wrong'}">Answer: ${D.gen_answer}</span>`+(D.ref_answer?`<br>Ref: ${D.ref_answer}`:'');}else{aBox.style.display='none';}
  const slider=document.getElementById('slider');
  slider.max=D.steps.length-1;slider.value=0;
  render(0);
}

function blockOf(p){const BR=window._BR;for(let i=0;i<BR.length;i++){if(p>=BR[i][0]&&p<BR[i][1])return i;}return BR.length-1;}
function confColor(c){let r,g,b;if(c<0.5){const t=c/0.5;r=220+(234-220)*t;g=38+(179-38)*t;b=38+(8-38)*t;}else{const t=(c-0.5)/0.5;r=234+(34-234)*t;g=179+(197-179)*t;b=8+(94-8)*t;}return`rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;}

let cur=0,playing=false,timer=null;
const speeds=[0.5,1,2,4,8];let si=1;

function render(idx){
  const D=window._D;cur=idx;document.getElementById('slider').value=idx;
  const st=D.steps[idx];const[bs,be]=st.br;
  document.getElementById('stepInfo').textContent=`Block ${blockOf(bs)} | Step ${idx+1}/${D.steps.length}`;
  const prev=idx>0?D.steps[idx-1]:null;
  const pm=new Map();if(prev)prev.tk.forEach(t=>pm.set(t.p,t));
  const container=document.getElementById('tokensDisplay');
  const outBox=document.getElementById('outputBox');
  container.innerHTML='';let out='';let lastBlk=-1;const THRESH=0.9;
  st.tk.forEach(tk=>{
    const blk=blockOf(tk.p);
    if(blk!==lastBlk){if(lastBlk!==-1){const sep=document.createElement('span');sep.className='blk-sep';container.appendChild(sep);out+=' | ';}const lb=document.createElement('span');lb.className='blk-label';lb.textContent='B'+blk;container.appendChild(lb);lastBlk=blk;}
    const sp=document.createElement('span');sp.className='tk';
    const decoded=!tk.m;const pt=pm.get(tk.p);const justDecoded=pt&&pt.m&&decoded;const inCur=tk.p>=bs&&tk.p<be;
    let cls='',status='';
    if(decoded&&inCur){if(justDecoded&&tk.f4){cls='saber-f4';status='Saber floor forced';}else if(justDecoded&&tk.c<THRESH){cls='forced';status='Forced';}else if(!justDecoded&&tk.x!==null){if(tk.x!==tk.d){cls='regret-d';status='Regret (diff)';}else if(tk.c<THRESH){cls='regret-s';status='Regret (conf)';}}}
    const justRemasked = pt && !pt.m && tk.m;
    if(cls){sp.classList.add(cls);}else if(decoded){sp.classList.add('decoded');status='Decoded';out+=tk.t;}else{sp.classList.add('masked');sp.style.background=confColor(tk.c);status='Masked';out+=' ';if(justRemasked||tk.brm){sp.classList.add('berm-rm');status='BERM remask';}}
    if(cls)out+=tk.t;
    sp.textContent=tk.m?`[${tk.t}]`:tk.t;
    const tip=document.createElement('span');tip.className='tip';
    let tipTxt=`Pos:${tk.p} | Conf:${tk.c.toFixed(4)} | ${status}`;
    if(cls==='regret-d'&&tk.xt)tipTxt+=` \u2192 wants "${tk.xt}"`;
    tip.textContent=tipTxt;sp.appendChild(tip);container.appendChild(sp);
  });
  outBox.textContent=out;
}

sel.addEventListener('change',e=>loadSample(+e.target.value));
document.getElementById('prevSample').addEventListener('click',()=>{if(sIdx>0)loadSample(sIdx-1);});
document.getElementById('nextSample').addEventListener('click',()=>{if(sIdx<SAMPLES.length-1)loadSample(sIdx+1);});
document.getElementById('slider').addEventListener('input',e=>render(+e.target.value));
document.getElementById('prevBtn').addEventListener('click',()=>{if(cur>0)render(cur-1);});
document.getElementById('nextBtn').addEventListener('click',()=>{if(cur<window._D.steps.length-1)render(cur+1);});
document.getElementById('playBtn').addEventListener('click',()=>{playing=!playing;const b=document.getElementById('playBtn');if(playing){b.textContent='\u23f8 Pause';b.classList.add('on');play();}else{b.textContent='\u25b6 Play';b.classList.remove('on');clearInterval(timer);}});
function play(){timer=setInterval(()=>{if(cur<window._D.steps.length-1)render(cur+1);else{playing=false;document.getElementById('playBtn').textContent='\u25b6 Play';document.getElementById('playBtn').classList.remove('on');clearInterval(timer);}},500/speeds[si]);}
document.getElementById('speedBtn').addEventListener('click',()=>{si=(si+1)%speeds.length;document.getElementById('speedBtn').textContent='Speed: '+speeds[si]+'x';if(playing){clearInterval(timer);play();}});
document.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'&&cur>0)render(cur-1);if(e.key==='ArrowRight'&&cur<window._D.steps.length-1)render(cur+1);if(e.key===' '){e.preventDefault();document.getElementById('playBtn').click();}});
loadSample(0);
</script>
</body>
</html>"""


def generate_multi_html(
    samples_json: List[dict],
    title: str = "Denoising Visualization",
) -> str:
    """
    Generate a single HTML file containing all samples with a sample navigator.

    Args:
        samples_json: List of dicts from process_sample().
        title: Page title.

    Returns:
        HTML string.
    """
    data_str = json.dumps(samples_json, ensure_ascii=False, separators=(",", ":"))
    html = _MULTI_HTML_TEMPLATE.replace("__SAMPLES_JSON__", data_str).replace("__TITLE__", title)
    return html


def pkl_to_html(
    pkl_path: str,
    output_path: str,
    sample_id: int = 0,
    tokenizer=None,
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct",
    batch_idx: int = 0,
) -> str:
    """
    End-to-end: pkl file → static HTML file.

    Args:
        pkl_path: Path to full_sequence.pkl
        output_path: Where to write the .html file
        sample_id: Which sample to visualize
        tokenizer: Pre-loaded tokenizer, or None to auto-load
        model_path: HF model path for tokenizer (used only if tokenizer is None)
        batch_idx: Which batch element (usually 0)

    Returns:
        The output file path.
    """
    data = load_pkl(pkl_path)

    if sample_id < 0 or sample_id >= len(data):
        raise ValueError(f"sample_id={sample_id} out of range [0, {len(data)})")

    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)

    log.info("Processing sample %d ...", sample_id)
    sample_json = process_sample(data[sample_id], tokenizer, batch_idx)

    title = f"Sample {sample_id} — {os.path.basename(pkl_path)}"
    html = generate_html(sample_json, title)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    log.info("Written %s (%.1f MB)", output_path, size_mb)
    return output_path


# ---------------------------------------------------------------------------
# Generation with collection (self-contained, no dependency on llada/generate.py)
# ---------------------------------------------------------------------------

def _lazy_torch():
    import torch
    import torch.nn.functional as F
    return torch, F


def generate_with_collection(
    model,
    prompt,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float = 0.9,
):
    """
    Run diffusion LLM generation while collecting full sequence snapshots.

    This is a self-contained function that mirrors the logic of
    generate_with_dual_cache but adds data collection at every step.
    Output format is identical to collect_full_sequence=True in the fork.

    Args:
        model:        The LLaDA model (already on device, in eval mode).
        prompt:       Token IDs tensor of shape (1, L).
        steps:        Total sampling steps across all blocks.
        gen_length:   Number of tokens to generate.
        block_length: Block size for semi-autoregressive decoding.
        temperature:  Gumbel noise temperature (0 = greedy).
        remasking:    'low_confidence' or 'random'.
        mask_id:      Token ID for [MASK] (default 126336 for LLaDA).
        threshold:    Confidence threshold for parallel decoding.

    Returns:
        x:            Final token tensor (1, L + gen_length).
        nfe:          Number of forward evaluations.
        history:      list[dict] — one per block, each with 'block_id' and 'steps'.
                      Each step has keys: step, x, x0, mask, confidence, current_block_range.
    """
    torch, F = _lazy_torch()

    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0
    history = []
    # Persistent confidence tracks values across blocks for visualization
    pconf = torch.zeros(B, Lp + gen_length, dtype=torch.float64, device=model.device)

    def _gumbel(logits):
        if temperature == 0:
            return logits
        lg = logits.to(torch.float64)
        noise = torch.rand_like(lg, dtype=torch.float64)
        return lg.exp() / ((-torch.log(noise)) ** temperature)

    def _compute_x0_conf(logits, mask):
        """Compute x0 proposals and their confidence from logits."""
        x0 = torch.argmax(_gumbel(logits), dim=-1)
        if remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float64), dim=-1)
            conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        else:
            conf = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
        return x0, conf

    def _transfer(logits, mask, x_cur, quota):
        """Select which masked tokens to unmask (threshold or top-k)."""
        x0, conf = _compute_x0_conf(logits, mask)
        x0 = torch.where(mask, x0, x_cur)
        neg = torch.tensor(torch.finfo(conf.dtype).min, device=conf.device, dtype=conf.dtype)
        cval = torch.where(mask, conf, neg)

        if threshold is not None:
            tidx = mask & (cval >= threshold)
            # Force at least one token per batch
            mx = torch.argmax(cval, dim=1, keepdim=True)
            force = torch.zeros_like(tidx).scatter_(1, mx, True) & mask
            tidx = tidx | force
        else:
            # Top-k by quota
            _, idx = torch.sort(cval, dim=1, descending=True)
            cols = torch.arange(cval.shape[1], device=cval.device).unsqueeze(0)
            tidx = (cols < quota.unsqueeze(1))
            scatter_t = torch.zeros_like(tidx, dtype=torch.int8)
            scatter_t = scatter_t.scatter(1, idx, tidx.to(torch.int8)).bool() & mask

            tidx = scatter_t

        return x0, conf, tidx

    def _snap(step_i, nb, s, e, x_now, x0_full, conf_full):
        """Take a snapshot of current state."""
        mask_now = (x_now == mask_id)
        pconf[:, s:e] = conf_full[:, s:e] if conf_full.shape[1] > (e - s) else conf_full
        return {
            "step": step_i,
            "block_id": nb,
            "x": x_now.cpu().numpy().copy(),
            "x0": x0_full.cpu().numpy().copy(),
            "mask": mask_now.cpu().numpy().copy(),
            "confidence": pconf.cpu().numpy().copy(),
            "current_block_range": (s, e),
        }

    def _get_num_transfer_tokens(block_mask, n_steps):
        total = block_mask.sum(dim=1)
        base = torch.div(total, n_steps, rounding_mode="floor")
        rem = total - base * n_steps
        ntt = base.unsqueeze(1).expand(-1, n_steps).clone()
        cols = torch.arange(n_steps, device=block_mask.device).unsqueeze(0)
        ntt = ntt + (cols < rem.unsqueeze(1)).long()
        return ntt

    with torch.no_grad():
        for nb in range(num_blocks):
            s = Lp + nb * block_length
            e = s + block_length
            block_steps = []

            bmask = (x[:, s:e] == mask_id)
            ntt = _get_num_transfer_tokens(bmask, steps_per_block)

            # --- Step 0: full-prefix forward ---
            out = model(x, use_cache=True)
            past_kv = out.past_key_values
            nfe += 1

            replace_pos = torch.zeros_like(x, dtype=torch.bool)
            replace_pos[:, s:e] = True

            gmask = (x == mask_id)
            gmask[:, e:] = False

            quota0 = None if threshold is not None else ntt[:, 0]
            x0_full, conf_full, tidx = _transfer(out.logits, gmask, x, quota0)
            x = torch.where(tidx, x0_full, x)

            block_steps.append(_snap(0, nb, s, e, x, x0_full, conf_full))

            # --- Steps 1 .. N ---
            for si in range(1, steps_per_block):
                if (x[:, s:e] == mask_id).sum() == 0:
                    break

                logits_blk = model(
                    x[:, s:e],
                    past_key_values=past_kv,
                    use_cache=True,
                    replace_position=replace_pos,
                ).logits

                mask_blk = (x[:, s:e] == mask_id)
                quota_i = None if threshold is not None else ntt[:, si]
                x0_blk, conf_blk, tidx_blk = _transfer(logits_blk, mask_blk, x[:, s:e], quota_i)

                blk_new = torch.where(tidx_blk, x0_blk, x[:, s:e])
                x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

                # Build full-sequence x0 snapshot
                x0_snap = x.clone()
                x0_snap[:, s:e] = torch.where(mask_blk, x0_blk, x[:, s:e])
                # Update persistent confidence for the block
                pconf[:, s:e] = torch.where(mask_blk, conf_blk, pconf[:, s:e])

                block_steps.append(_snap(si, nb, s, e, x, x0_snap, pconf))
                nfe += 1

            history.append({"block_id": nb, "steps": block_steps})

    return x, nfe, history


def generate_and_visualize(
    model,
    tokenizer,
    prompt_text: str,
    output: str = "viz.html",
    gen_length: int = 128,
    steps: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    threshold: float = 0.9,
    mask_id: int = 126336,
    open_browser: bool = True,
) -> str:
    """
    End-to-end: text prompt → model generation → static HTML visualization.

    Args:
        model:        Loaded LLaDA model (on GPU, eval mode).
        tokenizer:    Loaded HF tokenizer.
        prompt_text:  The user's prompt string.
        output:       Output HTML file path.
        gen_length:   Number of tokens to generate.
        steps:        Total denoising steps.
        block_length: Block size.
        temperature:  Sampling temperature.
        threshold:    Confidence threshold for parallel decoding.
        mask_id:      Mask token ID.
        open_browser: Whether to auto-open the HTML in browser.

    Returns:
        Path to the generated HTML file.
    """
    torch, _ = _lazy_torch()

    # Tokenize prompt
    messages = [{"role": "user", "content": prompt_text}]
    chat_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(chat_input)["input_ids"]
    input_ids = torch.tensor(input_ids, device=model.device).unsqueeze(0)

    log.info("Prompt length: %d tokens, generating %d tokens ...", input_ids.shape[1], gen_length)

    # Generate with collection
    x, nfe, history = generate_with_collection(
        model, input_ids,
        steps=steps, gen_length=gen_length, block_length=block_length,
        temperature=temperature, threshold=threshold, mask_id=mask_id,
    )

    # Decode final text
    final_text = tokenizer.decode(x[0, input_ids.shape[1]:], skip_special_tokens=True)
    log.info("Generated text: %s", final_text[:200])
    log.info("NFE: %d", nfe)

    # Convert to HTML
    sample_json = process_sample(history, tokenizer, batch_idx=0)
    html = generate_html(sample_json, f"Visualization — {prompt_text[:60]}")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML saved to %s", os.path.abspath(output))

    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(output)}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate static HTML visualization from full_sequence.pkl"
    )
    parser.add_argument("pkl_file", help="Path to full_sequence.pkl")
    parser.add_argument("--sample_id", "-s", type=int, default=0, help="Sample index (default: 0)")
    parser.add_argument("--all", action="store_true", help="Export all samples (output must be a directory)")
    parser.add_argument("--output", "-o", default=None, help="Output .html path or directory (with --all)")
    parser.add_argument("--model_path", default="GSAI-ML/LLaDA-8B-Instruct", help="Tokenizer model path")
    parser.add_argument("--no_tokenizer", action="store_true", help="Skip tokenizer, show token IDs only")
    parser.add_argument("--batch_idx", type=int, default=0, help="Batch element index (default: 0)")
    args = parser.parse_args()

    tokenizer = None if args.no_tokenizer else load_tokenizer(args.model_path)

    if args.all:
        data = load_pkl(args.pkl_file)
        out_dir = args.output or "viz_output"
        os.makedirs(out_dir, exist_ok=True)
        for sid in range(len(data)):
            out_path = os.path.join(out_dir, f"sample_{sid}.html")
            sample_json = process_sample(data[sid], tokenizer, args.batch_idx)
            title = f"Sample {sid} — {os.path.basename(args.pkl_file)}"
            html = generate_html(sample_json, title)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            log.info("Written %s", out_path)
        log.info("Done! %d files in %s", len(data), out_dir)
    else:
        out_path = args.output or f"viz_sample_{args.sample_id}.html"
        pkl_to_html(args.pkl_file, out_path, args.sample_id, tokenizer, args.model_path, args.batch_idx)


if __name__ == "__main__":
    main()
