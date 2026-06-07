# Confidence Telemetry HTML Series — Handoff Document

A standalone reference for continuing work on the confidence-telemetry HTML cluster pages. Hand this entire document to a new chat at the start of the conversation. It contains everything needed to add new data, build new pages, or update existing ones without re-establishing context.

---

## 1. What this project is

A series of standalone HTML pages, one per **(model × benchmark)** cell, that visualize confidence telemetry from LLM evaluation runs. Each page reads a CSV (or several) of model responses with attached self-confidence signals, and renders:

- Headline KPIs (item count, accuracy, mean confidence, etc.)
- A confidence distribution histogram, stacked correct/wrong
- A calibration breakdown (accuracy by confidence bucket)
- A confusion matrix (for Yes/No benchmarks) or signal-comparison bars
- A per-source breakdown
- A browseable, searchable, filterable item list with the full prompt, model response, two-pass critique, and all telemetry fields

There's also a **cross-cluster correlation matrix** page that pools every Qwen item across benchmarks and shows Pearson/Spearman/Kendall correlations between confidence signals.

The pages are all self-contained: the data is embedded as JSON in a `<script id="data-blob">` tag, all assets are inline, no external dependencies except Google Fonts.

---

## 2. The model lineup

**Currently in the series:**

| Model | Benchmark | Page | Items | Accuracy |
|---|---|---|---|---|
| Qwen3 6-35B-A3B-instruct | GSM8K | `gsm8k_confidence_browser.html` | 145 | 95.2% |
| Qwen3 6-35B-A3B-instruct | MMLU-Pro | `mmlupro_confidence_browser.html` | 74 | 75.7% |
| Qwen3 6-35B-A3B-instruct | TriviaQA | `triviaqa_confidence_browser.html` | 150 | 77.3% |
| Qwen3 6-35B-A3B-instruct | StrategyQA | `strategyqa_confidence_browser.html` | 142 | 74.6% |
| Qwen3 6-35B-A3B-instruct | LegalBench | `legalbench_confidence_browser.html` | 194 (2 subsets) | 65.5% (combined) |
| Gemma4-31B-instruct | GSM8K | `gemma_gsm8k_confidence_browser.html` | 146 | 95.9% |
| Gemma4-31B-instruct | StrategyQA | `gemma_strategyqa_confidence_browser.html` | 98 | 85.7% |

**Cross-cluster:**
- `correlation_matrix.html` — Pearson/Spearman/Kendall correlation matrices over **all Qwen items combined** (currently 511 across 4 benchmarks; LegalBench is not yet included in the matrix). Has a method toggle and a per-benchmark breakout.

**New models incoming** (per user message): **Gemma4** (more data) and **GPT-OSS**. Plan to add pages following the same conventions.

---

## 3. Data format (CSV schema)

All evaluation files share this schema, with some variation across runs.

### Required columns (every file)

```
idx                              — integer, unique within a single source file
question                         — string, the prompt
ground_truth                     — string, the correct answer
model_answer                     — string, what the model said
is_correct                       — boolean
verbalized_confidence            — float 0–10, model's self-rated confidence
single_pass_confidence           — float 0–10, returned alongside single_pass_correct
single_pass_correct              — boolean, model's self-judgment
more_likely_than_not             — boolean, probabilistic self-check
logit_confidence_geom            — float, geometric mean of per-token probabilities of answer
logit_confidence_mean_prob       — float, arithmetic mean of same
logit_confidence_min             — float, weakest token p
seq_confidence_mean              — float, mean log-prob of the full sampled response (large negative numbers)
two_pass_critique                — string, the model's critique of its own answer
two_pass_finish_reason           — string
two_pass_was_truncated           — boolean
main_pass_finish_reason          — string
main_pass_was_truncated          — boolean
full_response                    — string, the raw model output
```

### MMLU-Pro additions (multiple-choice only)

```
answer_token_entropy             — Shannon entropy over A–J letter probs (lower = more peaked)
chosen_answer_raw_prob           — raw probability the model assigned to the picked letter
answer_letter_probs              — JSON-encoded dict of per-letter probabilities
chosen_letter                    — A through J
top_answer_letter                — the letter with the highest p
prob_A, prob_B, … prob_J         — individual per-letter probabilities
```

### Gemma & LegalBench-format additions

```
was_forced                       — boolean. For Gemma on GSM8K/SQA this is always True;
                                   for legalbench it's always False. It's a methodology
                                   field, not a failure signal — DON'T treat it as
                                   "the model gave up". For Gemma it means the prompt
                                   protocol always forced an `Answer: X` finalizer.
forced_answer_response           — string, the forced finalizer
answer_extraction_failed         — boolean, set when the answer parser failed
```

### Important schema quirks observed

- **`mmlu35b__1_.csv` was EXCLUDED** from the MMLU-Pro page — it has a different schema with semantic entropy fields. Don't include it.
- **For StrategyQA Qwen, `answer_token_entropy` and `chosen_answer_raw_prob` columns exist but are all NaN.** The columns are there because of shared schema, but the underlying computation isn't done for yes/no answers. Same for LegalBench Qwen — column present, values all NaN.
- **For GSM8K and TriviaQA Qwen**, the same is true — `answer_token_entropy` / `chosen_answer_raw_prob` are NaN in the consolidated embeds. **Only MMLU-Pro Qwen actually has populated values** for these two signals.

### File naming convention observed

`<N>seed<S><benchmark>_confidence(withnewSE)?_<Model>.csv` where N is item count, S is the random seed. Some files have variations: `__1_`, `_detailed_`, etc.

---

## 4. The build pattern (per cluster)

### Step 1 — Inspect

Always start by checking each new file:
- Shape, accuracy, schema diff vs existing
- Verbalized confidence distribution
- Yes/No bias (for binary benchmarks)
- Overlap with any existing embed (`idx` collisions need checking)

### Step 2 — Consolidate

For each cluster, build an `*_embed.json` file:

```python
frames = []
for label, fname in sources.items():
    df = pd.read_csv(os.path.join(UPLOADS, fname))
    df['__source__'] = label                  # human-readable source key
    df['__source_file__'] = fname             # the filename
    frames.append(df)
big = pd.concat(frames, ignore_index=True, sort=False)

# Verify duplicate idx are consistent across files (they should be — same item evaluated
# multiple times with different seeds should give identical results if deterministic).
# If they're NOT consistent, you may have schema collisions — see LegalBench section below.

def merge_group(grp):
    row = grp.iloc[0].to_dict()
    row['__sources__'] = sorted(grp['__source__'].unique().tolist())
    row['__source_files__'] = sorted(grp['__source_file__'].unique().tolist())
    return row

deduped = pd.DataFrame([merge_group(g) for _, g in big.groupby('idx')])
```

### Step 3 — Clean values for embedding

```python
def clean_val(v):
    if v is None: return None
    if isinstance(v, str):
        if v in ('nan','None','NaN'): return None
        if v in ('True','true'): return True
        if v in ('False','false'): return False
        try:
            f = float(v)
            if math.isnan(f): return None
            if f.is_integer() and abs(f) < 1e15: return int(f)
            return f
        except: return v
    if isinstance(v, float) and math.isnan(v): return None
    return v

# Apply per-field. Rename __field__ → field (drop the double underscore prefix).
```

### Step 4 — Inject into HTML

```python
with open('legalbench_embed.json') as f:
    data = f.read().replace('</', '<\\/')   # CRITICAL: escape </ to prevent
                                            # the </script> tag from breaking
final = template.replace('__DATA_PLACEHOLDER__', data)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(final)
```

The HTML template has this near the end:

```html
<script id="data-blob" type="application/json">__DATA_PLACEHOLDER__</script>
<script>
  const DATA = JSON.parse(document.getElementById('data-blob').textContent);
  // ...
</script>
```

### Step 5 — Update copy strings

Most pages have static numbers in the masthead deck, the meta-strip, the headline title, and the footer that need to match the new data. Search for old values and replace:

```python
# Typical strings to update:
# - The headline title: "Eighty-seven arithmetic problems" → "One hundred forty-five..."
# - Meta-strip: "Source files 5" → "Source files 6"
# - Footer: "5 raw artifacts · 145 unique items"
# - Optionally section subheads
```

### Step 6 — Verify with playwright

Always run a headless chromium check:

```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1400, 'height': 1100})
    errors = []
    page.on('pageerror', lambda e: errors.append(str(e)))
    page.on('console', lambda m: errors.append(f'[{m.type}] {m.text}') if m.type=='error' else None)
    page.goto(f'file://{out_path}')
    page.wait_for_load_state('networkidle')
    page.wait_for_timeout(500)

    diag = page.evaluate('''() => ({
      kpis: document.querySelectorAll('.kpi').length,
      items: document.querySelectorAll('.item').length,
      result_count: document.getElementById('result-count')?.textContent,
      title: document.querySelector('h1.title')?.innerText,
    })''')
    for k,v in diag.items(): print(f'  {k}: {v}')
    real_errors = [e for e in errors if '403' not in e and 'Failed to load' not in e]
    print('  errors:', real_errors or 'none')

    # Always test the interactive bits:
    # - Click "Wrong" filter, verify count matches
    # - Click "Truncated" filter
    # - Hover histogram bars (tooltip should appear)
    # - Click calib row / hist bar (filter chip should appear and item list narrows)
    # - For pages with toggles: cycle through all toggle states
```

The 403 / "Failed to load" errors come from Google Fonts being blocked in the sandbox — those are noise and can be filtered.

### Step 7 — Present

```python
present_files(filepaths=["/mnt/user-data/outputs/<name>.html"])
```

---

## 5. Per-cluster aesthetics (each page has its own identity)

| Cluster | Aesthetic | Fonts | Palette |
|---|---|---|---|
| GSM8K Qwen | Warm editorial magazine | Fraunces + Inter | Cream, warm browns |
| MMLU-Pro Qwen | Cool dark laboratory | Archivo Narrow + IBM Plex Mono | Dark navy, teal accent |
| TriviaQA Qwen | Sepia archival ledger | Newsreader + JetBrains Mono | Cream, sepia |
| StrategyQA Qwen | Modernist editorial | Instrument Serif + PT Sans Narrow + DM Mono | Cream / black / orange (#d56026) |
| LegalBench Qwen | Judicial / court document | Cormorant Garamond + Libre Caslon Text + IBM Plex Mono | Parchment, seal-red (#6b1d1d), court-blue (#2a3e5c), gold |
| Correlation matrix | Statistical journal | EB Garamond + IBM Plex Sans/Mono | Off-white, teal (#1f5666) + orange (#a13816) |
| GSM8K Gemma | Blueprint engineer | DM Mono + Spectral + Inter | Pale blue-grey with grid pattern, engineer blue, burnt orange |
| StrategyQA Gemma | Forest-and-sage | Crimson Pro + JetBrains Mono + Inter | Warm cream, moss green (#4a6a48), rust (#7a3617) |

**Design principle**: every page should have its **own visual identity** but share the same **structural rhythm**. New model+benchmark pages should pick an aesthetic that doesn't collide with existing ones. The aesthetic should *fit* the data's story (e.g., LegalBench is judicial because of subject matter; Gemma GSM8K is engineer-blue because of clean technical competence).

### Shared structural rhythm (always present)

1. Masthead (court-stamp/plate header, big serif title, italic deck paragraph, meta-strip)
2. §01 Headline KPIs (4 cards)
3. §02 Distribution(s) — always include a histogram, **always with hoverable bars** (see §6 below)
4. §03 / §04 — varies by benchmark type:
   - **GSM-style** (open-ended): signal comparison bars (mean signal correct vs wrong)
   - **Yes/No**: confusion matrix + Yes/No bias chart
   - **Multiple-choice**: per-letter probability chart
5. §05 Per-source table
6. §06 Item browser with search + verdict filter + sort

### Family-consistent class names (reuse these)

```
.kpi-row, .kpi, .kpi.notable, .kpi.good       — KPI cards
.panel, .panel h3, .panel .panel-sub          — content panels
.panel-grid                                    — 2-column layout
.hist-svg, .bar-correct, .bar-wrong, .bar-hit, .bar-overlay
                                              — histogram (hoverable bars below)
.hist-legend, .hover-hint                     — hist legend + click hint
.hover-tooltip                                — tooltip floats over chart
.confusion, .confusion .cell, .head, .head-side  — confusion matrix grid
.calib-table, .bar-stack, .bar-correct/.bar-wrong/.pct  — calibration table
.compare-bars                                 — mean-signal-correct-vs-wrong panel
.source-table                                 — provenance table
.controls, .seg, .seg button.on, .seg-lbl    — filter controls
.items-list, .item, .item-head, .item-body   — record browser
.item .idx, .badge, .conf-pip, .preview, .chev  — item-row pieces
.subset-chip                                  — small inline subset label
.filter-chip                                  — active filter banner above items
```

Each page customizes the CSS variables (`--paper`, `--ink`, `--good`, `--wrong`, accent colors) but the structure stays the same.

---

## 6. Hoverable histogram bars (REQUIRED on all new pages)

User explicitly requested this. Every histogram should show a tooltip on bar hover with:
- Bucket value (the confidence number)
- Total items in bucket
- Correct count (in green)
- Wrong count (in red/page-accent)
- Accuracy in bucket (%)

### Reference implementation (drop-in)

```html
<!-- HTML: the histogram canvas + tooltip mount -->
<svg class="hist-svg" id="hist-verb" viewBox="0 0 720 280" preserveAspectRatio="none"></svg>
<div class="hist-legend">
  <span><span class="swatch" style="background: var(--good);"></span> CORRECT</span>
  <span><span class="swatch" style="background: var(--wrong);"></span> WRONG</span>
  <span class="hover-hint">↳ hover bars for counts</span>
</div>

<!-- Tooltip mount near the end of body -->
<div class="hover-tooltip" id="hover-tip"></div>
```

```css
/* Bars + hit-rect spanning the entire column so it's easy to hover */
.hist-svg .bar-correct { fill: var(--good); }
.hist-svg .bar-wrong   { fill: var(--wrong); }
.hist-svg .bar-hit     { fill: transparent; cursor: pointer; }

/* The overlay is rendered conditionally when a bar is active/hovered */
.hist-svg .bar-overlay { fill: none; stroke: transparent; stroke-width: 2; pointer-events: none; transition: stroke 0.08s; }
.hist-svg .bar-hit:hover ~ .bar-overlay,
.hist-svg .bar-overlay.hover { stroke: var(--accent); }

.hover-tooltip {
  position: fixed; pointer-events: none;
  background: var(--ink); color: var(--paper);
  padding: 12px 16px;
  font-family: 'DM Mono', monospace; font-size: 11px;
  z-index: 200; opacity: 0; transition: opacity 0.1s;
  border-left: 3px solid var(--accent);
}
.hover-tooltip.on { opacity: 1; }
```

```js
const tip = document.getElementById('hover-tip');
function showTip(html, e) {
  tip.innerHTML = html;
  tip.classList.add('on');
  tip.style.left = (e.clientX + 16) + 'px';
  tip.style.top  = (e.clientY + 16) + 'px';
}
function moveTip(e) { tip.style.left = (e.clientX + 16) + 'px'; tip.style.top = (e.clientY + 16) + 'px'; }
function hideTip() { tip.classList.remove('on'); }

function drawHist(svgId, field, labelText) {
  const svg = document.getElementById(svgId);
  const W = 720, H = 280, PADL = 36, PADR = 14, PADT = 22, PADB = 38;
  const innerW = W - PADL - PADR, innerH = H - PADT - PADB;
  // Build buckets 0–10 from DATA
  const buckets = Array.from({length: 11}, () => ({c:0, w:0}));
  for (const r of DATA) {
    const v = r[field]; if (v == null) continue;
    const i = Math.round(v); if (i < 0 || i > 10) continue;
    if (r.is_correct) buckets[i].c++; else buckets[i].w++;
  }
  const maxN = Math.max(...buckets.map(b => b.c + b.w), 1);
  const barW = innerW / 11 * 0.78, gap = innerW / 11;

  let html = '';
  // gridlines + y labels, axis line, x label …
  buckets.forEach((b, i) => {
    const total = b.c + b.w;
    const x = PADL + i*gap + (gap - barW)/2;
    const wHt = (b.w/maxN) * innerH, cHt = (b.c/maxN) * innerH;
    const yW = PADT + innerH - wHt - cHt, yC = PADT + innerH - cHt;
    if (total > 0) {
      html += `<rect class="bar-wrong" x="${x}" y="${yW}" width="${barW}" height="${wHt}"/>`;
      html += `<rect class="bar-correct" x="${x}" y="${yC}" width="${barW}" height="${cHt}"/>`;
      html += `<text class="total" x="${x + barW/2}" y="${yW - 6}" text-anchor="middle">${total}</text>`;
    }
    html += `<text class="axis-text" x="${x + barW/2}" y="${H - PADB + 14}" text-anchor="middle">${i}</text>`;
    // Transparent hit-rect spanning the entire column for easy hovering/clicking
    const hitX = PADL + i*gap;
    html += `<rect class="bar-hit" x="${hitX}" y="${PADT}" width="${gap}" height="${innerH}" data-bucket="${i}" data-c="${b.c}" data-w="${b.w}"/>`;
    // Overlay outline (visible only on hover/active)
    const overlayX = x - 3, overlayW = barW + 6;
    const overlayY = total ? (yW - 3) : (PADT + innerH - 1);
    const overlayH = total ? (wHt + cHt + 6) : 1;
    html += `<rect class="bar-overlay" data-bucket-overlay="${i}" x="${overlayX}" y="${overlayY}" width="${overlayW}" height="${overlayH}"/>`;
  });
  svg.innerHTML = html;

  // Wire hover
  svg.querySelectorAll('.bar-hit').forEach(hit => {
    const bucket = parseInt(hit.dataset.bucket, 10);
    const c = parseInt(hit.dataset.c, 10);
    const w = parseInt(hit.dataset.w, 10);
    const total = c + w;
    const acc = total > 0 ? (100*c/total).toFixed(0) + '%' : 'n/a';
    const overlay = svg.querySelector(`[data-bucket-overlay="${bucket}"]`);
    hit.addEventListener('mouseenter', (e) => {
      if (overlay) overlay.classList.add('hover');
      if (total === 0) {
        showTip(`<div class="head">Confidence = ${bucket}</div><div class="stat-row"><span class="lbl">items</span><span class="val">0</span></div>`, e);
      } else {
        showTip(`
          <div class="head">Confidence = ${bucket}</div>
          <div class="stat-row"><span class="lbl">items in bucket</span><span class="val">${total}</span></div>
          <div class="divider"></div>
          <div class="stat-row"><span class="lbl">correct</span><span class="val good">${c}</span></div>
          <div class="stat-row"><span class="lbl">wrong</span><span class="val wrong">${w}</span></div>
          <div class="divider"></div>
          <div class="stat-row"><span class="lbl">accuracy in bucket</span><span class="val">${acc}</span></div>
        `, e);
      }
    });
    hit.addEventListener('mousemove', moveTip);
    hit.addEventListener('mouseleave', () => { if (overlay) overlay.classList.remove('hover'); hideTip(); });
  });
}
drawHist('hist-verb', 'verbalized_confidence', 'verbalized confidence (0–10)');
```

---

## 7. The clickable calibration filter pattern (also part of the standard)

On the LegalBench page (and recommended for any future page with calibration), the calibration table rows and histogram bars are **clickable** — clicking filters the item browser below to just the items in that confidence bucket. The filter shows as a seal-red chip:

```html
<!-- HTML: add a click hint below the calib table -->
<table class="calib-table" id="calib-table"></table>
<p class="calib-click-hint">Click any row to see the items in that confidence bucket below.</p>

<!-- And a chip mount above the items list -->
<div id="conf-chip-mount"></div>
```

```js
// State
let currentConfFilter = null;

function setConfFilter(c) {
  currentConfFilter = c;
  renderCalibration();   // updates active row styling
  renderHistogram();     // updates active bar overlay
  renderConfChip();      // shows/hides chip
  renderItems();         // re-filters list
  if (c !== null) {
    const target = document.getElementById('items-section');
    if (target) target.scrollIntoView({behavior: 'smooth', block: 'start'});
  }
}

function renderConfChip() {
  const mount = document.getElementById('conf-chip-mount');
  if (currentConfFilter === null) { mount.innerHTML = ''; return; }
  const c = currentConfFilter;
  const inBucket = DATA.filter(r => Math.round(r.verbalized_confidence) === c);
  const correctInBucket = inBucket.filter(r => r.is_correct).length;
  const acc = inBucket.length ? (100*correctInBucket/inBucket.length).toFixed(0) + '%' : '—';
  mount.innerHTML = `
    <div class="filter-chip">
      <span class="chip-lbl">Filtered by verbalized confidence</span>
      <span class="chip-val">= ${c}</span>
      <span class="chip-meta">${inBucket.length} items · ${correctInBucket}/${inBucket.length} correct (${acc})</span>
      <button class="chip-clear" id="chip-clear">Clear</button>
    </div>
  `;
  document.getElementById('chip-clear').addEventListener('click', () => setConfFilter(null));
}

// renderItems() should respect currentConfFilter too:
if (currentConfFilter !== null) {
  items = items.filter(r => Math.round(r.verbalized_confidence) === currentConfFilter);
}
```

**Hoisting gotcha**: if `drawHist()` is called at script init and references `currentConfFilter`, declare that state variable BEFORE the `drawHist` definition. Function declarations hoist; `let` doesn't initialize.

---

## 8. Special case — pooling subsets (LegalBench pattern)

LegalBench is the one cluster with **two distinct subtasks** (hearsay rules + privacy/ToS questions) pooled into a single page. The pattern:

1. **Namespace records** with a `__subset__` field at concat time (so `'hearsay'` or `'privacy'`)
2. **Disambiguate idx collisions** using a synthetic uid: `f"{subset}-{idx}"`, kept in `__uid__`
3. **Render KPIs three times** (combined / hearsay / privacy) stacked vertically
4. **Add a global subset toggle** that gates the analytical sections (calibration, histogram, confusion, source table, items)
5. **Add a subset chip** to each item row in the browser, plus a subset filter in the controls

If a future model+benchmark also needs this (e.g., LegalBench under Gemma), use the same pattern. The subset toggle CSS lives in `legalbench.html.template` and can be lifted.

---

## 9. The correlation matrix page

`correlation_matrix.html` pools all **Qwen** items across 4 benchmarks (GSM8K, MMLU-Pro, TriviaQA, StrategyQA) into a cross-cluster meta-view. **It does NOT currently include the LegalBench cluster** (added later, would bump pooled n from 511 to either 605 if hearsay-only or 705 if both subsets). When new data lands, the matrix needs:

1. Recompute `corr_matrix_data.json` using `scipy.stats.{pearsonr, spearmanr, kendalltau}` pairwise, handling NaN per-pair
2. Re-inject into `corr.html.template` at the `__DATA_PLACEHOLDER__`
3. Update the hardcoded pooled count "511" in 7 places of the template (use the sed pattern documented previously, or just do it programmatically)

The page has **three correlation methods toggleable** in §04 (combined matrix) and §05 (per-cluster small multiples), but §01 (the three-method comparison) and §02 (by-benchmark with Pearson only) and §06 (where the methods disagree) are static visualizations. Story of the page: rank-based methods (Spearman/Kendall) reveal that `answer_token_entropy` and `chosen_answer_raw_prob` (MMLU-Pro only) are much stronger predictors than Pearson suggests.

**If extending to Gemma**: the matrix is currently Qwen-only. Adding Gemma would require a **second matrix page** (Gemma matrix) or a fourth dimension on the existing page (model toggle). User's call.

---

## 10. Source files currently in `/mnt/user-data/uploads/` (as of handoff)

### Qwen GSM8K (5 files, 145 unique items)
```
24gsm8k_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv         (5 rows)
1_24gsm8k_confidencewithnewSE1_24Qwen3_6-35B-A3B-instruct.csv    (24 rows)
24_15samples_gsm8k.csv                                            (15 rows)
49_Seed24gsm8k_confidencewithnewSE_detailed_Qwen3_6-35B-A3B-instruct.json  (50 rows)
61seed18gsm8k_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv   (61 rows)
```

### Qwen MMLU-Pro (7 files, 74 unique items)
```
mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct__1_.csv     (14, "main")
21mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct__1_.csv   (5)
23mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv       (5)
24mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv       (5)
25mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct__1_.csv   (5)
21seed88mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv (21)
25seed55mmlupro_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv (25)
```

### Qwen TriviaQA (currently only 1 file on disk, but 150 in embed)
```
35samples_74seedtriviaqa_confidencewithnewSE_Qwen3_6-35B-A3B-instruct__1_.csv  (35)
```
**Note**: the consolidated `triviaqa_embed.json` contains 150 items, so other source files were processed earlier and removed from uploads. If rebuilding from scratch is needed, just keep using the existing embed.

### Qwen StrategyQA (4 files, 142 unique items)
```
15_Seed50strategyqa_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv  (15)
35seed25strategyqa_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv   (35)
50seed40strategyqa_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv   (50)
52seed77strategyqa_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv   (52)
```

### Qwen LegalBench (2 files, 194 unique items, 2 subsets)
```
94seed55legalbench_confidencewithnewSE_Qwen3_6-35B-A3B-instruct.csv  (94, hearsay subset)
100seed23legalbench_confidence_Qwen3_6-35B-A3B-instruct.csv          (100, privacy subset)
```
The two files have completely different questions (different LegalBench subtasks). The 23 idx collisions are just sequential row numbers within each file — NOT the same questions. Namespace by `__subset__`.

### Gemma4 GSM8K (3 files, 146 unique items)
```
5seed1gsm8k_confidencewithnewSE_Gemma4-31B-instruct.csv      (5)
50_SEED50gsm8k_confidencewithnewSE_Gemma4-31B-instruct.csv   (50)
95seed100gsm8k_confidencewithnewSE_Gemma4-31B-instruct.csv   (95)
```

### Gemma4 StrategyQA (2 files, 98 unique items)
```
15seed1strategyqa_confidencewithnewSE_Gemma4-31B-instruct.csv  (15)
85seed20strategyqa_confidencewithnewSE_Gemma4-31B-instruct.csv (85)
```

### Excluded files (do not include)
```
mmlu35b__1_.csv                              — different schema with semantic entropy fields
Screenshot_2026-05-10_at_11_23_41_AM.png     — not a data file
```

---

## 11. Existing embed JSONs in `/home/claude/work/`

These are the "canonical" pre-cleaned data files. Re-using them is preferred over rebuilding from CSVs.

```
embed_data.json              — GSM8K Qwen (145 items)
mmlupro_embed.json           — MMLU-Pro Qwen (74 items)
triviaqa_embed.json          — TriviaQA Qwen (150 items)
strategyqa_embed.json        — StrategyQA Qwen (142 items)
legalbench_embed.json        — LegalBench Qwen (194 items, with __subset__ field)
gemma_gsm8k_embed.json       — GSM8K Gemma (146 items)
gemma_sqa_embed.json         — StrategyQA Gemma (98 items)
corr_matrix_data.json        — pre-computed Pearson/Spearman/Kendall matrices for Qwen
```

**HTML templates** (the structural shells with `__DATA_PLACEHOLDER__`):

```
gsm8k.html.template            (not preserved — generated directly)
mmlupro.html.template
triviaqa.html.template
strategyqa.html.template
legalbench.html.template       — the most complex; includes subset toggle + clickable calib
corr.html.template             — correlation matrix page
gemma_gsm8k.html.template      — blueprint aesthetic, hoverable bars
gemma_sqa.html.template        — forest aesthetic, includes Qwen-bias comparison panel
```

---

## 12. Common pitfalls & gotchas

1. **`</` escaping**: always replace `</` with `<\/` in the data string before injecting into `<script id="data-blob">`, or the embedded JSON will break the script tag and the page won't load.

2. **Boolean handling**: pandas converts booleans to strings on JSON export. The `clean_val` function must handle `'True'`/`'False'` strings, and NaN values. Use the canonical clean_val pasted above.

3. **Refresh pattern is regex-based**: for in-place data updates, find the existing `<script id="data-blob">…</script>` tag, replace just the inner content, leave layout/CSS/JS untouched. The data is fairly large, so don't accidentally try to read the whole HTML into a Python string + use replace on a small substring.

4. **Static count strings**: when refreshing data, the headline title, meta-strip, and footer have **hardcoded counts** that need to be updated separately. Search for the old number (e.g., "49 unique", "Forty-nine"). Common locations:
   - `<h1 class="title">` (often spells out the number)
   - `<div class="docket-strip">` or `.meta-strip` cells
   - `<footer>` row text
   - "n=X" mentions in panel-sub descriptions
   - Anywhere the old accuracy percentage appears (95.2%, 75.7% etc.)

5. **The `was_forced` field is methodology, not failure**: Gemma's GSM8K/SQA files have `was_forced = True` for every item. This is just the eval harness protocol. Don't surface it as a story.

6. **The MMLU-Pro-only fields**: `answer_token_entropy` and `chosen_answer_raw_prob` columns exist across many CSV files but are **only populated for MMLU-Pro**. The other clusters have NaN even when the column is present. Treat them as MMLU-Pro-only when building cross-cluster views.

7. **Subset-aware filtering** (LegalBench): when a confidence filter is active and a subset toggle is set, the conf filter is scoped to the current subset of the analytical view. See `setConfFilter` / `renderConfChip` in `legalbench.html.template`.

8. **Schema collisions**: if two source files for what appears to be the same benchmark have `idx` overlap but the *questions are different*, they're actually different subtasks (this happened with LegalBench). Check question text on overlapping idx before merging — if they're different, namespace via `__subset__`.

9. **Don't include extra/blank rows in CSVs**: pandas sometimes pulls in trailing whitespace rows. Filter on `df['idx'].notna()` if you see weirdness.

10. **Network limits**: bash has network access only to a small allowlist of package mirrors and `api.anthropic.com`. Google Fonts work because they're loaded by the browser viewer, not by bash.

---

## 13. Workflow checklist for adding a new (model, benchmark) cluster

```
[ ] Inspect uploaded CSV(s): shape, accuracy, schema, distribution shape
[ ] Verify any idx-overlap items across files have consistent telemetry
[ ] If overlap items are different questions, plan __subset__ namespacing
[ ] Build the consolidated embed JSON (build_embed pattern in §4 above)
[ ] Pick a distinct aesthetic (table in §5, must not collide visually)
[ ] Choose 1-2 page-specific narratives based on the data:
       - Calibration shape (peaked at 10? bell curve? broken?)
       - Yes/No bias for binary benchmarks
       - Mean signal correct vs wrong gaps
       - Comparisons to peer model/cluster
[ ] Build the HTML template by cloning the closest existing template
       and customizing the masthead deck, KPI sub-copy, and panel narrations
[ ] Always include the hoverable histogram (§6)
[ ] Include calibration row + hist bar click-to-filter (§7) if calibration is
     a story
[ ] Inject data, run playwright verification (§4 Step 6)
[ ] Update the cross-cluster correlation matrix data if this is a new Qwen cluster
       (Gemma/GPT-OSS clusters do NOT currently feed into the matrix — that's
       Qwen-only by design)
[ ] Present file
```

---

## 14. What's likely incoming (from user)

**More Gemma4 data** — likely more GSM8K, StrategyQA, possibly new benchmarks for Gemma (MMLU-Pro? TriviaQA? LegalBench?). Treat new benchmarks for Gemma as **new clusters** with their own pages, not as additions to existing Qwen pages. The series organizes by `(model, benchmark)` tuples — each tuple gets its own page.

**GPT-OSS data** — a new model. Same pattern: each `(GPT-OSS, benchmark)` combination gets its own page, with a freshly chosen aesthetic that doesn't collide with the eight existing pages. Possible aesthetic territories still open:
- Industrial / Bauhaus poster (yellow/red/black)
- Newspaper broadsheet (high serif, narrow columns)
- Scientific datasheet (white, sans-serif, tight grids)
- Vintage computing (CRT green, monospace everything)
- Watercolor / artist's notebook (soft pastels, sketch-like)
- Mineral / archaeological catalog (tan, sepia, bone)

**Cross-model comparison pages may be requested** — e.g., a "Qwen vs Gemma vs GPT-OSS on GSM8K" comparison. The pattern would be similar to the LegalBench dual-subset page but with three models instead of two subsets. Save this for when the user asks; don't pre-build.

**Updates to existing pages** are routine: a new CSV for an existing cluster → the file refresh pattern in §4.

---

## 15. Quick-reference paths

```
Uploads     /mnt/user-data/uploads/
Outputs     /mnt/user-data/outputs/          (where present_files reads from)
Scratch     /home/claude/work/               (embeds + templates live here)
Skills      /mnt/skills/public/              (use docx skill if user asks for a Word doc; otherwise direct file creation is fine)
```

---

## 16. The narrative voice

These pages are written in an **editorial / journalistic register**, not academic or marketing. The deck text always tells a specific story about the data — the *one most interesting finding* surfaced at the top of the page. Examples:

- GSM8K Qwen: "Eighty-seven arithmetic problems, six ways of asking the model how sure it is"
- MMLU-Pro Qwen: "Forty-nine multiple-choice items, A through J, scored against six confidence signals"
- StrategyQA Qwen: "When the answer is YES, the model still likes to say NO"
- LegalBench Qwen: "Coin flips, dressed in robes" → restructured to "The same model, two different witnesses"
- Gemma GSM8K: "Almost always right, *always* says so"
- Gemma StrategyQA: "Gemma reads the question, *then* answers it"

The voice prefers **specific numbers over hedges**, treats the model as a subject with quirks, and isn't afraid of italics for emphasis. Cuts run as italic subordinate clauses set off by em-dashes. Verdict lines at the bottom of contradiction panels read like a closing-argument summary.

When writing new page narratives:
1. Look at the data; find the **one weirdest thing**
2. Name it in the title in a way that's a phrase, not a label
3. The deck paragraph explains the title in concrete numbers
4. The verdict/closing paragraph of each panel re-ties to the same story

---

## 17. One last thing — the `journal.txt` reference

In `/mnt/transcripts/journal.txt` (if accessible to a new chat with transcript memory) there should be a catalog of prior transcripts in this project. The current transcript can be accessed via `/mnt/transcripts/` if the new chat has access. If not, this document is the canonical handoff.

---

## 18. Generation-pipeline fixes for BASE models (added this session)

> Scope note: sections 1–17 are about the **HTML visualization** layer. This
> section is about the **generation/eval pipeline** that produces the CSVs —
> `confidence.py`, `data_utils.py`, `evaluation.py`, `model_utils.py`.
>
> **CANONICAL CODE LOCATION (important):** the real, version-controlled pipeline
> is the git repo at `~/Documents/GitHub/DCAK_RTA_v8-Transfer`, package
> `DavidsDatasets/` (remote: `github.com/glitching-pineapple/DCAK_RTA_v8-Transfer`,
> branch `main`). **All edits must go here** so they can be committed/pushed.
> Earlier in the session some edits were mistakenly made to scratch copies
> (`Downloads/confidence (4).py`, `Desktop/Datasets/` — the latter is a
> partly-broken `medqa` WIP that doesn't compile). Ignore those scratch copies;
> the repo is the source of truth. The repo's `generate_with_logits` is richer
> than the scratch snapshots: it returns a **5-tuple**
> `(text, token_probs, tokens, raw_scores, meta)`, calls `_detect_truncation`,
> and `raw_scores` feeds the MCQ answer-token-entropy feature. `verify_rubric.py`
> enforces that 5-tuple + `_detect_truncation` contract — keep it intact.

### 18.1 The symptom

Base-model runs (Llama-3.1-8B-base, Gemma4-31B-base) had many rows with **empty
`model_answer` / `answer_extraction_failed=True`**, or answers marked wrong
despite the model answering correctly.

### 18.2 Root cause — TWO distinct failure modes, one per model

**Llama-3.1-8B-base → repetition loops.** Greedy decoding with no repetition
control makes a base model collapse into copying the question verbatim forever.
It never reaches `Answer:`, hits `max_new_tokens` (`finish_reason=length`,
`was_truncated=True`), and `model_answer` is empty. Fingerprint: `full_response`
is the question repeated dozens of times. ~5% of GSM8K rows; ~16% of StrategyQA
rows (of which ~half loops, ~half prose-only).

**Gemma4-31B-base → over-generation + last-match parser.** Gemma reliably emits a
clean `Answer:` line, then *keeps writing* — restating the template or
hallucinating new Q&As and answering those. The parser took the **last** `Answer:`
(or grabbed a literal `<YOUR_ANSWER>` placeholder), so a correct answer was
recorded as the continuation's answer. TriviaQA examples: idx 2836 answered
"Orchid" (correct) → stored "Harry Potter…"; idx 4334 "phrenology" → stored "Men
in Black"; idx 11835 "Glamorgan" → stored "A League of Their Own"; idx 13548
"1997" → stored "<YOUR_ANSWER>". These are **correct answers mismarked**.

The Gemma StrategyQA `is_correct=False` rows are mostly a THIRD thing: **genuine
wrong answers** (clean reasoning to the wrong yes/no). Real calibration findings —
not a bug, not fixable by re-running (greedy is deterministic) or by few-shot.

### 18.3 The fixes (DONE — applied in the repo `DavidsDatasets/`, working tree, not yet committed)

All four landed in `~/Documents/GitHub/DCAK_RTA_v8-Transfer/DavidsDatasets/`.
Both files compile; structural rubric checks pass; the parser fix was verified on
the real Gemma failure rows (idx 2836 → recovers "Orchid"; idx 13548 → recovers
"1997"; idx 138 refusal → still `None`, correctly).

1. **Repetition guards in `confidence.py:generate_with_logits`** (Llama loops):
   `repetition_penalty=1.2`, `no_repeat_ngram_size=3` — but **gated on
   `MODEL_VARIANT == "base"`** (auto-resolved from config). Instruct decoding is
   the original code path, untouched → existing instruct results stay
   reproducible byte-for-byte. The n-gram ban makes verbatim question-repetition
   impossible.

2. **Clean-forward-pass measurement on the base path** (keeps logit metrics AND
   answer-token-entropy honest): the guards warp `outputs.scores`. The repo
   version uses `raw_scores` (= `outputs.scores`) for the MCQ
   `extract_answer_token_entropy`, so on the base path we re-derive **both**
   `token_probs` and `raw_scores` from a single clean teacher-forced forward pass
   over `outputs.sequences` (raw, unwarped logits), kept aligned 1:1 with the
   generated tokens and shaped `(1, vocab)` to match. Instruct path still reads
   `outputs.scores` exactly as before.

3. **Stop sequences in `generate_with_logits`** (Gemma over-generation), base
   path only: `stop_strings=["\nQuestion:", "\nAnswer the following", "\nSolution:"]`
   + `tokenizer=tokenizer`. Generation ends at the real attempt.

4. **First-block truncation for all extractors** (Gemma last-match bug): helper
   `_truncate_to_first_block(response)` cuts after the first `Correct: Yes/No`
   line (or first restart marker if truncated). Defined and applied in BOTH:
   - `confidence.py` → `extract_verbalized_confidence`, `extract_more_likely_than_not`
   - `data_utils.py` → `extract_model_answer`, `extract_model_answer_strict`
     (applied right after `_strip_harmony_envelope`). Defined self-contained in
     each file — NOT imported across them, because `confidence.py` already lazily
     imports `data_utils` and a top-level back-import would risk a circular import.

The 5-tuple return and the `_detect_truncation` call are preserved, so
`verify_rubric.py` still passes.

### 18.4 Why we did NOT add few-shot

Wrong tool for these models, deliberately skipped: Llama's problem is decoding
(loops), not format knowledge; Gemma's is over-generation + parsing (more example
blocks = more pattern to continue = worse). And it's a confidence study —
exemplars with `Confidence: N` lines would anchor the verbalized-confidence
distribution, which is a dependent variable. Few-shot only pays off if a model
truly can't produce the format, which neither showed.

### 18.5 Logit-confidence metrics WILL change — expected, not a bug

`seq_confidence_mean` (= log-prob SUM, length-sensitive), `logit_confidence_geom`,
`logit_confidence_mean_prob`, `logit_confidence_min` are aggregates over the
generated tokens, so the fixes legitimately move them:
- **Llama:** the decoded *sequence itself* changes (no loop) → different tokens →
  different aggregates. The clean-forward-pass change does NOT independently shift
  values (old greedy runs had no warpers, so old scores ≈ raw); it only protects
  the metric now that warpers exist. No double-counting — the change is from the
  better sequence.
- **Gemma:** in the OLD files these aggregates were computed over answer **+ all
  continuation tokens**, so continuation rows are diluted/contaminated. New runs
  (with stop sequences) compute them over just the answer block.

### 18.6 Does Gemma need a re-run? + what to do with existing Gemma files

**No full re-run.** Triage existing Gemma CSVs by row type:

| Gemma row type | is_correct | verbalized conf | logit conf | action |
|---|---|---|---|---|
| Clean `eos`, single answer (most SQA, many TriviaQA) | right | valid | valid | **keep as-is** |
| Genuine wrong answer (SQA reasoning errors) | right (model wrong) | valid | valid | **keep — real finding** |
| Continuation / mis-extracted (TriviaQA truncated) | **fix by re-parsing** `full_response` with first-block rule | **contaminated** (two-pass ran on wrong answer) | **contaminated** (aggregated over continuation tokens; per-token probs NOT in CSV → not recoverable) | re-parse `is_correct`; for confidence columns either **drop** or **targeted re-run** those idxs |

Only GPU need for Gemma: the small continuation set IF you want its confidence
signals — and that's a targeted re-run of those idxs (ideally just the two-pass
step), not the whole benchmark. Find them via `main_pass_was_truncated == True`
or >1 `Answer:` in `full_response`.

**Llama DOES need a full re-run** per (model × benchmark): loop rows have no
recoverable answer, and the decoding change alters every row. Leave **instruct**
runs untouched — they had neither failure mode; re-running only breaks
comparability.

### 18.7 Parser-only recovery has limits (measured)

On the 24 empty StrategyQA-Llama rows, conservative patterns recovered ~2 cleanly;
~11 were loops (no answer in text → need the decoding fix); ~11 were prose-only
conclusions where aggressive regex risks the WRONG label. Parser improvements are
free and worth doing but are NOT a substitute for the decoding fix; for a
calibration study a mis-parsed label is worse than a dropped row.

---

## 19. Where things stand + next steps (ready to execute)

### 19.1 Immediate: commit & push (nothing is committed yet)

The four fixes in §18.3 are in the working tree of
`~/Documents/GitHub/DCAK_RTA_v8-Transfer` but **not committed**. Source changes:
`DavidsDatasets/confidence.py`, `DavidsDatasets/data_utils.py`.

```
cd ~/Documents/GitHub/DCAK_RTA_v8-Transfer
git add DavidsDatasets/confidence.py DavidsDatasets/data_utils.py
git commit -m "base-model decoding guards + first-block answer extraction"
git push
```

`git status` will ALSO show `DavidsDatasets/__pycache__/*.pyc` as modified — those
are byte-compiled caches that are (unusually) tracked in this repo; any
compile/import regenerates them. Do **not** stage them (the `git add` above lists
source files explicitly, so they're excluded).

### 19.2 Repo hygiene — stop the `.pyc` churn (user already greenlit asking)

The user is ready to do this. Untrack the caches and ignore them going forward:

```
cd ~/Documents/GitHub/DCAK_RTA_v8-Transfer
printf '\n__pycache__/\n*.pyc\n' >> .gitignore     # create .gitignore if absent
git rm -r --cached DavidsDatasets/__pycache__
git add .gitignore
git commit -m "stop tracking __pycache__ / *.pyc"
```
After this, `*.pyc` modifications stop appearing in `git status`.

### 19.3 NEXT FEATURE — refusal detection (the "before we do that" item)

**The finding that motivates it (idx 138, LegalBench, Gemma2-9B-instruct).**
This was the lone empty `model_answer` in that file. It is NOT a loop and NOT
over-generation — it is a **refusal/abstention**, a *third* failure category:
- The question ("Do the terms imply that NYT makes no assurances…") expects a
  Yes/No, but the model replied *"Please provide the terms you would like me to
  analyze…"* — `main_pass_finish_reason=eos`, not truncated, no `Answer:` line.
- Extraction failed → the forcing fallback fired (`was_forced=TRUE`), but the
  forced text was `"Answer: I need the terms to answer"` — **no Yes/No to parse**
  → `model_answer` empty, `answer_extraction_failed=TRUE`.
- It's the ONLY empty one because other "please provide the terms" refusals in
  that file happened to get forced to `"Answer: No"` (which parsed). idx 138 is
  the case where even the forced pass refused.

**Why none of the §18 fixes touch it:** repetition guards, stop sequences, and
first-block truncation all assume an answer exists somewhere in the text. Here
there genuinely isn't one. The lever for refusals is the prompt or explicit
labeling — not decoding/parsing.

**Spec to implement (conservative — never invent a label):**
1. Add a boolean result field, e.g. `is_refusal`, set True when extraction
   yields no answer (main AND forced both fail to produce a parseable
   Yes/No / number / letter) AND the response matches a refusal pattern such as
   `please provide`, `I need the (text|terms|document)`, `cannot (determine|answer)
   .* without`, `provide the terms`. Keep the pattern tight; prefer undercounting
   refusals to mislabeling a real answer.
2. Wire it in `evaluation.py` at the spot where `model_answer` is computed and the
   forcing fallback runs (search both branches that unpack
   `… = generate_with_logits(...)`). Add `is_refusal` to the result dict; make
   sure `save_utils.py` persists the new column.
3. **Analysis treatment:** exclude refusals from accuracy/calibration, or report
   them as their own bucket. Do NOT force a coin-flip Yes/No — that injects noise
   into the very calibration signal the study measures.
4. Optional, discuss with user first: a stronger forcing prompt that demands a
   Yes/No even under uncertainty. For a calibration study, flagging refusals is
   usually cleaner than suppressing them — confirm which they want.
5. Keep the `verify_rubric.py` contract intact (5-tuple, `_detect_truncation`); add
   a rubric check for the new column if you extend that file.

This was explicitly deferred by the user ("before we do that…") until the
repo-location and git-hygiene issues were sorted — which §19.1–19.2 now cover. So
the next session can go straight to building 19.3.

---

End of handoff document.
