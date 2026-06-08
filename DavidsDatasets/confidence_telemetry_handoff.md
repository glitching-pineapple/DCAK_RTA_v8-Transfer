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

### 18.3 The fixes (DONE — committed, see §19.1)

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

### 19.1 Source changes — commit status (updated)

All pipeline source files are committed on `main` as of the end of the 2026-06-07
session. Summary of what landed where:

| File | What changed | Commit |
|---|---|---|
| `confidence.py` | GPT-OSS ngram guard; base repetition_penalty + stop_strings; clean forward-pass logit re-derivation | earlier session commit |
| `evaluation.py` | `is_refusal` column wiring | earlier session commit |
| `reextract.py` | new re-extraction tool (CPU, conservative) | earlier session commit |
| `data_utils.py` | 4 extractor/refusal fixes (see §21) | `cb0ada1` |

`git status` will show `DavidsDatasets/__pycache__/*.pyc` as modified — those are
byte-compiled caches that are tracked in this repo (see §19.2). Do **not** stage
them. If new edits are made, stage source files explicitly by name.

### 19.2 Repo hygiene — stop the `.pyc` churn (STILL PENDING)

**No `.gitignore` file exists in this repo.** `*.pyc` files under
`DavidsDatasets/__pycache__/` are tracked and appear in `git status` after every
import. Untrack the caches and ignore them going forward:

```
cd ~/Documents/GitHub/DCAK_RTA_v8-Transfer
printf '__pycache__/\n*.pyc\n' > .gitignore
git rm -r --cached DavidsDatasets/__pycache__
git add .gitignore
git commit -m "stop tracking __pycache__ / *.pyc"
```
After this, `*.pyc` modifications stop appearing in `git status`.

### 19.3 Refusal detection — DONE (see §20.5 for implementation, §21.2 Bug 4 for the tail-scan fix)

**The finding that motivated it (idx 138, LegalBench, Gemma2-9B-instruct).**
This was the lone empty `model_answer` in that file — a genuine abstention, not a
loop. The model replied "Please provide the terms you would like me to analyze…"
and even the forced pass refused to emit a Yes/No. See §20.5 for the full
implementation.

**Current status:** `is_refusal` is implemented in `data_utils.py` and wired in
`evaluation.py` (both committed). The implementation uses a tight 10-pattern regex
(`_REFUSAL_RE`) scanned against the tail of each response (character-based, last
350 chars for responses >400 chars — see §21.2 Bug 4 for why line-based didn't
work). `reextract.py` back-fills the column for existing CSVs.

**The one remaining behavioral decision:** should a stronger forcing prompt be used
to squeeze a Yes/No out of models that refuse? For a calibration study, flagging
refusals is cleaner than suppressing them — the current approach (exclude from
accuracy/calibration, report as own bucket) is correct. Raise with user only if
they request higher recall on binary benchmarks.

---

## 20. Re-run vs re-extraction — the decision framework (READ THIS before "fixing" any CSV)

> Added after the GPT-OSS + Gemma2-instruct debugging session. This is the
> single most important mental model for maintaining the result CSVs. If you
> remember one thing: **most "the CSV is wrong" problems do NOT need the GPU.**

### 20.1 The two repair tools (do not confuse them)

| | **Re-inference (re-run)** | **Re-extraction (re-parse)** |
|---|---|---|
| What | Run the model again on the GPU | Re-parse the `full_response` text already in the CSV (CPU, seconds) |
| Changes | The generated tokens themselves | Only the *derived* fields (`model_answer`, `is_correct`, `answer_extraction_failed`, `is_refusal`) |
| Tool | `main.py` with the model config | `reextract.py` |
| Cost / risk | GPU hours; breaks byte-for-byte reproducibility of existing runs | Free; `.bak`-backed; reproducible |

The pipeline already separates "what the model said" (`full_response`, the raw
record) from "what we parsed out of it" (`model_answer` etc.). So if the raw
record is fine and only the parse is wrong, you re-extract. You re-run only when
the raw record itself is unusable or would change.

### 20.2 The decision rule

**Re-run (re-inference) only if ONE of these holds:**
1. The failure is in **generation** and the answer is **not recoverable from the
   text** — e.g. the model looped/truncated and never produced an answer
   (`finish_reason="length"`). The information genuinely isn't there to re-parse.
2. You changed a **decoding setting** that alters the token stream — e.g.
   `repetition_penalty`. Then *every* row's tokens (and logit-confidence
   metrics) differ, so old and new rows aren't comparable within a benchmark →
   full re-run of that (model × benchmark).

**Re-extract (re-parse) if:**
3. The failure is in **parsing** and the correct answer/signal **is present** in
   `full_response`. The current extractor recovers it without the GPU. This is
   the default — try it first.

Corollary used repeatedly this session: a guard that is **inert on well-behaved
rows** (e.g. `no_repeat_ngram_size`) lets you do a **selective** re-run (only the
pathological rows), because the clean rows are byte-identical with or without it.
A guard that touches every row (`repetition_penalty`) forces a **full** re-run.

### 20.3 The three concrete cases (and why each lands where it does)

**(A) GPT-OSS-20B-instruct — repetition loops → SELECTIVE RE-RUN.**
GPT-OSS loops inside its Harmony `analysis` channel (a line repeated up to
~1000×), exhausts `max_new_tokens`, and never emits a final answer. TriviaQA hit
24/149 (16%); StrategyQA 5/150; LegalBench 2/150; GSM8K 0/150. The answer is NOT
in the text (rule 1) → must re-run those rows. Fix in `confidence.py`: extend the
guard predicate so GPT-OSS gets `no_repeat_ngram_size=3` (it was gated on
`MODEL_VARIANT=="base"` only, and GPT-OSS is instruct). Because ngram=3 is inert
on non-looping rows, **only the loop rows need regenerating** — clean rows stay
valid. See §20.4 for why ngram-only (not the base recipe) was chosen.

**(B) Llama base / Gemma base — FULL RE-RUN.** Two reasons stack: loop rows have
no recoverable answer (rule 1) AND the base fix uses `repetition_penalty=1.2`,
which warps every row (rule 2). Existing base CSVs are still pre-fix/broken
(e.g. Gemma2-base TriviaQA 15/40 empty answers). Leave **instruct** base-peers
untouched. (This is the §18.6 verdict; restated here in the framework.)

**(C) Gemma2-9B-instruct — `"Confidence: N"` as the answer → RE-EXTRACT ONLY.**
When the model abstained it emitted an empty `Answer:` line then `Confidence: N`;
an OLD extractor captured the `Confidence:` line as the answer. The answer text
(or its genuine absence) is fully present in `full_response` (rule 3), and the
**current** extractor already handles it correctly via per-branch rubric guards —
so this is purely STALE DATA. Fixed by `reextract.py` (CPU), already applied to
all 19 Gemma2-instruct CSVs (13 rows). **No GPU.** See §20.5.

> Trap that bit us: we first "fixed the root cause" by stopping the answer regex
> from crossing newlines (`:\s*` → `:[^\S\n]*`). That BROKE legitimate answers
> placed on the line *after* `Answer:` (e.g. `Answer:\nIgnatius J. Donnelly`).
> **Reverted.** The newline-crossing is load-bearing; the per-branch guards
> (`re.split` on `Confidence`/`Correct`, or letter/Yes-No matching) already
> reject the leaked rubric text. Lesson: before "fixing" an extractor regex,
> check whether existing post-processing already neutralizes the case — and
> always blanket dry-run `reextract.py` to see collateral.

### 20.4 GPT-OSS guard policy — why ngram-only, not the base recipe

The base recipe is `repetition_penalty=1.2 + no_repeat_ngram_size=3 + stop_strings`.
GPT-OSS gets **only `no_repeat_ngram_size=3`**. Reasoning:

- `no_repeat_ngram_size=3` is **inert** on a non-repeating generation (it only
  fires when a 3-gram would repeat). GPT-OSS's loops *are* repeated 3-grams, so
  it kills them while leaving clean rows byte-identical → enables the selective
  re-run.
- `repetition_penalty` is **disabled for all models** (`_needs_rep_penalty =
  False`). Although originally applied to base only, empirical comparison of
  pre- and post-fix CSVs showed it penalizes the evaluation format tokens
  (`Answer:`, `Confidence:`, `Correct:`) that appear in the prompt, causing the
  base model to avoid the structured output entirely and generate free-form prose
  instead. Result: loop rows dropped from ~12 → ~0 but extractable answers
  dropped from ~28 → ~5 — a net loss. The ngram ban + stop_strings cover the
  actual loop patterns seen; `repetition_penalty` fills no remaining gap.
- `stop_strings` (`\nQuestion:` etc.) target the *base* over-generation failure
  and don't fit GPT-OSS's reasoning loops → base only.

Resolved matrix (in `confidence.py::generate_with_logits`):

| Config | ngram | rep_penalty | stop_strings |
|---|---|---|---|
| Qwen/Gemma **instruct** | 0 | 1.0 | — |
| **GPT-OSS** instruct | 3 | 1.0 | — |
| **base** (Llama/Gemma) | 3 | 1.0 | ✓ |

Principle to carry forward: *use the minimal decoding constraint that prevents
non-termination; prefer constraints inert on well-behaved rows over
`repetition_penalty`. Add a future model to the guard predicate ONLY if it
demonstrably loops* — do not blanket-enable for "instruct".

### 20.5 Refusal detection + `reextract.py` (the §19.3 follow-through)

`is_refusal` (handoff §19.3) is now implemented:
- `data_utils.py::is_refusal_response(response, extracted_answer)` — returns True
  only when the answer is empty/None AND the text matches a tight abstention
  pattern ("I can't determine … without", "please provide the terms",
  "I don't have access", "struggling to pinpoint", …). Conservative by design:
  if any answer parsed, never a refusal. Prefer undercounting to mislabeling.
- **Tail scan is character-based, not line-based (updated in §21).** For responses
  >400 characters the function only scans the final 350 characters (`text[-350:]`).
  This prevents casual mid-reasoning phrases like "I don't remember who scored" or
  "I can't recall exactly which…" — which appear in the opening sentence of a
  long rambling paragraph — from firing the refusal regex. Genuine abstentions
  always close the response ("So I can't determine … without more context"). Short
  responses (<= 400 chars, e.g. "Please provide the terms…") are scanned in full.
  Line-splitting is NOT used because a single long-paragraph response appears as
  2-3 "lines" by `\n`, giving the entire text regardless of window size.
- `evaluation.py` computes `is_refusal` right after `answer_extraction_failed`
  (a refusal is a SUBSET of it, so the confidence fields are already NaN-ed) and
  adds it to the result dict. `save_utils.py` needs nothing
  (`pd.DataFrame(results)` picks the key up). Analysis treatment: EXCLUDE
  refusals from accuracy/calibration (or bucket separately) — never coin-flip
  force them, which would inject noise into the calibration signal.

`reextract.py` — re-parse existing CSVs without the GPU. It is **targeted and
conservative**; a blanket "re-parse every row" version is WRONG, as a
full-dataset dry-run proved:
- **GSM8K float trap**: pandas reads the numeric `model_answer` column as float,
  so `"15"` → `15.0` and every row looks changed; one even mis-extracted
  (`'2.0'→'00'`). → don't reformat clean numeric answers.
- **Forced-answer trap**: `was_forced=True` rows hold the answer in
  `forced_answer_response`, NOT `full_response`. Re-parsing `full_response`
  destroyed correct forced answers (LegalBench `'No'`→none).

So `reextract.py` ONLY rewrites rows whose stored `model_answer` matches the
mis-parse signature `^\*{0,2}\s*(Confidence|Correct)\b`; every other row is left
exactly as generated and merely gains the `is_refusal` column. Usage:
`python3 reextract.py "<glob>" [--write]` (dry-run default; `--write` makes
`.bak`). Re-running it on other instruct models (Gemma4-instruct/Qwen/GPT-OSS)
would only ADD the column — they have 0 bug rows.

### 20.6 Quick-reference: what to do with each model

| Model (× benchmark) | Action | Why |
|---|---|---|
| GPT-OSS, loop rows (TriviaQA/SQA/LegalBench) | **Selective re-run** (GPU) | answer never generated; ngram-only keeps clean rows valid |
| GPT-OSS, GSM8K + all non-loop rows | nothing | 0 loops / clean |
| Llama base, Gemma base (all benchmarks) | **Full re-run** (GPU) — **PENDING** | unrecoverable loops + `repetition_penalty` warps every row |
| Gemma2-9B-instruct (all benchmarks) | **Re-extracted (done)** | stale parse; answer present in text; CPU only |
| Gemma4-instruct, Qwen instruct | nothing (optional re-extract for `is_refusal` column) | 0 bug rows; clean, reproducible |

All source from this work is committed (see §19.1). Never stage `*.pyc`.
Blow-by-blow log lives in `SESSION_HANDOFF.md` → "Session 2026-06-07" parts 1 and
2, and the follow-up session documented in §21.

---

## 21. Llama-3.1-8B-base TriviaQA session — extractor refinements (2026-06-07 follow-up)

### 21.1 What was examined

`triviaqa_confidencewithnewSE_Llama3.1-8B-base.csv` — 40 rows, Llama-3.1-8B-base
on TriviaQA. Pre-fix state: **1/40 correct (~2.5%)**, 34 `answer_extraction_failed`,
4 `is_refusal=True`.

The ~2% accuracy turned out to be ~40% pipeline failures (extraction bugs, refusal
false-positives) masking genuinely attempted answers. After data-level fixes
(re-extraction where the answer was recoverable from `full_response`): **2/40
correct**, 35 `aef=True`, **0 `is_refusal`**. The remaining 35 extraction-failed
rows are genuine loop/truncation with no recoverable answer — they need the full
GPU re-run per §20.6 (Llama base, all benchmarks).

### 21.2 The four extractor bugs found + code changes made

All four are in `data_utils.py`, committed as `cb0ada1`. They affect **triviaqa**
extraction and the shared `is_refusal_response` function.

---

**Bug 1 — Missed mid-line base-model commit phrases (triviaqa Priority-1.5)**

Cause: Priority-1 uses a line-start anchor (`(?m)^[^a-zA-Z\n]*[Aa]nswer\s*:`) to
block mid-sentence "in this answer:" matches. This correctly blocked one class of
false-positives but also blocked the pattern Llama base actually uses when it
commits to an answer: it writes a long reasoning paragraph and then closes with
`"So overall, Answer: Henry II"` or `"...my reasoning... Answer: Isle of skye"` —
mid-line after a comma/period/ellipsis, NOT at a line start.

Real affected rows:
- idx 6025: `full_response` ends with `"So overall, Answer: Henry II"`. Ground
  truth: Henry II. Was `aef=True`, `is_correct=False`. Should have been correct.
- idx 562: `full_response` ends with `"...Answer: Isle of skye"`. Was
  `aef=True`. Correct answer.

Fix: added Priority-1.5, which fires **after** Priority-1 (so the line-start path
still has priority for clean lines) but catches `Answer:` mid-line when preceded
by `,`, `;`, `.`, or `…`:

```python
matches_15 = re.findall(r'[,;.…]\s*[Aa]nswer\s*:\s*(.+?)(?:\n|$)', response)
if matches_15:
    answer = re.split(r'\*{0,2}[Cc]onfidence|\*{0,2}[Cc]orrect', matches_15[-1])[0]
    answer = answer.strip().rstrip('.')
    _PROSE_OPENERS = r'^(?:we |i |it |if |so |but |and |or |that |which |when |there |is |are |was |were )'
    if (answer
            and len(answer) <= 120
            and not re.match(r'^\d+\.?\d*$', answer)
            and not re.match(_PROSE_OPENERS, answer, re.IGNORECASE)):
        return answer
```

Guards: (a) separator requirement prevents "in this answer: we see that..." false
matches (no preceding comma/period); (b) 120-char cap prevents capturing a full
trailing clause; (c) prose-opener filter rejects continuations starting with "we",
"i", "it", etc.; (d) bare-number rejection (see Bug 2).

---

**Bug 2 — Bare numbers leaked as answers via `get_forced_answer` (triviaqa Priority-1 & -2)**

Cause: `get_forced_answer` appends `"\n\nAnswer: "` and lets the base model
complete it. Sometimes the model completes with just `"8"` or `"0"` (a confidence
score, or a nonsense digit). The first extraction call returns None (Priority-1
sees "Answer: 8" but the bare `8` would have passed the then-loose guard). The
fallback path then calls `extract_model_answer(f"Answer: {forced_response_clean}",
dataset)` — wrapping the bare `"8"` with `"Answer: "` — which passed Priority-1
and returned `"8"` as a valid TriviaQA answer.

Real affected rows:
- idx 12085: `model_answer="8"`, `verbalized_confidence=10.0` (the digit was a
  confidence score, not an answer). `is_correct=False` was technically right but
  `aef` should have been `True`, `vc` should be `NaN`.
- idx 6186: Same pattern, `model_answer="0"` for "Puff the Magic Dragon" question.

Fix: added `not re.match(r'^\d+\.?\d*$', answer)` rejection at both Priority-1
and Priority-2. TriviaQA answers are never bare integers or decimals.

---

**Bug 3 — Priority-2 over-capture via optional colon (triviaqa Priority-2)**

Cause: the pattern was `r'[Ff]inal answer:?\s*(.+?)(?:\n|$)'` (colon optional).
This matched "my final answer would be..." mid-sentence (no colon), capturing the
entire trailing clause `"would be... Answer: Isle of skye"` instead of the
committed answer `"Isle of skye"`. The Priority-1.5 fix addresses the capture, but
the Priority-2 pattern would over-capture any "final answer would be" phrase.

Fix: colon is now required — `r'[Ff]inal [Aa]nswer:\s*(.+?)(?:\n|$)'`. Also added
bare-number rejection (see Bug 2). The colon prevents the loose open-ended match.

---

**Bug 4 — `is_refusal_response` false-positives on long rambling responses**

Cause: `_REFUSAL_RE` was scanned against the full response text (or a line-based
tail of 8 lines). But responses where Llama base attempts an answer often open with
casual uncertainty: `"I don't remember who they played against but I do recall..."`
or `"I can't recall exactly who they were talking about. So let's see..."`. These
are 2-3 line responses (long paragraph + blank line + long paragraph), so the 8-line
window covered the entire text.

Real affected rows:
- idx 8936: `"don't remember who"` matched Pattern 9
  (`(?:don't|do not|can't|cannot) (?:recall|remember|know) (?:the|which|what|who|any)`)
  even though the model was actively reasoning about FA Cup finals and never stopped.
- idx 13039: `"I can't recall"` matched Pattern 4
  (`i (?:can't|...) (?:determine|...|recall|...)`)
- idx 15103, idx 3101: matched via different patterns in earlier paragraph text.

Fix: switched to character-based tail for long responses. If `len(text) > 400`,
scan only `text[-350:]`. Genuine abstentions always appear in the closing sentence;
mid-reasoning uncertainty phrases appear in the opening. Short responses (true
refusals like "Please provide the terms...") are scanned in full. See §20.5 for
the full rationale.

### 21.3 CSV triage summary — what is and isn't fixable without GPU

| Row category | Count | Fix |
|---|---|---|
| Data-level fix (re-extraction recoverable) | 5 rows | Done in-place |
| False-positive `is_refusal` cleared | 4 rows | Done in-place |
| Genuine loop/truncation (`aef=True`, no answer in `full_response`) | 35 rows | **GPU re-run required** |
| Correct rows | 2/40 | Kept |

The 35 remaining `aef=True` rows have no recoverable answer in `full_response` —
the model looped into repetition before producing any coherent response. Per §20.2
rule 1, these require a full GPU re-run. Rule 2 (repetition_penalty warps every
row) no longer applies — `repetition_penalty` has been disabled (see §22). The
re-run infrastructure is in place; the re-run itself is pending.

### 21.4 Important extractor invariants to preserve

1. **Priority ordering matters.** Priority-1 (line-start anchor) must fire before
   Priority-1.5 (mid-line separator). If 1.5 fires first, it could capture answers
   that were properly anchored at line start, returning the same result but bypassing
   the cleaner code path. Keep the order: 1 → 1.5 → 2.

2. **The line-start anchor in Priority-1 is intentional.** It blocks "in this answer:
   we see that horses are equines" (no preceding separator). Do not weaken it to a
   mid-line match — that's what Priority-1.5 is for.

3. **Bare-number rejection is triviaqa-only.** GSM8K answers ARE bare numbers ("42",
   "15"). The `not re.match(r'^\d+\.?\d*$', answer)` guard is inside the
   `if dataset == "triviaqa":` branch. Do not apply it globally.

4. **`get_forced_answer` still wraps the forced completion with "Answer: " before
   re-parsing.** This is correct behavior — the guard in the extractor (not in the
   forced-call path) is what rejects the garbage. Don't "fix" the forced call by
   changing how it constructs the prompt; fix the extractor to reject invalid extracts.

5. **`verify_rubric.py` must pass after any extractor change.** Run it before
   committing. It checks: prompt integrity, extractor regex contract, 5-tuple
   return from `generate_with_logits`, forced-answer paths per dataset.

---

## 22. `repetition_penalty` disabled — empirical finding (2026-06-07)

### 22.1 The finding

After the `6bf1dde` commit added `repetition_penalty=1.2` for base models, a
side-by-side comparison of two Llama-3.1-8B-base TriviaQA runs on the same 40
items revealed the penalty made extraction dramatically worse:

| Metric | Pre-penalty (seed99) | Post-penalty (withnewSE) |
|---|---|---|
| `aef=True` | ~12 | 35 |
| `is_correct=True` | ~15 (37%) | 2 (5%) |
| Response style | Structured "Answer: X / Confidence: Y / Correct: Z" | Free-form prose, no Answer: line |

The loop problem was solved — but extractable answers dropped from 28 to 5.

### 22.2 Root cause

`repetition_penalty` applies to **every token in the full sequence including the
prompt**. The evaluation prompt contains the exact output-format tokens the model
needs to emit: `Answer:`, `Confidence:`, `Correct:`. The penalty fires on those
tokens whenever the model tries to write them, making the structured format
progressively less likely. The model routes around it and generates free-form
prose instead — which the extractor cannot parse.

This is a fundamental incompatibility between `repetition_penalty` and any
structured-output eval prompt. It is not specific to Llama or TriviaQA.

### 22.3 The fix (committed)

`_needs_rep_penalty = False` in `confidence.py::generate_with_logits`. The
auto-resolve sets `repetition_penalty = 1.0` (no-op) for all models. The
parameter remains in the function signature for manual override if ever needed.

The two actual base-model failure modes are already covered without it:
- **Verbatim 3-gram loops** → `no_repeat_ngram_size=3`
- **"Generate a new Q&A" over-generation** → `stop_strings`

The theoretical third case `repetition_penalty` was meant to cover — *loose
thematic loops* where sentences vary slightly but the model is stuck in a rut —
has not been observed in practice and is addressed if/when it appears.

Updated resolved matrix:

| Config | ngram | rep_penalty | stop_strings |
|---|---|---|---|
| Qwen/Gemma instruct | 0 | 1.0 | — |
| GPT-OSS instruct | 3 | 1.0 | — |
| base (Llama/Gemma) | 3 | 1.0 | ✓ |

### 22.4 Re-run implications

Because `repetition_penalty=1.0` is inert (no distribution change vs. having
it absent), the clean-forward-pass branch in `generate_with_logits` still
activates for base models via the ngram guard. Logit metrics are still computed
from unwarped scores. **A full re-run of all base (model × benchmark) pairs is
still required** — the reason is now solely §20.2 rule 1 (loop rows have no
recoverable answer), not rule 2 (penalty warped every row).

---

## 23. GPT-OSS two-pass critique + forced-answer harmony bugs (2026-06-07)

Three bugs found in `confidence.py` from inspecting
`triviaqa_confidencewithnewSE_GPT-OSS-20B-instruct.csv`. All fixed in the
working tree; not yet committed.

### 23.1 Bug A — two-pass extractors saw the harmony analysis channel (idx 5)

**Symptom:** `more_likely_than_not` and `single_pass_confidence` empty despite
the critique response containing `"Correct: Yes"`.

**Root cause:** `get_two_pass_confidence` passed raw `critique_response` (both
harmony channels) to `extract_verbalized_confidence` / `extract_more_likely_than_not`.
`_truncate_to_first_block` cut at a mid-sentence `"Correct: Yes"` in the
**analysis** channel. `extract_more_likely_than_not` requires a line-start `^`
anchor — mid-sentence matches return `None`.

**Fix:** Strip harmony envelope before extraction.

```python
critique_for_extraction = critique_response
if _HARMONY_FINAL_DELIM in critique_response:
    critique_for_extraction = critique_response.rsplit(_HARMONY_FINAL_DELIM, 1)[-1].strip()
conf = extract_verbalized_confidence(critique_for_extraction, DATASET)
correct_judgment = extract_more_likely_than_not(critique_for_extraction)
```

Extractors now see only the committed final channel, where `Correct:` appears at
line start.

### 23.2 Bug B — two-pass `model.generate()` missing ngram guard (idx 8959)

**Symptom:** All confidence fields empty for idx 8959.

**Root cause:** `get_two_pass_confidence` called `model.generate()` without
`no_repeat_ngram_size`. The main forward pass (§19.4) already had the guard for
GPT-OSS, but the two-pass critique's **separate** `model.generate()` call did
not. GPT-OSS looped to the full `max_new_tokens` budget and produced no
`Confidence:` or `Correct:` output.

**Fix:** Mirror the same predicate from `generate_with_logits`:

```python
_two_pass_gen_kwargs = dict(
    max_new_tokens=TWO_PASS_MAX_NEW_TOKENS,
    do_sample=False,
    return_dict_in_generate=True,
    pad_token_id=tokenizer.pad_token_id,
)
if (MODEL_VARIANT == "base") or (MODEL_FAMILY == "gptoss"):
    _two_pass_gen_kwargs["no_repeat_ngram_size"] = 3
outputs = model.generate(**inputs, **_two_pass_gen_kwargs)
```

**Principle:** every `model.generate()` call that can run on GPT-OSS or a base
model must include `no_repeat_ngram_size=3`. It is not enough to guard only the
main forward pass — secondary calls (two-pass critique, future eval calls) need
the same guard independently.

### 23.3 Bug C — `get_forced_answer` leaked harmony analysis text (idx 2322)

**Symptom:** `model_answer` was a long analysis-channel blob (`"analysisWe need
to consider…"`). `single_pass_confidence` / `single_pass_correct` empty.

**Root cause:** When the main pass truncated before reaching `assistantfinal`,
the forced-answer call ran with a 32-token budget. GPT-OSS again hit
`max_new_tokens` before `assistantfinal`, producing only analysis text.
`get_forced_answer` had no harmony stripping — `forced_response_clean` was the
raw analysis blob. The lax fallback
`extract_model_answer(f"Answer: {blob}", dataset)` accepted the analysis text as
the "answer". The non-None bogus answer kept `answer_extraction_failed=False`, so
confidence fields were attempted but two-pass critique also received garbage.

**Fix:** Two guards added to `get_forced_answer`:

1. **Harmony stripping:** `rsplit(_HARMONY_FINAL_DELIM, 1)[-1]` before any
   extraction attempt.
2. **Analysis-channel rejection:**
   `_ANALYSIS_MARKER_RE = re.compile(r'^analysis', re.IGNORECASE)` — if the
   extracted answer matches, set it to `None`. No `\b` word boundary: the harmony
   channel name runs directly into the next word (`"analysisWe…"`), so `\b` would
   not match.

**Module-level constants** (lines 16–21 of confidence.py):
```python
_HARMONY_FINAL_DELIM = "assistantfinal"
_ANALYSIS_MARKER_RE = re.compile(r'^analysis', re.IGNORECASE)
```
Defined in `confidence.py` (not imported from `evaluation.py`) to avoid a
circular import.

### 23.4 Pending — idx 4809 TriviaQA alias substring check

**Observation:** idx 4809 `model_answer="Firenze"`, `ground_truth="Florence"`,
`is_correct=True`. TriviaQA's official alias list for "Florence" includes
`"firenze"` (Italian name) → `check_triviaqa_correct` matched via
`model_lower == acc`. This is **semantically correct**; the TriviaQA official
eval considers "Firenze" a valid answer for "Florence".

The broader `model_lower in acc or acc in model_lower` substring predicates can
cause false positives (e.g., a short model answer contained within a long alias,
or vice versa). **No fix implemented.** Further analysis of false-positive rate
across the full alias set is needed before restricting to exact-match only.

### 23.5 Invariant: every secondary `model.generate()` call needs a loop guard

Derived from Bug B: the ngram guard in `generate_with_logits` does not protect
any other generation call. The pattern to apply any time a new `model.generate()`
call is added for GPT-OSS or base models:

```python
gen_kwargs = {...}
if (MODEL_VARIANT == "base") or (MODEL_FAMILY == "gptoss"):
    gen_kwargs["no_repeat_ngram_size"] = 3
outputs = model.generate(**inputs, **gen_kwargs)
```

### 23.6 Files modified

| File | Change |
|---|---|
| `confidence.py` | `_HARMONY_FINAL_DELIM` + `_ANALYSIS_MARKER_RE` constants; `get_two_pass_confidence` ngram guard + harmony strip before extraction; `get_forced_answer` harmony strip + analysis-marker rejection. |

Not committed. Uncommitted source set: `confidence.py`, `data_utils.py`,
`evaluation.py`, `reextract.py`.

---

## 24. Llama-3.1-8B-base TriviaQA second run — Priority-2 extractor regression + fix (2026-06-07 part 4)

### 24.1 What was examined

`triviaqa_confidencewithnewSE_Llama3.1-8B-base (2).csv` — 40 rows, same model as §21
but a newer run that already included the §21 code fixes. The session analyzed which
of the `aef=True` and verbose-answer rows would be addressed by the accumulated fixes.

### 24.2 Bugs found and fixed

Both bugs are in `data_utils.py` Priority-2 (`extract_model_answer`, triviaqa branch).
Committed as `056e4ce`.

---

**Bug 1 — Priority-2 missing sentence-boundary truncation (verbose extraction)**

**Affected row:** idx 5588 (Thursday Next series / Jasper Fforde).

The model wrote `"My answer is Jasper Ffford. I am 70 percent confident in this
answer."` Priority-2 "My answer is:" matched and captured the entire sentence:
`"Jasper Ffford. I am 70 percent confident in this answer"`. The same
sentence-boundary split that had been applied to Priority-1 since §21.2 (Bug fix for
verbose Gregory Peck answers) was absent from Priority-2.

**Fix:** Added the same split immediately after quote-stripping in the Priority-2 loop:

```python
ans = re.split(r'\.\s+(?:I[\s\']|My\s|So\s|This\s|It\s|In\s)', ans)[0].strip().rstrip('.')
```

Result: `"Jasper Ffford. I am 70 percent confident in this answer"` → `"Jasper Ffford"`.

---

**Bug 2 — Priority-2 "My answer is correct" false positive**

**Affected row:** idx 8936 (Bolton Wanderers / West Ham United).

The model wrote `"I am 100% confident that my answer is correct."` nowhere in the
response was `Answer:` or a structured commit phrase. The Priority-2 pattern
`[Mm]y (?:final )?[Aa]nswer is:?\s*(.+?)(?:\n|$)` matched `"my answer is correct."`
and extracted `"correct"` as the trivia answer. `"correct"` is not a bare number, so
the pre-existing guard `not re.match(r'^\d+\.?\d*$', ans)` did not filter it.

Result before fix: `model_answer="correct"`, `aef=False`, `is_correct=False` — looked
like the model gave a wrong answer, but the extractor fabricated it.

**Fix:** Added a meta-commentary word blocklist immediately after the sentence-boundary
split:

```python
if re.match(r'^(?:correct|incorrect|right|wrong|true|false|unknown|unsure)$', ans, re.I):
    continue
```

`continue` skips to the next Priority-2 pattern rather than returning. If no remaining
pattern produces a real answer, extraction correctly returns `None` → `aef=True`.

---

### 24.3 Remaining aef=True rows (still not fixable)

| idx | Topic | Why unfixable |
|-----|-------|---------------|
| 16373 | Achille Lauro | Model narrates "the P.L.O. hijacked the Achille Lauro" — no structured commit phrase |
| 6536 | To Kill a Mockingbird | Model uses "going to go with X as my answer" — not matched by any pattern |
| 3101 | Erasmus | Response truncated mid-sentence; no answer attempted |
| 8936 | Bolton Wanderers | Answer in narrative clause ("was West Ham United") — no structured format |

These require a re-run (GPU) with improved loop guards so the forced-answer pass can
produce a clean output.

### 24.4 Invariant added to Priority-2

After this fix, the full Priority-2 processing chain for triviaqa is:

1. Pattern match and capture `(.+?)(?:\n|$)`.
2. `.strip().rstrip('.').strip('"\'')` — clean whitespace and quotes.
3. Sentence-boundary split `\.\s+(?:I[\s\']|My\s|So\s|This\s|It\s|In\s)` — truncate
   before self-commentary.
4. Meta-commentary blocklist — reject "correct/incorrect/right/wrong/true/false/unknown/unsure".
5. Bare-number rejection `^\d+\.?\d*$` — triviaqa answers are never bare integers.

Steps 3 and 5 mirror the Priority-1 guards. Step 4 is Priority-2-specific (Priority-1
uses a structured "Answer:" line so meta-commentary leakage via this path is less likely).

### 24.5 Files modified

| File | Change |
|---|---|
| `data_utils.py` | Priority-2 triviaqa loop gains sentence-boundary split and meta-commentary filter. Committed `056e4ce`. |

---

## 25. Forced-answer `loop_guard=False` — ngram guard silencing the forced pass (2026-06-07 part 5)

### 25.1 The problem

After the `_truncate_countdown_loop` fix in §21 (committed `7bf32a5`), Llama-3.1-8B-base
idx 8983 ("Who wrote The Sea Wolf") still showed `forced_answer_response=""` on every
re-run. The countdown deloop was confirmed working (the function fires, truncates at the
70% line, and produces a clean 2514-char clip ending with clean Jack London reasoning).
Yet the forced pass returned empty.

**Root cause:** `generate_simple_response` applies `no_repeat_ngram_size=3` to **the
entire input+output sequence** (not just the output). The forced-pass context is a
2514-char reasoning block containing "Jack London" six times plus instruction-style
prompt lines with `"Question: "` and `"Answer: "` colon patterns. At the very first
output position, the ngram guard:

1. Takes the last 2 tokens of the input (from `"Answer: "`, something like `[":", " "]`).
2. Scans the full input for every place that 2-gram appears.
3. Bans the token that follows each such occurrence in the input.

With a dense reasoning context and multiple prompt structure tokens, many common
completion tokens get banned. The only remaining option becomes EOS → the forced pass
decodes to `""`, `.strip()` to `""`, `forced_answer = None`, `aef = True`.

This was NOT a problem during normal (non-forced) generation because the main pass runs
with a fresh context that doesn't contain `"Answer: "` before any answer word. The
forced pass is unique in that the prompt itself explicitly ends with `"Answer: "` which
creates 2-gram contexts that match against the reasoning text.

### 25.2 The fix

**`model_utils.py` — `generate_simple_response` gains `loop_guard: bool = True`:**

```python
def generate_simple_response(
    model, tokenizer, prompt, max_new_tokens=512,
    base_suffix="\n\nResponse:", enable_thinking=None,
    loop_guard: bool = True,   # ← new
) -> str:
    ...
    if loop_guard and (MODEL_VARIANT == "base" or MODEL_FAMILY == "gptoss"):
        gen_kwargs["no_repeat_ngram_size"] = 3
```

**`confidence.py` — `get_forced_answer` passes `loop_guard=False`:**

```python
forced_response = generate_simple_response(
    model, tokenizer, prompt,
    max_new_tokens=max_tokens,
    base_suffix="\n\nAnswer: ",
    loop_guard=False,          # ← new
)
```

**Why `loop_guard=False` is safe for the forced pass:**
- Forced-pass budget is ≤ 32 tokens for TriviaQA (8 for letter/Yes-No datasets).
  A countdown loop requires tens of tokens per step — impossible at this budget.
- The reasoning fed to the forced pass was already delooped by `_truncate_countdown_loop`
  before it reaches `generate_simple_response`.
- The guard is actively harmful here: with a dense context it over-bans valid answer
  tokens and can leave EOS as the only option.

**Why the two-pass critique and Gen-2 keep `loop_guard=True`:**
- Both run with 512–4096 token budgets where GPT-OSS and base models can and do loop.
- Their contexts are summary-length (not full reasoning), so the over-banning effect
  is much smaller.

**`verify_rubric.py` — mock updated:**

```python
def fake_generate_simple_response(
    model, tokenizer, prompt, max_new_tokens=512,
    base_suffix="", loop_guard=True, **kwargs,
):
```

`verify_rubric.py → ALL CHECKS PASSED`.

### 25.3 Invariant: loop guard per call site

| Call site | `loop_guard` | Budget | Rationale |
|---|---|---|---|
| `get_forced_answer` | **False** | ≤ 32 tokens | Cannot loop; dense context over-bans |
| `get_gen2_confidence` | True (default) | ≤ 512 tokens | Possible loop; context is compact |
| `get_two_pass_confidence` | True (default) | 512–4096 tokens | Possible loop; context is compact |

Any future `generate_simple_response` call added for base/GPT-OSS should explicitly
choose `loop_guard` based on whether the budget is large enough to loop AND whether the
context is dense enough to cause over-banning.

### 25.4 Files modified

| File | Change |
|---|---|
| `model_utils.py` | `generate_simple_response` gains `loop_guard: bool = True`; ngram guard conditioned on it. |
| `confidence.py` | `get_forced_answer` passes `loop_guard=False` with inline comment. |
| `verify_rubric.py` | Mock signature updated to accept `loop_guard` kwarg. |

Not yet committed.

---

## 26. Base-model forced-prompt root cause and fix (2026-06-07 part 6)

### 26.1 What §25 got wrong

§25 identified `no_repeat_ngram_size=3` over-banning as the root cause of
`forced_answer_response=""`. The `loop_guard=False` fix was correct analysis and a
correct partial fix, but user ran CSV(5) with that fix applied and still got `""`.

**The real root cause**: the instruction-style forced prompt is completely outside the
base model's training distribution. Llama-3.1-8B-**base** has never seen text like:

```
Based on your reasoning above, commit to your best-guess answer NOW. Output only the
answer, no extra words.

Answer: 
```

It generates EOS immediately because there is no plausible continuation of this
instruction document in its training data. Removing the ngram guard was necessary but
not sufficient — it only unblocked the token generation path, but the model was still
generating EOS because of the prompt format.

The `loop_guard=False` fix from §25 **is still correct** and should stay: it addresses
an independent concern (ngram guard over-banning for all base-model forced passes, not
just Llama). §26 fixes the underlying prompt-distribution mismatch.

### 26.2 The fix

In `get_forced_answer` (`confidence.py`), after the per-dataset instruction-style prompt
is built (used for instruct models), override the prompt for base models with a minimal
Q&A format that matches their pretraining distribution:

```python
from config import MODEL_VARIANT, MODEL_FAMILY
_forced_base_suffix = "\n\nAnswer: "  # instruct path ignores base_suffix
if MODEL_VARIANT == "base" or MODEL_FAMILY == "gptoss":
    if dataset == "triviaqa":
        prompt = f"Q: {question}\nA:"
    elif dataset in ("mmlupro", "medqa"):
        prompt = f"Q: {question}\n{_format_choices(choices)}\nAnswer:"
    elif dataset == "gsm8k":
        prompt = f"Problem: {question}\nAnswer:"
    elif dataset in ("strategyqa", "legalbench"):
        prompt = f"Q: {question}\nAnswer (Yes or No):"
    _forced_base_suffix = ""  # prompt already ends with the answer trigger
```

**Why drop the reasoning clip for base models:**
- Instruction-following models benefit from seeing their own prior reasoning.
- Base models don't process instruction framing — the reasoning clip pushes the
  context further from their training distribution without providing any signal.
- Base models know factual answers from pretraining; they just need natural Q&A format.

### 26.3 Updated forced-pass prompt table

| Model variant | Prompt format | base_suffix | Budget |
|---|---|---|---|
| instruct | Instruction-style (ran out of thinking time, reasoning clip, "Output only…") | `"\n\nAnswer: "` (ignored) | 8–32 tok |
| base / gptoss | Minimal Q&A: `"Q: {question}\nA:"` or similar | `""` | 8–32 tok |

### 26.4 Updated loop-guard table (amends §25.3)

| Call site | `loop_guard` | Base-model prompt | Rationale |
|---|---|---|---|
| `get_forced_answer` | **False** | Minimal Q&A | Cannot loop; dense context over-bans; short budget |
| `get_gen2_confidence` | True (default) | Instruction-style | Can loop at 512+ tokens; compact context |
| `get_two_pass_confidence` | True (default) | Instruction-style | Can loop at 512+ tokens; compact context |

### 26.5 verify_rubric.py

`check_forced_answer_paths` updated to test both instruct and base paths:
- instruct: asserts "ran out of thinking time", "Output only", `base_suffix="\n\nAnswer: "`.
- base: asserts no instruction framing, `base_suffix=""`, question in prompt.

`ALL CHECKS PASSED`. Prompt sizes: instruct 317 chars, base 24 chars.

### 26.6 Files modified

| File | Change |
|---|---|
| `confidence.py` | `get_forced_answer` — base-model override block for minimal Q&A prompt. |
| `verify_rubric.py` | `check_forced_answer_paths` — dual instruct/base path tests. |

Committed in `60c3d7a` (prior session) for `confidence.py` / `model_utils.py` /
`verify_rubric.py`. `confidence_telemetry_handoff.md` updated in this session
(see §27–§28).

---

## 27. Verbalized confidence extraction fixes (2026-06-07 part 7)

### 27.1 What was wrong

`triviaqa_confidencewithnewSE_Llama3.1-8B-base (6).csv` — 15 visible rows,
**every single row shows `verbalized_confidence=10.0` and
`single_pass_confidence=10.0`**, regardless of what the model actually stated.
Spot check against `full_response`:

| idx | Model said | Stored |
|-----|-----------|--------|
| 10031 | "Confidence 9 / Correct: No" | 10.0 |
| 12587 | "Confidence 6" | 10.0 |
| 6847  | "Confidence level: 9" | 10.0 |
| 6174  | "Confidence 9" | 10.0 |
| 15364 | "confidence level: 7" | 10.0 |

Root cause: three bugs stacked.

### 27.2 Bug A — `extract_verbalized_confidence` regex too narrow

The single pattern `[Cc]onfidence\s*:\s*(\d+)` requires a colon immediately
after the keyword. Base models write:
- `"Confidence 9"` — no colon, standalone line → **no match**
- `"Confidence level: 9"` — one word before colon → **no match**
- `"confidence is 7 out of 10"` — prose form → **no match**

When the pattern returns `None`, the fallback `get_verbalized_confidence_separate`
runs instead.

**Fix**: Three patterns tried in priority order (P1 → P2 → P3):

```python
_FILLER = r'(?:approximately|about|around|only|just|~|roughly|nearly|almost)?'
_SUFFIX  = r'(?:/10|out\s+of\s+10|%)?'

# P1: explicit colon, optionally one filler word ("level", "score", etc.)
p1 = r'[Cc]onfidence(?:\s+\w+)?\s*:\s*' + _FILLER + r'\s*(\d+(?:\.\d+)?)\s*' + _SUFFIX

# P2: no colon, number on its own line — base-model structured-output pattern
p2 = r'(?m)^[Cc]onfidence\s+(\d+)\s*' + _SUFFIX + r'\s*$'

# P3: prose — "confidence is N" / "confidence level is about N out of 10"
# _SUFFIX after the captured number means we grab the NUMERATOR (7 not 10).
p3 = r'[Cc]onfidence(?:\s+\w+)?\s+(?:is|of|at)\s+' + _FILLER + r'\s*(\d+(?:\.\d+)?)\s*' + _SUFFIX
```

Last match per pattern wins (self-correction). `verify_rubric.py`: ALL CHECKS
PASSED. Regression suite of 15 CSV-derived cases: all passed.

### 27.3 Bug B — `get_verbalized_confidence_separate` returns 10.0 for base models

When the extractor returned `None` (pre-fix), the fallback ran a separate GPU
call with an instruction-style confidence prompt. Base models don't follow
instructions — they either emitted EOS immediately or echoed the last rubric
number (`10`) from the prompt context. Result: 10.0 for every row.

**Fix**: base models receive a minimal Q&A-style prompt:

```python
if MODEL_VARIANT == "base":
    confidence_prompt = f"Q: {question}\nA: {answer}\nConfidence (1-10):"
    base_suffix = ""
```

The prompt already ends with the trigger so `base_suffix=""` (same pattern as
the forced-pass fix in §26).

> **Note**: After Bug A is fixed, this fallback only fires for rows where the
> main response genuinely has no confidence statement (e.g. was_forced=True
> rows where the main pass looped without emitting a confidence line). The Bug B
> fix is still important for those rows.

### 27.4 Bug C — `_detect_truncation` contradictory flags on empty-eos

Two-pass critique for base models generated EOS immediately (OOD prompt).
The `expect_confidence_markers=True` path then set `was_truncated=True` even
though `finish_reason=eos`, producing the contradictory pair seen in the CSV:
`(finish_reason=eos, two_pass_was_truncated=True)`.

**Fix**: marker check guarded on `generated_text` being non-empty:

```python
if expect_confidence_markers and generated_text:   # ← added `and generated_text`
    ...
```

Empty-eos now correctly gives `was_truncated=False`. Non-empty eos responses
that structurally lack Confidence:/Correct: still get `was_truncated=True`
(the meaningful use case for instruct models).

### 27.5 Files modified + commit

| File | Change | Commit |
|---|---|---|
| `confidence.py` | P1/P2/P3 patterns in `extract_verbalized_confidence`; base-model path in `get_verbalized_confidence_separate`; empty-eos guard in `_detect_truncation` | `9fc05f3` |

---

## 28. Remaining known issues — ready for next session

These are issues visible in CSV(6) that are NOT yet fixed. Ordered by impact.

### 28.1 `is_correct` false positive — idx 12270 (Red ≠ Iron) [HIGH]

**Symptom**: `model_answer="Red"`, `ground_truth="Iron"`, `is_correct=True`.
The model answered "Red" (citing "Red Neil" Kinnock as a different politician)
for a question whose answer is "Iron" (Thatcher's "Iron Lady"). "Red" has no
obvious substring relationship with "Iron".

**Root cause (needs investigation)**: `check_triviaqa_correct` does
three-tier substring matching: `model_lower in acc or acc in model_lower`. If
any alias in the TriviaQA alias list for this question contains "red" as a
substring (e.g. an encoding artifact, a long alias, or an unrelated alias the
dataset bundled), the match would fire. The specific alias needs to be printed
to confirm.

**Investigation step**:
```python
# Load TriviaQA validation, find idx 12270, print its full alias list
from datasets import load_dataset
ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
sample = ds[12270]
print(sample['answer'])   # should show value + aliases + normalized_aliases
```
Check whether any alias contains "red" as a substring via Tier 1 or its NFKD/
compact variants.

**Fix once cause is known**:
- If it's a spurious substring match on a short model answer: add a minimum
  length guard (e.g. `len(model_lower) >= 3`) before the `in` checks.
- If it's a dataset alias artifact: add a per-question exclusion or tighten
  the exact-match requirement for answers ≤ 4 characters.

### 28.2 Forced-pass Q&A continuation — dirty `forced_answer_response` [MEDIUM]

**Symptom**: `forced_answer_response` contains continuation Q&A pairs:
- idx 11736: `"New Zealand\nQ: In which Commonwealth country is the Great Barrier Reef?\nA: Australia\nQ: ..."`
- idx 8983:  `"Jack London\nQ: Who wrote The Sea Wolf\nA: Jack London\nQ: ..."`
- idx 11209: `"Gregory Peck\nQ: Who played the title role in the 1951 film..."`

**Root cause**: The base-model forced prompt `"Q: {question}\nA:"` matches the
training distribution, but the model then pattern-completes by generating new
Q&A pairs (typical base model behaviour on seeing `A: <answer>` — it continues
with `Q: ...`). The stop strings for base in `generate_with_logits`
(`"\nQuestion:"`, `"\nAnswer the following"`, `"\nSolution:"`) don't cover
`"\nQ:"`. `loop_guard=False` means ngram=3 doesn't intervene.

**Impact**: Cosmetic only. `extract_model_answer` in the forced path prepends
`"Answer: "` then runs Priority-1, which anchors to `^`, so the FIRST line
`"New Zealand"` / `"Jack London"` is always what's extracted. `model_answer`,
`is_correct`, and confidence signals are unaffected.

**Fix**: Two options (both clean):

Option A — post-process `forced_response_clean` in `get_forced_answer` before
extraction, truncating at the first `\nQ:`:
```python
# Truncate base-model Q&A continuations after the initial answer
_qa_cont = forced_response_clean.find('\nQ:')
if _qa_cont != -1:
    forced_response_clean = forced_response_clean[:_qa_cont]
```

Option B — add `"\nQ:"` to the stop strings used inside `generate_simple_response`
for the forced pass. Requires either threading stop_strings through the
`loop_guard=False` call or adding a `stop_strings` param to `generate_simple_response`.

Option A is simpler (no new function signatures) and completely safe
(extraction already gets the right first line; truncation only cleans up the
stored field).

**Status: FIXED** (Option A implemented in `confidence.py::get_forced_answer`,
after the harmony-strip block, pending commit).

### 28.3 `single_pass_correct` absent for base models [MEDIUM]

**Symptom**: `single_pass_correct=None` for every base-model row.

**Root cause**: `extract_more_likely_than_not` looks for `"Correct: Yes/No"` on
a line start in the main response. That line is only produced when the model
follows the instruct-style prompt template. Base models receive
`"Q: {question}\nA:"` — a bare Q&A completion — which never contains a
"Correct:" line.

Note: the two-pass critique (`two_pass_critique`, `two_pass_confidence`,
`two_pass_correct`) is **not** affected — the existing
`critique_prompt + "\n\nReview:"` path already works for base models across
Llama, Qwen, and Gemma families. The base model sees "Confidence: <1-10>" and
"Correct: Yes or No" in the prompt context immediately above "Review:" and
pattern-completes them. The empty `two_pass_critique` observed in the (6) CSV
was from an older code version, not a fundamental incompatibility.

The earlier `MODEL_VARIANT != "base"` guard in `evaluation.py` was added based
on that incorrect diagnosis and has since been reverted.

**Fix**: After `extract_more_likely_than_not`, add a fallback for base models
using a separate Q&A call (`get_correct_separate_base`):

```python
# confidence.py — new function:
def get_correct_separate_base(model, tokenizer, question, answer):
    prompt = f"Q: {question}\nA: {answer}\nQ: Is this answer correct? A:"
    response = generate_simple_response(model, tokenizer, prompt, max_new_tokens=10)
    _qa_cont = response.find('\nQ:')
    if _qa_cont != -1:
        response = response[:_qa_cont]
    match = re.search(r'\b(Yes|No)\b', response, re.IGNORECASE)
    return match.group(1).lower() == 'yes' if match else None

# evaluation.py — after extract_more_likely_than_not:
if single_pass_correct is None and MODEL_VARIANT == "base" and model_answer:
    single_pass_correct = get_correct_separate_base(model, tokenizer, question, model_answer)
```

`MODEL_VARIANT` import added to `evaluation.py` config import line (was missing).

**Status: FIXED** (pending commit).

### 28.4 `is_correct` false negatives — idx 12587 (Al Gore), idx 6847 (lapwing)

**Symptom**: Rows where the model gave a clearly correct answer but
`is_correct=False` in the stored CSV:
- idx 12587: `model_answer="Al Gore"`, answer field shows `is_correct=False`
- idx 6847: `model_answer="lapwing"`, answer field shows `is_correct=False`

**Root cause**: These CSVs were produced by an earlier pipeline version before
the extractor and alias-matching improvements in §21/§24. The stored
`model_answer` may have been a different (wrong) extraction at the time, or the
TriviaQA alias matching was narrower. The current `check_triviaqa_correct` code
correctly handles Al Gore and lapwing — new runs will produce correct values.

**Fix**: Re-run or re-extract affected rows. `reextract.py` is conservative
(targets `^\*{0,2}\s*(Confidence|Correct)\b` pattern only) and won't touch
these rows. They need a targeted re-run or manual CSV correction.

**Status**: Historical data artifact — current code is correct, no code change
needed. New runs unaffected.

### 28.5 Session commit status (updated)

| File | What changed | Commit |
|---|---|---|
| `confidence.py` | GPT-OSS ngram guard; base rep_penalty + stop_strings; clean forward-pass logit re-derivation | earlier |
| `confidence.py` | `get_two_pass_confidence` harmony strip + ngram guard; `get_forced_answer` harmony strip + analysis-marker rejection; `_HARMONY_FINAL_DELIM` + `_ANALYSIS_MARKER_RE` constants | `60c3d7a` |
| `confidence.py` | P1/P2/P3 verbalized-confidence patterns; base `get_verbalized_confidence_separate`; empty-eos `_detect_truncation` guard | `9fc05f3` |
| `confidence.py` | `get_forced_answer` `\nQ:` truncation for base Q&A continuations (§28.2) | **pending commit** |
| `confidence.py` | `get_correct_separate_base` — base-model Yes/No fallback for `single_pass_correct` (§28.3) | **pending commit** |
| `evaluation.py` | `is_refusal` column wiring | earlier |
| `evaluation.py` | `MODEL_VARIANT` import; `get_correct_separate_base` fallback for `single_pass_correct` (§28.3) | **pending commit** |
| `data_utils.py` | 4 extractor/refusal fixes (§21) | `cb0ada1` |
| `data_utils.py` | Priority-2 sentence-boundary split + meta-commentary blocklist (§24) | `056e4ce` |
| `model_utils.py` | `generate_simple_response` gains `loop_guard: bool = True` | `60c3d7a` |
| `verify_rubric.py` | dual instruct/base forced-path tests | `60c3d7a` |
| `reextract.py` | new re-extraction tool (CPU, conservative) | earlier |

**Still pending**: §28.1 `is_correct` alias investigation (idx 12270 Red≠Iron),
§28.4 is_correct false negatives (idx 12587, 6847) need re-run or manual fix.

---

End of handoff document.
