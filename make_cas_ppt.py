#!/usr/bin/env python3
"""Simple, clean PPT explaining CAS + CPSS with a toy example."""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

HERE = Path(__file__).resolve().parent

DARK = RGBColor(0x1F, 0x3B, 0x57)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
GREEN = RGBColor(0x2F, 0x8F, 0x4E)
RED = RGBColor(0xB3, 0x2D, 0x2D)
GRAY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xEC, 0xF2, 0xF8)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def title_bar(slide, text, subtitle=None):
    bar = slide.shapes.add_shape(1, 0, 0, prs.slide_width, Inches(1.05))
    bar.fill.solid(); bar.fill.fore_color.rgb = DARK
    bar.line.fill.background()
    tf = bar.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(32); p.font.bold = True; p.font.color.rgb = WHITE
    if subtitle:
        p2 = tf.add_paragraph(); p2.text = subtitle
        p2.font.size = Pt(16); p2.font.color.rgb = RGBColor(0xBF, 0xD5, 0xE8)


def bullets(slide, items, left=0.7, top=1.4, width=12.0, height=5.6, size=22):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if isinstance(item, tuple):
            text, level, bold, color = item
        else:
            text, level, bold, color = item, 0, False, None
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ("• " if level == 0 else "   – ") + text
        p.font.size = Pt(size if level == 0 else size - 3)
        p.font.bold = bold
        p.font.color.rgb = color if color else RGBColor(0x22, 0x22, 0x22)
        p.space_after = Pt(10)
    return box


def flow_box(slide, x, y, w, h, text, fill=ACCENT, font_size=16):
    shp = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    tf = shp.text_frame
    tf.word_wrap = True
    tf.text = text
    for p in tf.paragraphs:
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(font_size); p.font.bold = True; p.font.color.rgb = WHITE
    return shp


def arrow(slide, x, y, w, h=0.4):
    shp = slide.shapes.add_shape(3, Inches(x), Inches(y), Inches(w), Inches(h))  # right arrow
    shp.fill.solid(); shp.fill.fore_color.rgb = GRAY
    shp.line.fill.background()


# ---------------- Slide 1: Title ----------------
s = add_slide()
bg = s.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = DARK; bg.line.fill.background()
tb = s.shapes.add_textbox(Inches(1), Inches(2.4), Inches(11.3), Inches(2.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Clause-Activation Signatures (CAS) with CPSS"
p.font.size = Pt(44); p.font.bold = True; p.font.color.rgb = WHITE
p2 = tf.add_paragraph(); p2.text = "A simple explanation — how we find out WHICH attack it is, and WHY"
p2.font.size = Pt(24); p2.font.color.rgb = RGBColor(0xBF, 0xD5, 0xE8)
p3 = tf.add_paragraph(); p3.text = "Based on: CAS for Attack-Family Attribution in IoMT"
p3.font.size = Pt(16); p3.font.color.rgb = RGBColor(0x8F, 0xAA, 0xC8)

# ---------------- Slide 2: The problem ----------------
s = add_slide()
title_bar(s, "The Problem We Solve")
bullets(s, [
    ("A hospital network (IoMT) has many connected medical devices.", 0, False, None),
    ("An intrusion detector can say: “this network traffic is an ATTACK”.", 0, False, None),
    ("But the security analyst needs more:", 0, True, None),
    ("WHICH attack family is it? (DDoS, DoS, MQTT, Recon, Spoofing...)", 1, False, DARK),
    ("WHY does the model think so? (what evidence?)", 1, False, DARK),
    ("Neural networks give a label, but no human-readable reason.", 0, False, None),
    ("CAS goal: give the attack family + a small, readable set of evidence rules.", 0, True, GREEN),
])

# ---------------- Slide 3: Tsetlin clause ----------------
s = add_slide()
title_bar(s, "Building Block: Tsetlin Machine Clauses")
bullets(s, [
    ("A Tsetlin Machine (TM) learns simple IF-rules called clauses.", 0, False, None),
    ("Each clause checks a few yes/no facts about the network flow.", 0, False, None),
    ("Example clause for the class DDoS:", 0, True, None),
    ("IF  packet_rate is HIGH  AND  protocol is TCP  AND  dst_port is 1883", 1, False, ACCENT),
    ("THEN vote for “DDoS”  (with a learned weight, e.g. +7)", 1, False, ACCENT),
    ("Each class (Benign, DDoS, DoS, ...) has its own clauses (we used 400 per class).", 0, False, None),
    ("So after training we have a big pool of clauses: 6 classes × 400 = 2,400 clauses.", 0, True, None),
    ("Problem: the pool is big and messy. Some clauses are useless or redundant.", 0, False, RED),
])

# ---------------- Slide 4: Big picture ----------------
s = add_slide()
title_bar(s, "CAS in One Picture — 4 Steps")
y = 2.6; w = 2.55; h = 1.5; gap = 0.45
steps = [
    ("1. Train TM", "get 2,400 clauses", ACCENT),
    ("2. Prune", "drop weak clauses\n2,400 → 390", ACCENT),
    ("3. CPSS", "keep clauses that are\npicked again & again", ACCENT),
    ("4. Score", "match evidence,\nname the attack", GREEN),
]
for i, (t, sub, c) in enumerate(steps):
    x = 0.6 + i * (w + gap)
    flow_box(s, x, y, w, h, t + "\n" + sub, fill=c, font_size=16)
    if i < 3:
        arrow(s, x + w + 0.06, y + h / 2 - 0.2, gap - 0.12)
tb = s.shapes.add_textbox(Inches(0.7), Inches(4.55), Inches(12), Inches(1.2))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Input: one new network flow  →  Output: attack family + which rules support / oppose it"
p.font.size = Pt(20); p.font.bold = True; p.font.color.rgb = DARK
p2 = tf.add_paragraph()
p2.text = "Everything is traceable: the final answer is built from clauses we can read."
p2.font.size = Pt(18); p2.font.color.rgb = GRAY
pic = s.shapes.add_picture(
    str(HERE / "IEEE-conference-template-062824 2" / "fig_cpss_protocol.png"),
    Inches(0.7), Inches(5.75), width=Inches(11.9))
cap = s.shapes.add_textbox(Inches(0.7), Inches(7.05), Inches(11.9), Inches(0.4))
cp = cap.text_frame.paragraphs[0]
cp.text = "The real pipeline from the paper (Fig. 1): signature extraction on top, scoring below."
cp.font.size = Pt(12); cp.font.color.rgb = GRAY

# ---------------- Slide 4b: Contributions ----------------
s = add_slide()
title_bar(s, "What Is New in This Paper? (Contributions)")
bullets(s, [
    ("1. CAS attribution layer", 0, True, ACCENT),
    ("Run CPSS on the frozen TM clause pool → get signed evidence", 1, False, None),
    ("(rules that support / oppose each attack family) for every flow.", 1, False, None),
    ("2. Pruning with a guarantee", 0, True, ACCENT),
    ("A simple rule to remove weak clauses, plus a margin condition that", 1, False, None),
    ("tells us when pruning cannot change the model's answer.", 1, False, None),
    ("3. Empirical study", 0, True, ACCENT),
    ("Full vs pruned pools on real IoMT data (CICIoMT2024),", 1, False, None),
    ("both multiclass (6 families) and binary (Benign vs Attack),", 1, False, None),
    ("plus a check showing the signs carry the information.", 1, False, None),
], size=22)

# ---------------- Slide 5: Step 1+2 ----------------
s = add_slide()
title_bar(s, "Step 1 & 2: Train, Then Throw Away Weak Clauses")
bullets(s, [
    ("Step 1 — Train a weighted Tsetlin Machine on IoMT traffic.", 0, True, None),
    ("Every clause gets a learned weight: big weight = important for the TM.", 0, False, None),
    ("Step 2 — Pruning (optional but helps):", 0, True, None),
    ("Remove the clauses with the smallest weights (the weak ones).", 1, False, None),
    ("Safety rule: if the model is confident by more than 2×(removed weight),", 1, False, None),
    ("its decision cannot change → pruning is safe.", 1, False, None),
    ("Result: 2,400 clauses → only 390 clauses (83.75% removed!)", 0, True, GREEN),
    ("And the TM still works almost the same (F1: 0.692 → 0.688).", 0, False, None),
    ("A smaller pool makes the next step cleaner.", 0, False, GRAY),
])

# ---------------- Slide 6: CPSS idea ----------------
s = add_slide()
title_bar(s, "Step 3: CPSS — the Key Idea", "Complementary-Pairs Stability Selection")
bullets(s, [
    ("Question: which clauses really identify a class (e.g. DDoS vs the rest)?", 0, True, None),
    ("Simple idea: a truly useful clause is picked AGAIN and AGAIN,", 0, False, None),
    ("even if we shuffle and split the data differently.", 0, False, None),
    ("How CPSS works:", 0, True, None),
    ("Split the training data into two halves. Do this B times (we tried 5–25).", 1, False, None),
    ("On every half: fit a simple sparse model (L1 logistic regression) that", 1, False, None),
    ("predicts “DDoS or not” using only the clause activations (which clauses fired).", 1, False, None),
    ("Count: in how many halves was each clause picked?", 1, False, None),
    ("Keep only clauses picked often enough (e.g. ≥ 80% of the halves).", 0, True, GREEN),
    ("This is called stability selection (Meinshausen & Bühlmann; Shah & Samworth).", 0, False, GRAY),
])

# ---------------- Slide 7: toy example table ----------------
s = add_slide()
title_bar(s, "Toy Example: CPSS for the Class “DDoS”")
bullets(s, [
    ("Imagine only 6 clauses and B = 5 pairs (10 halves). Threshold = 8 out of 10.", 0, False, None),
], top=1.25, height=1.0)

rows = [
    ("Clause (readable rule)", "Picked in", "Keep?", "Sign", "Meaning"),
    ("c1: high packet rate AND TCP", "10 / 10", "YES", "+", "supports DDoS"),
    ("c2: many unique dst ports", "9 / 10", "YES", "−", "argues against DDoS"),
    ("c3: dst port = 1883 (MQTT)", "9 / 10", "YES", "−", "argues against DDoS"),
    ("c4: small packets AND UDP", "3 / 10", "no", "—", "unstable → drop"),
    ("c5: long flow duration", "2 / 10", "no", "—", "unstable → drop"),
    ("c6: src port is random", "5 / 10", "no", "—", "not stable enough"),
]
tbl_shape = s.shapes.add_table(7, 5, Inches(0.7), Inches(2.15), Inches(11.9), Inches(4.0))
tbl = tbl_shape.table
tbl.columns[0].width = Inches(4.6)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(16)
            if r == 0:
                p.font.bold = True; p.font.color.rgb = WHITE
            else:
                p.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
        if r == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = DARK
        elif r % 2 == 0:
            cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT
        else:
            cell.fill.solid(); cell.fill.fore_color.rgb = WHITE
        if r in (2, 3) and c == 2:
            for p in cell.text_frame.paragraphs:
                p.font.color.rgb = GREEN; p.font.bold = True
tb = s.shapes.add_textbox(Inches(0.7), Inches(6.4), Inches(12), Inches(1))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "Signature for DDoS = {c1 (+), c2 (−), c3 (−)}  — small, stable, and readable."
p.font.size = Pt(18); p.font.bold = True; p.font.color.rgb = DARK

# ---------------- Slide 8: signs ----------------
s = add_slide()
title_bar(s, "Step 3b: The Sign Matters — Evidence FOR and AGAINST")
bullets(s, [
    ("Each kept clause gets a sign from the model’s coefficient:", 0, False, None),
    ("Positive sign (+)  →  “inculpatory” evidence: supports this attack family.", 1, True, GREEN),
    ("Negative sign (−)  →  “exculpatory” evidence: argues against this family.", 1, True, RED),
    ("Why this is useful for an analyst:", 0, True, None),
    ("Not just “the model says DDoS”, but:", 1, False, None),
    ("“It looks like DDoS because clause c1 fired,", 1, False, None),
    ("and nothing argues strongly against it.”", 1, False, None),
    ("Sanity check from the paper: if we IGNORE the signs,", 0, False, None),
    ("macro-F1 collapses from 0.765 to 0.025 → the signs carry the real information.", 0, True, DARK),
])

# ---------------- Slide 9: scoring ----------------
s = add_slide()
title_bar(s, "Step 4: Scoring a New Network Flow")
bullets(s, [
    ("A new flow arrives. We check which signature clauses fire on it.", 0, False, None),
    ("Score for each class =", 0, True, None),
    ("(stability of fired “for” clauses  −  stability of fired “against” clauses)", 1, False, DARK),
    ("÷  total stability of “for” clauses", 1, False, DARK),
    ("Toy example — new flow X, checking class DDoS:", 0, True, None),
    ("c1 (+, stability 1.0) fires   →  +1.0", 1, False, GREEN),
    ("c2 (−, stability 0.9) does NOT fire   →  no penalty", 1, False, None),
    ("c3 (−, stability 0.9) fires   →  −0.9", 1, False, RED),
    ("Score(DDoS | X) = (1.0 − 0.9) / 1.0 = 0.1", 1, True, DARK),
    ("We compute this score for all 6 classes; the highest score wins.", 0, False, None),
    ("Every point in the score is traceable to a readable clause.", 0, True, GREEN),
])

# ---------------- Slide 10: results ----------------
s = add_slide()
title_bar(s, "Does It Work? Results on CICIoMT2024 (IoMT dataset)")
rows = [
    ("Method", "macro-F1 (6 attack families)", "What it means"),
    ("TM alone (full pool)", "0.692", "good detector, no explanation"),
    ("CAS (full pool)", "0.765", "evidence-based answer is better"),
    ("CAS (pruned pool)", "0.787", "best: fewer clauses, better answer"),
]
tbl_shape = s.shapes.add_table(4, 3, Inches(0.7), Inches(1.6), Inches(11.9), Inches(2.4))
tbl = tbl_shape.table
tbl.columns[0].width = Inches(3.6); tbl.columns[1].width = Inches(3.6); tbl.columns[2].width = Inches(4.7)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c); cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(17)
            p.font.color.rgb = WHITE if r == 0 else RGBColor(0x22, 0x22, 0x22)
            if r == 0: p.font.bold = True
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK if r == 0 else (LIGHT if r % 2 == 0 else WHITE)
bullets(s, [
    ("Binary (Benign vs Attack): CAS reaches AUROC 0.929 vs 0.874 for the TM.", 0, False, None),
    ("Pruning removed 83.75% of clauses and the answer got BETTER, not worse.", 0, False, None),
    ("Signatures are compact: e.g. 63 clauses explain Benign instead of 167.", 0, False, None),
], top=4.5, size=20)

# ---------------- Slide 11: summary ----------------
s = add_slide()
title_bar(s, "Summary in 4 Sentences")
bullets(s, [
    ("1. Train a Tsetlin Machine → it gives thousands of readable IF-rules (clauses).", 0, False, None),
    ("2. Throw away weak clauses (pruning) → smaller, cleaner pool.", 0, False, None),
    ("3. CPSS keeps only clauses picked again and again across data splits,", 0, False, None),
    ("with a sign: supports (+) or argues against (−) each attack family.", 1, False, None),
    ("4. A new flow is scored by matching this signed evidence → attack family + reasons.", 0, False, None),
    ("Honest limitations:", 0, True, RED),
    ("Tested on one capped dataset; needs validation on other data.", 1, False, None),
    ("Signatures are evidence, not proof of a cause.", 1, False, None),
], size=22)

out = str(HERE / "CAS_CPSS_simple_explained.pptx")
prs.save(out)
print("saved:", out)
