# Humanizing generated prose

Trimmed from a general-purpose "humanize text" skill, with a few ideas absorbed from
[lguz/humanize-writing-skill](https://github.com/lguz/humanize-writing-skill) (MIT). Only what
applies to writing the next chapter of a novel survived; see *Removed on purpose* at the end for
what went and why.

Two jobs: strip the phrasing that makes prose read as machine-written, and make the voice match
the chapters already in `master_book.md`.

Where this is implemented:

| Piece | Where |
|---|---|
| Rules sent with every writing call | `src/nextchap/prompts/humanize.md`, injected into `writer_system.md` |
| Banned phrases the style guide enforces | `src/nextchap/prompts/style_default.md` then `critic.banned_phrases()` |
| Deterministic checks (stock phrases, flat rhythm) | `src/nextchap/pipeline/humanize.py` |
| Who acts on a finding | `critic_system.md`; flagged scenes go through `rewrite_scene` |
| Off switch | `generation.humanize = false` |

This file is the rationale, not the deliverable: the prompt is short on purpose because it ships
in the system message of every scene. It is not a rewrite pipeline of its own.

## 1. The book outranks this document

The existing chapters are the writing sample. Match them before applying anything below.
Extract, from a few recent chapters:

1. **Sentence length pattern.** Variance, signature fragments, one recurring long-sentence shape.
2. **Word choice level.** Plain ("thing", "stuff") or ornate ("the scent of iron and rain")?
   Contractions or not?
3. **Paragraph shape.** Long paragraphs that build, or one sentence per line? Does the book
   ever close a paragraph on a beat?
4. **Paragraph and chapter openers.** Straight into a scene, or a line of context first?
5. **Punctuation habits.** Em dashes, ellipses, parentheses, semicolons, dashes in dialogue.
6. **Verbal tics.** Phrases the narration and each character reuse.
7. **Transition style.** Explicit connectors ("Later that night,") or none at all?

When anything below conflicts with the sample, **the sample wins**. Never upgrade the vocabulary
or the register of the book. An ornate book stays ornate; a terse one stays terse.

## 2. LLM tells to strip

| Tell | Do instead |
|---|---|
| Negation pivot: "not just X, it's Y", "not X but Y", "more X than Y" | Say what the thing is. |
| Significance inflation: "stood as a testament to", "marked a pivotal moment", "something shifted" | Make the concrete claim or cut the line. |
| Copula avoidance: "served as", "played the role of", "boasted" | "was", "had". |
| Tricolon (three parallel beats), anaphora, chiasmus, balanced parenthetical pairs | Break the symmetry. Two beats, or make the third one different. |
| Mini-aphorism paragraph closer ("That was the part that stuck.") | Fold it into the previous sentence or cut it. |
| Reveal framing: "turns out", "what he didn't expect was", "he realised then" | State the discovery directly. |
| Either/or framing of a messy situation | Name the actual situation. |
| Hedge stacking: "somewhat", "a mix of", "couldn't help but", "seemed to" | Assert, unless the POV genuinely doesn't know. |
| Vague attribution: "some said", "people whispered", "the stories said" | Name who, or drop it. |
| Generic sensory filler: "the air was thick with", "a shiver ran down her spine" | One specific detail instead: what she smells, hears, touches. |
| The explanatory sentence after a beat already landed | Cut it. Trust the line. |
| Recaps and summary closers at chapter end | End on the last dramatic beat. |
| Elegant variation (synonym cycling): "the protagonist… the main character… the central figure" | Pick one noun per referent; vary with a pronoun. |
| Filler wrappers: "due to the fact that", "in the event that", "has the ability to", "with regard to" | "because", "if", "can", "about". |
| "It is important to note", "in many cases", "generally speaking" | Cut. |
| Rhetorical question answered in the next breath ("Why did it matter? Because—") | Say it once, directly. |
| **Secondary convergence:** the banned phrase is gone and the same replacement lands everywhere, or every sentence now opens the same way | The fix for a cliché is never a different cliché. Vary the substitute and the opener. |
| Every paragraph the same length, and every paragraph closed on a beat | Let paragraph length follow the scene's energy. Some paragraphs just stop. |
| Prose visibly straining to look human (manufactured quirks, staged sloppiness) | Over-correction is its own tell. Slightly awkward beats deliberately awkward. |

Dialogue is where the machine voice hides best: no character explains their own subtext, no
speech is a tidy thesis, no two characters share one vocabulary.

## 3. Rhythm

Uniform sentence length is the loudest tell and it survives word swaps.

- Alternate. Follow a long unfolding sentence with a short one. Fragments count as sentences.
- At least one sentence of six words or fewer per stretch of ~150 words.
- Never more than three consecutive sentences of similar length. Mid-scene is where prose
  collapses into a single comfortable band.
- If every sentence is grammatically complete and every clause connected, the paragraph is too
  smooth. One thought per scene may go slightly unresolved.
- Paragraphs vary too. The same length everywhere, each one closed with a beat, is the same
  tell one level up. `dominant_opener()` in `pipeline/humanize.py` measures the convergence
  case: one non-function word opening 30% or more of a chapter's non-function sentence openers.

## 4. Controlled irregularity

Real prose breaks grammar in specific, deliberate ways. Use these **only where the source
chapters do** — this is voice, not damage:

- Fragments for emphasis. Not tonight.
- Sentences that open with And, But, So.
- A comma splice in narration during fast action.
- Dropped subjects and contractions in dialogue; interruptions and trailing off ("I never
  said—").
- A run-on in a breathless or panicked voice.
- Repeating a word rather than cycling a synonym.
- A sentence that starts one way and changes direction.

**Never** as a "human touch": typos, misspellings, wrong homophones, garbled word order,
inconsistent names, drifting tense or POV. Those read as broken, and they contradict the
established text.

## 5. Banned in this book

`style_default.md` (which `critic.banned_phrases()` reads) and `pipeline/humanize.py::TELLS`
hold the enforced list; keep the three files in sync. The quality bar for a new entry: it reads as
machine-written *in fiction*, or it is a chapter-fiction cliché. Not business jargon, not
marketing copy, and not a word the book itself uses.

Before a chapter ships, scan for these and quote each hit — not from memory. Regenerated prose
reintroduces the same tells at the same rate as the first draft.

## 6. Hard limits

- Never invent lore, names, numbers or facts to gain specificity. Detail comes from the world
  state, the plan, and the retrieved context, or it doesn't come.
- A rewrite pass may change expression only, never what happens.
- No structural additions the source book doesn't use: no headings, lists, scene labels, or
  author's notes.
- POV, tense and person never drift, however good the "better register" looks.

## Removed on purpose

The original skill served one-shot chat rewrites across many registers. Cut, because a novel
pipeline has different constraints:

- **Anti-detection framing and detector theory** — perplexity/burstiness metrics, stdev targets,
  word-count gates, GPTZero/Pangram/Binoculars, RAID. We are matching a book's voice, not beating
  a classifier, and the numbers can't be counted reliably inside a generation pass.
- **Best-of-N scoring, iterative paraphrase, self-rewrite distance checks, embedding-guided
  synonym swaps, watermark stripping.** Cost, and they degrade continuity.
- **Curly-quote normalization and the em-dash-per-300-words cap.** Typography follows the source
  text; the writer's output is plain UTF-8 prose, not a submission to a form.
- **Output-shape rules** — no preamble, no changelog, append-a-note-when-input-has-no-anchors.
  The writer prompt already returns prose only.
- **Every non-fiction register** — technical, business, Slack, blog, essay, email, academic.
- **Generating from scratch, "human text" decoding parameters** (temperature/top-p advice) — the
  engine owns model parameters and the client only sends supported ones.
- **The knowledge-cutoff, sycophancy and chat-artifact word lists.** They cannot appear in
  chapter prose because there is no conversation to leak into it.
