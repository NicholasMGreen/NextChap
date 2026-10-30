Record what CHANGED in the world state because of Chapter $chapter.

Rules:
- upsert_characters: only characters who are new or whose fields changed. Use the existing "id" for known characters, null for new ones. Set unchanged fields to null.
- new_threads: plot threads this chapter opens (one sentence each).
- resolved_thread_ids: ids (like "t004") of active threads this chapter clearly resolves. Only use ids that appear in the current state.
- new_world_rules, new_key_items: only genuinely new facts.
- upsert_locations / upsert_factions: new places/factions or changed descriptions.
- current_arc, story_tone, narrative_pov: null unless they changed.
- last_chapter_hook: REQUIRED. How this chapter ends, in one or two sentences.

Current state:
$state

Chapter $chapter text:
$content
