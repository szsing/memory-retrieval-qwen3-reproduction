# Unified native readout sensitivity, 2026-11-22

This is a corrective controlled readout experiment. The older 400-character-per-unit adaptation discarded structured RootMem personal-state fields and A-Mem linked memory units. Preserve both versions; never select a protocol based on the resulting score.

- Same local Qwen3-8B BF16/no-thinking endpoint, temperature 0, answer prompt and output budget 2048.
- InMind: 240 background lines plus only the current target fact. Primary held-out set = first 61 unseen-domain tasks; all 125 includes development domains. This is not the official conversation timeline protocol.
- Original batched HyDE/properties/gate outputs are frozen from strict241-queryfrozen-v2. Retrieval is unchanged; exact reconstruction of all earlier clipped inputs is asserted before scoring.
- Whole native memory units are packed in retrieval order into a maximum 4096-token context. A-Mem may retain linked neighbors; RootMem retains skill JSON fields. Raw methods retain their selected six lines.
- Mem0 reads the same persisted case store; A-Mem replays the cached target append and query keywords from the same prefix. A-Mem asserts zero new model responses during replay.
- RootMem is an explicit symlink to root-native4096-v1, using the same actual native retrieval and whole-unit budget. It is not a new independent replication.
- No-context and oracle references are unchanged; all 125 oracle fact strings are at most 131 characters, so their reader inputs are identical.
- Native memory systems use official source components with a local-model/BGE adaptation. They are not exact paper configurations or a reproduction of the friend's unprovided RootMem script.
- Official InMind answer-only and application judge prompt texts are used for rescoring, but the judge is still local Qwen, not an official leaderboard judge. Model-based judgments require independent review before publication.

Prior source-clipped and diagnostic single-query outputs remain immutable in their original directories. Source code and cache hashes are recorded alongside the older protocol manifests.

