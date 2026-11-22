"""Update slice (the client's example 3: "I live in Beijing", later "I moved to Shenzhen").

For the first 15 tasks of each test log, the model writes one later user line that reverses or changes the
preference, said in passing like the original. The line is inserted near the end of the log (after the
original). The request is unchanged; the right notes now hold the NEW line and should not hold the old one.

  python update_bed.py      writes bed/update.jsonl (target = new line, stale = old line)
"""
import json
import os

import lm
from methods import batched, by_id, numbered

HERE = os.path.dirname(os.path.abspath(__file__))


def gen_changes(prefs):
    p = lambda ch: ("Each item is a preference a user stated in a chat weeks ago. Write one later user message in "
                    "which the user mentions in passing that this preference has changed (reversed or replaced, "
                    "with a short reason), inside a message that is mainly about something else, 30-60 words. "
                    "Reply with a JSON list of objects {\"id\": <id>, \"line\": <message>, \"new_pref\": <the new "
                    "preference in one sentence>}.\n\n" + numbered(ch))

    def parse(j, ch):
        d = {}
        for r in j if isinstance(j, list) else []:
            try:
                d[int(r["id"])] = (r["line"], r["new_pref"])
            except Exception:
                pass
        return [d.get(i) for i, _ in ch]

    return batched(list(enumerate(prefs)), 15, p, parse)


if __name__ == "__main__":
    logs = [json.loads(l) for l in open(os.path.join(HERE, "bed", "prefeval.jsonl"))][4:12]
    out = []
    for L in logs:
        tasks = L["tasks"][:15]
        ch = gen_changes([t["pref"] for t in tasks])
        lines = list(L["lines"])
        new_tasks = []
        # insert all change lines after the last original line, in task order, spread over the final 20 lines
        insert_at = len(lines) - 20
        add = []
        for t, c in zip(tasks, ch):
            if c:
                add.append((t, c))
        for k, (t, (line, newp)) in enumerate(add):
            pos = insert_at + k
            lines.insert(pos, line)
        # recompute indices: lines at >= insert_at shifted
        n_add = len(add)
        for k, (t, (line, newp)) in enumerate(add):
            old = t["target"][0]
            old = old if old < insert_at else old + n_add
            new_tasks.append(dict(q=t["q"], target=[insert_at + k], stale=[old], pref=newp, old_pref=t["pref"],
                                  topic=t["topic"]))
        out.append(dict(log=L["log"] + "u", lines=lines, tasks=new_tasks))
    with open(os.path.join(HERE, "bed", "update.jsonl"), "w") as f:
        for L in out:
            f.write(json.dumps(L) + "\n")
    print(len(out), "logs", sum(len(L["tasks"]) for L in out), "tasks", lm.STATS)
