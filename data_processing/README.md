# Data processing and formats

The eRisk collections are not redistributed with this code. This document
describes what `python -m data_processing.build_graph_pkl` reads and writes, so
a collection can be converted from its original distribution format.

Sentence-encoder checkpoints and preprocessing defaults are configured in
`config/data_processing.yaml`. Command-line arguments can override those
defaults for an individual collection.

## Input — `--mode conversation`

One JSON file per user, named `<user_id>.json`, inside `--raw-dir`. Each element
is one round (one discussion thread), **in the order the evaluation server reveals
them**:

```json
[
  {
    "submission": {
      "submission_id": "t3_abc123",
      "user_id": "subject1234",
      "title": "thread title",
      "body": "thread body",
      "created_utc": "2024-03-01 18:22:41",
      "target": "True"
    },
    "comments": [
      {
        "comment_id": "t1_def456",
        "parent_id": "t3_abc123",
        "user_id": "other_user",
        "body": "reply text",
        "created_utc": "2024-03-01 19:03:12",
        "target": "False"
      }
    ]
  }
]
```

Notes:

* `target` marks nodes written by the subject of the timeline. A node also counts
  as a target node when its `user_id` equals the filename stem.
* `parent_id` gives the reply structure. A comment with no parent is attached to
  the submission.
* `created_utc` is parsed by `pandas.to_datetime`; any format it accepts works.
  Unparseable timestamps become 0.
* `[deleted]` / `[removed]` bodies are allowed; the node keeps its graph position.
  Authors that are missing or deleted receive a per-node anonymous identity so
  their memories never merge with anyone else's.

## Input — `--mode posts`

For collections that only distribute target-authored posts. One JSON file per
user, named `<user_id>.json`:

```json
[
  {"post_id": "p1", "date": "2017-05-02 11:04:00", "text": "post text"},
  {"post_id": "p2", "date": "2017-05-09 08:31:17", "text": "another post"}
]
```

Each relevant post becomes a one-round conversation rooted at that post. Because
these collections carry no conversational context, replies are generated with an
LLM and supplied through `--synthetic-comments` as a single JSON file:

```json
{
  "2017_subject42": {
    "p1": ["generated reply", "another generated reply"],
    "p2": ["generated reply"]
  }
}
```

Each generated reply becomes a non-target child of its post, authored by a
distinct interlocutor and timestamped one minute after the post. Posts without
replies still train; attention simply sees no context that round.

Use `--user-prefix` when the golden truth is keyed with a collection prefix
(`--user-prefix 2017_` turns `subject42.json` into user id `2017_subject42`).

## Labels

Either the eRisk golden-truth text format, one user per line:

```
2017_subject42 1
2017_subject43 0
```

or a JSON object with the same mapping. Accepted label spellings are
`1/0`, `true/false`, `yes/no`, `positive/negative`, `pos/neg`, `control`.

## Output — timeline pickle

A dict mapping `user_id` to a list of rounds sorted by `conv_id`:

```python
{
    "subject1234": [
        {
            "conv_id": 0,
            "context_only": False,  # True when the round has no target post
            "min_ts": 1709316161000000000,
            "events": [
                {
                    "node_id": "t3_abc123",
                    "parent": None,
                    "user_id": "subject1234",
                    "rel_hidden": tensor,  # [n_sentences, 1024]
                    "emo_sev": tensor,  # [n_sentences, 28] = 7 emotion + 21 severity
                    "ts_raw": 1709316161000000000,
                    "is_target": True,
                    "local_neighbours": [1, 2],  # indices within this round
                }
            ],
        }
    ]
}
```

`local_neighbours` lists the non-target nodes one reply hop from a target node
(its parent and its direct children) — the edges attention reads. Non-target
events always carry an empty list.

`emo_sev` concatenates emotion probabilities with symptom-severity scores that
have been zeroed for sentences below `--relevance-threshold`, so irrelevant
sentences contribute no symptom signal while still occupying their position.

Only rounds up to the last one containing a relevant target post are kept: later
rounds can no longer change any decision. Rounds with interlocutor activity but no
target post are retained as `context_only` — they emit no prediction but do
advance interlocutor memory.

These pickles hold dense sentence embeddings and are large (tens of GB for a full
collection). Build them once and reuse them across runs.
