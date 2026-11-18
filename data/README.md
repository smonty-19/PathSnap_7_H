# `data/` (NOT committed, git-ignored)

Put the provided dataset here on your own laptop:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/...
data/processed/train.parquet          <- built by Monty, shared via Drive/USB
data/processed/student_test.parquet
```

Evaluator trips go on Laptop 1 only, on the evaluation day (point `PATHSNAP_EVAL_DIR` at them). **Never train or tune on them.**
