## Summary

<!-- What does this PR change and why? -->

## Type of change

- [ ] New model architecture
- [ ] Data pipeline change (new source, new feature, preprocessing)
- [ ] Hyperparameter / training loop change
- [ ] Refactor (no functional change)
- [ ] Bug fix
- [ ] Documentation

## Validation loss

<!-- If training was affected, report val loss before → after. Even a rough number is fine. -->

| Model | Before | After |
|---|---|---|
| | | |

## Checklist

- [ ] No `scaler.pkl`, `best_model.pt`, or other `.pt`/`.pkl` artefacts committed
- [ ] Train/test split remains chronological (no shuffle before split)
- [ ] Scaler is fit on the train portion only
- [ ] Target column order unchanged (or inverse-transform logic updated accordingly)
- [ ] CLAUDE.md / AGENTS.md updated if the pipeline or conventions changed
