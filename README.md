# Manual test checklist

- [ ] Dataset class folders contain readable images.
- [ ] Train/validation/test folders have matching class names.
- [ ] Saved checkpoints reload and predict the expected number of classes.
- [ ] Inference uses the same image normalization as training.
- [ ] Metrics are calculated on held-out test data only.
- [ ] LIME returns an overlay for a test image.
- [ ] SHAP output dimensions and class selection are verified.
- [ ] Invalid uploads and missing checkpoints are handled.
- [ ] No invented metrics or clinical claims appear in the report.
