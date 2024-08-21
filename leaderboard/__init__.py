SUPPORTED_METRICS = [
    "avg_mcauroc",  # for classification tasks
    "exact_match",  # for QA tasks
    "acc",  # for multichoice tasks
    "rouge_raw_r2_mid_f_without_bootstrap", # for summarization tasks
    "rouge_raw_r2_mid_f",  # for summarization tasks, older metric version for back compatibility
    "word_perplexity",  # for language modeling tasks
]
EXTRA_INFO_RELEASE_KEYS = [
    'filtered_resps',
    'doc_id',
]
