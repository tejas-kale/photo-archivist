# Model quality rubric

Score each model output per image without looking at the model name. Do not open `*_key.csv` until every row in `*_feedback_template.csv` is complete.

## Files

After running `model_eval.py`, review:

- `eval/model_quality_results_blind.csv` — model outputs with variants `A`, `B`, ... and no model names
- `eval/model_quality_results_feedback_template.csv` — fill this in while reviewing blind outputs
- `eval/model_quality_results_key.csv` — reveal model names only after scoring

Static examples are also provided:

- `eval/model_quality_feedback_template.csv`
- `eval/model_quality_feedback_template.json`

## Ground truth fields

Fill these from your judgement of the actual image, not from either model output:

- `correct_people_count`: integer count of visible people. Use `0` if none. If partly visible people matter, count them if you expect the archive to mention them.
- `correct_rating`: one of `keep`, `review`, `cull`. Use your archive judgement: memorable/useful photo, needs human review, or obvious discard.
- `correct_time_of_day`: short label such as `daytime`, `night`, `sunset`, `indoor`, `unknown`.
- `correct_lighting`: short label such as `natural`, `indoor`, `low`, `harsh`, `mixed`, `flash`.
- `correct_activity`: 2-5 words for what is happening, e.g. `wedding portrait`, `child playing`, `screenshot chat`, `landscape`.

## Score fields

Use `0`, `1`, or `2`.

| Field | 0 | 1 | 2 |
|---|---|---|---|
| `people_count_score` | wrong | close or debatable | correct |
| `description_score` | misleading or useless | broadly useful but vague/incomplete | specific and accurate |
| `activity_score` | wrong | vague or partial | correct |
| `lighting_time_score` | wrong | acceptable but vague | correct |
| `rating_score` | bad keep/review/cull call | debatable | good call |
| `json_score` | failed or invalid | valid but missing important fields | valid and complete |

Use `notes` for anything qualitative: hallucinated objects, missed faces, too verbose, too generic, correctly spotted context, etc.

Decision rule: keep `gemma4:e2b` if `gemma4:e4b` is not clearly better enough to justify the extra heat, latency, and laptop slowdown.
