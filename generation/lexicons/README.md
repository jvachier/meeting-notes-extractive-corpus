# Lexicons

One file per slot (`<slot_lower>.txt`), one value per line. Lines starting with `#` and blank
lines are ignored. Every value is invented or very common: no real people, companies,
products or brands.

Each value has to read naturally wherever a seed uses the slot, so every lexicon has a fixed
form. Seed writers rely on these forms.

| Slot | Form | Example values | Typical seed use |
|---|---|---|---|
| `PERSON` | First name only. Never a common English word ("Will", "June", "Grace") | Amara, Jonas, Priya | "{PERSON} from legal" |
| `TEAM` | Lowercase noun phrase **with** "the" | the platform team | "{TEAM} owns it" |
| `PROJECT` | Capitalised code name, no article | Bluefin | "the {PROJECT} launch" |
| `PRODUCT` | Invented product name, no article | Ledgerloom | "{PRODUCT} users" |
| `CLIENT` | Invented company name, no article | Norvale Logistics | "the call with {CLIENT}" |
| `TOOL` | Lowercase noun phrase **with** "the" | the CI pipeline | "{TOOL} is down" |
| `DATE_REL` | Reads after "by", "before", "until" | next Tuesday, end of sprint | "by {DATE_REL}" |
| `DATE_ABS` | Month and day, reads after "by", "on" | March 14 | "on {DATE_ABS}" |
| `NUMBER` | A count ≥ 2, words or digits, may be hedged | twelve, about 200 | "{NUMBER} tickets" |
| `PERCENT` | Spoken percentage | 12 percent | "up {PERCENT}" |
| `MONEY` | Spoken amount with currency | $40,000, 60 thousand euros | "costs {MONEY}" |
| `METRIC` | Lowercase noun phrase, **no** article | churn, p95 latency | "our {METRIC} dropped" |

Values that start with a lowercase letter (`TEAM`, `TOOL`, `DATE_REL`, `METRIC`, …) may land
at the start of a sentence. The composer capitalises the first character of every rendered
sentence; that is rendering only and never touches a label.
