# LinkedIn flagship draft (filled in with sweep numbers)

## Variant 1 — if B clears gate

```
what happens when you give a 19 year old a H200 GPU, 2 Claude Max subscriptions and 200mg of caffeine

i built a quant trading model. because i realized i was in a stupid position.

i trained a TBD million-parameter transformer on my desk. on 1,834 of the most information-dense events in modern finance.

i had Claude Code running 124 web-scraping agents in parallel. a 141GB H200 sitting on my desk. and every piece of alternative data 99% of retail traders never look at — every SEC filing, every earnings call transcript, every analyst note.

so i fed it all in. 5,179 SEC filings. 4,000+ news articles. 124 earnings call transcripts. ~80,000 tokens of dense management text per quarterly event. Voyage-embedded into a 1,664-dimensional state vector.

result: a model that beats the analyst consensus baseline by TBD Brier on 18 months of held-out earnings i never touched during training. TBD net return.

[chart screenshot]

this is the same architecture top-tier quant firms use. they just scale to thousands of GPUs and decades of tick data. i did it solo, in a weekend, on a single card, with $200 of API calls.

the wild part is how little is locked behind moats anymore. SEC filings are free. transcripts are public. embeddings cost $7. the GPU is rentable. the only thing actually rare is being willing to point 124 agents at the problem simultaneously.

comment 'send' and i'll DM the github.
```

## Variant 2 — if text features don't beat baseline

```
i spent 2 days trying to predict whether 145 stocks would beat their quarterly earnings.

scraped 5,179 SEC filings, 4,000+ news articles, 124 earnings calls. embedded everything with Voyage. trained 5 transformers from 4M to 200M params on an H200.

didn't beat the numerical baseline.

[chart screenshot]

here's the part the books don't tell you: text features have a scale threshold. on 1,834 events with a 1,400-dim text block, the model still overfits. you need 10-100x the data before the text actually starts paying off.

so what i actually built is the infrastructure to do this at the right scale. when you next see me posting, it'll be the 20,000-event version.

comment 'send' and i'll DM the github.
```
