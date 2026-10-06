# Notes

## A linear model scores a song as 47. Why is that a problem?

A probability is a chance, so it has to sit between 0 and 1. A linear model only
multiplies and adds, so it can give any number: 47, -3, a million. "47" cannot be
the chance that I like a song.

Sigmoid squashes any number into the 0 to 1 range. Big positive scores land near
1, big negative scores near 0, and a score of 0 lands on exactly 0.5, a coin
flip. For 47 it gives almost exactly 1, meaning "almost certainly yes". A higher
score still means a higher chance.

## Sigmoid or tanh?

Both have the same S shape, but sigmoid runs from 0 to 1 and tanh from -1 to 1.

I would use sigmoid when the answer is a chance, like "will I like this song?",
because a chance cannot be negative.

I would use tanh when the answer has two directions with a neutral middle, like
"dislike, don't care, like", because its output is centred on 0.
