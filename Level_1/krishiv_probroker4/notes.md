1. A linear model scores a song as 47. Why is that a problem if you wanted the probability that you will like it, and what does sigmoid do about it?

A. A probility is a chance not a certainity so it is between the values of 0 and 1. A linear model can only multiply and add, so an integar like "47" can't be the chance I like that song.

So sigmoid smashes any number into the 0 to 1 range. Big positive numbers are near 1 while the big negative integars are near 0. For 47 it gives almost exactly 1, meaning "almost certainly yes". A higher
score still means a higher chance.

1. Sigmoid and tanh look similar. When might you prefer one over the other? (There is no single right answer. Reason it out.)

Both have the same steps, but sigmmoid only runs from 0 to 1 and tanh is from -1 to 1. I would use sigmoid in siutations where the answer has a chance/propbability, therefore the value can't be negative. I would use tanh when the answer has two different direction with a neutral middle, therefore the output can be centered on 0.