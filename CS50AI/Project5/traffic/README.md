At first i tried to just copy and paste the code from the lesson but it didn't work,
because after read the code I realize that when create a Conv2D layer I have to input the right shape which is
(30, 30, 3) or (IMG_WIDTH, IMG_HEIGHT, 3) but I didn't notice that.

After that I have issue with the output but this time I don't have a clear instruction (in the log)
about why it happend so after a bit of reading and searching, I find out that the output layer should also have 43 (NUM_CATEGORIES) instead of 10.

I also tried to put more Conv2D layer and max-pooling a couple of times between the first and the last layer and it improve the acuracy a little bit.