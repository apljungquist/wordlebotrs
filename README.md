# WordleBot

A bot that plays wordle games.

## Try it
Try it out by going to [webassembly.sh](https://webassembly.sh/) and entering `wordlebot`.

![screenshot](docs/webassembly.sh_.png)

## How it works

The bot implements a greedy strategy that seeks to learn as much information as possible with every guess.
To do this it chooses the word that maximizes the entropy of the score.
If multiple words are equally good it chooses one that is also a plausible answer.

This strategy finds that optimal starting words include raise, soare and tares for [absurdle](https://qntm.org/files/wordle/index.html), [wordle](https://www.powerlanguage.co.uk/wordle/) and [botfights](https://botfights.io/event/botfights_i) respectively.
