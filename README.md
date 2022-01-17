# WordleBot

A bot that plays wordle[^1] games[^2].

## How it works

The bot divides the game into two phases, exploration and exploitation.
During both phases it employs a greedy strategy.

During exploration it seeks to learn as much information as possible.
It therefore guesses the word for which it expects the score to contain the most information.
In case of the honest game this is the entropy of the random variable that is the score.
In case of the adversarial game this is the minimum surprisal of seeing the score.

When there are 3 or fewer possible answers left the bot moves to the exploitation phase.
During the exploitation phase it will choose a guess as before but only among the possible answers remaining. 

This strategy finds that the optimal starting word is serai, soare and tares for [absurdle](https://qntm.org/files/wordle/index.html), [wordle](https://www.powerlanguage.co.uk/wordle/) and [botfights](https://botfights.io/event/botfights_i) respectively.

[^1]: Implements only the first round i.e. finding an optimal starting word.
[^2]: Implements strategies for both adversarial and honest games but only for 5 letter words.
