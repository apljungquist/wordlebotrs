use std::hash::Hash;
use std::{cmp, fs, iter};

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use rayon::prelude::*;
use structopt::StructOpt;

fn _score<const N: usize>(guess: &Word<N>, answer: &Word<N>) -> Score<N> {
    let len = answer.len();
    assert_eq!(len, guess.len());
    assert_eq!(len, answer.len());

    let mut result: Score<N> = [0; N];
    let mut remaining = *answer;

    for (i, (g, a)) in guess.iter().zip(answer.iter()).enumerate() {
        if g == a {
            result[i] = 3;
            remaining[i] = ' ';
        }
    }

    for (i, g) in guess.iter().enumerate() {
        if result[i] == 3 {
            continue;
        }
        match remaining.iter().position(|v| v == g) {
            Some(j) => {
                result[i] = 2;
                remaining[j] = ' ';
            }
            None => {
                result[i] = 1;
            }
        }
    }

    result
}
impl<T: ?Sized> MyItertools for T where T: Iterator {}
trait MyItertools: Iterator {
    fn fast_counts(self) -> HashMap<Self::Item, usize>
    where
        Self: Sized,
        Self::Item: Eq + Hash,
    {
        let mut result = HashMap::new();
        self.for_each(|item| *result.entry(item).or_insert(0) += 1);
        result
    }
}

fn _entropy<T>(distribution: &HashMap<T, usize>) -> u64 {
    let denominator = distribution.values().sum::<usize>() as f64;
    let result = distribution
        .values()
        .map(|v| {
            let p = *v as f64 / denominator;
            p * -f64::log2(p)
        })
        .sum::<f64>();
    (1_000_000_000.0 * result) as u64
}

fn _min_surprise<T>(distribution: &HashMap<T, usize>) -> Option<u64> {
    let numerator = *distribution.values().max()? as f64;
    let denominator = distribution.values().sum::<usize>() as f64;
    Some((1_000_000_000.0 * -f64::log2(numerator / denominator)) as u64)
}

#[derive(Debug)]
struct Constraint<const N: usize> {
    permitted: [HashSet<char>; N],
    lo: HashMap<char, usize>,
    hi: HashMap<char, usize>,
}

impl<const N: usize> Constraint<N> {
    fn new() -> Self {
        let permitted = iter::repeat(
            "abcdefghijklmnopqrstuvwxyz"
                .chars()
                .collect::<HashSet<char>>(),
        )
        .take(N)
        .collect::<Vec<HashSet<char>>>()
        .try_into()
        .unwrap();
        Self {
            permitted,
            lo: HashMap::new(),
            hi: HashMap::new(),
        }
    }

    fn from_clues(clues: &[(Word<N>, Score<N>)]) -> Self {
        let mut result = Self::new();
        if clues.is_empty() {
            return result;
        }
        for (guess, score) in clues {
            result.update(guess, score);
        }
        result
    }

    fn update(&mut self, guess: &Word<N>, score: &Score<N>) {
        let mut required = HashSet::new();
        for (i, (g, s)) in guess.iter().zip(score.iter()).enumerate() {
            match s {
                1 => {
                    self.permitted[i].remove(g);
                    if !required.contains(&g) {
                        for p in self.permitted.iter_mut() {
                            p.remove(g);
                        }
                    }
                }
                2 => {
                    self.permitted[i].remove(g);
                    required.insert(g);
                }
                3 => {
                    self.permitted[i].clear();
                    self.permitted[i].insert(*g);
                    required.insert(g);
                }
                _ => {
                    panic!("Invalid score {:?}", score);
                }
            }
        }

        let positive = guess
            .iter()
            .zip(score.iter())
            .filter_map(|(g, s)| match s {
                2 => Some(g),
                3 => Some(g),
                _ => None,
            })
            .fast_counts();
        let negative = guess
            .iter()
            .zip(score.iter())
            .filter_map(|(g, s)| match s {
                1 => Some(g),
                _ => None,
            })
            .fast_counts();
        for (k, v) in positive {
            let lo = self.lo.entry(*k).or_insert(0);
            *lo = cmp::max(*lo, v);
            if negative.contains_key(&k) {
                let hi = self.hi.entry(*k).or_insert(5);
                *hi = cmp::min(*hi, v);
            }
        }
    }

    fn permits(&self, answer: &Word<N>) -> bool {
        for (a, p) in answer.iter().zip(&self.permitted) {
            if !p.contains(a) {
                return false;
            }
        }

        let counts = answer.iter().fast_counts();
        for (k, lo) in self.lo.iter() {
            match counts.get(k) {
                Some(v) if lo <= v => continue,
                _ => return false,
            }
        }
        for (k, hi) in self.hi.iter() {
            match counts.get(k) {
                Some(v) if v <= hi => continue,
                _ => return false,
            }
        }

        true
    }
}

type Word<const N: usize> = [char; N];
type Score<const N: usize> = [u8; N];

struct Bot<const N: usize> {
    allowed_guesses: Vec<Word<N>>,
    allowed_answers: Vec<Word<N>>,
    adversarial: bool,
    cache: HashMap<Vec<(Word<N>, Score<N>)>, Word<N>>,
    num_cache_hit: usize,
}

impl<const N: usize> Bot<N> {
    fn new(guesses: Vec<Word<N>>, answers: Vec<Word<N>>, adversarial: bool) -> Self {
        Self {
            allowed_guesses: guesses
                .into_iter()
                .chain(answers.iter().cloned())
                .unique()
                .collect(),
            allowed_answers: answers,
            adversarial,
            cache: HashMap::new(),
            num_cache_hit: 0,
        }
    }

    fn choice(&mut self, clues: &[(Word<N>, Score<N>)]) -> Option<Word<N>> {
        if let Some(result) = self.cache.get(clues) {
            self.num_cache_hit += 1;
            return Some(*result);
        }

        let constraint = Constraint::from_clues(clues);

        let plausible_answers: Vec<&Word<N>> = self
            .allowed_answers
            .iter()
            .filter(|a| constraint.permits(a))
            .collect();
        // Before the ordering accounted for plausible answers this reduced the number of guesses.
        // Now it is only an optimization and provides a 2-3x speedup.
        let good_guesses: Vec<&Word<N>> = if plausible_answers.len() <= 3 {
            plausible_answers.clone()
        } else {
            self.allowed_guesses.iter().collect()
        };
        let guesses: Vec<(u64, &Word<N>)> = match self.adversarial {
            false => good_guesses
                .into_par_iter()
                .map(|guess| {
                    (
                        _entropy(
                            &plausible_answers
                                .iter()
                                .map(|answer| _score(guess, answer))
                                .fast_counts(),
                        ),
                        guess,
                    )
                })
                .collect(),
            true => good_guesses
                .into_par_iter()
                .map(|guess| {
                    (
                        _min_surprise(
                            &plausible_answers
                                .iter()
                                .map(|answer| _score(guess, answer))
                                .fast_counts(),
                        )
                        .expect("at least one answer"),
                        guess,
                    )
                })
                .collect(),
        };

        let plausible_answers: HashSet<&Word<N>> = plausible_answers.into_iter().collect();
        let best = guesses
            .into_iter()
            .max_by_key(|(info, guess)| (*info, plausible_answers.contains(guess), *guess))?;
        self.cache.insert(clues.to_vec(), *best.1);
        Some(*(best.1))
    }
}

fn _play<const N: usize>(bot: &mut Bot<N>, answer: &Word<N>) -> Vec<(Word<N>, Score<N>)> {
    let mut clues: Vec<(Word<N>, Score<N>)> = Vec::new();
    loop {
        let choice = bot.choice(&clues).unwrap();
        let score = _score(&choice, answer);
        clues.push((choice, score));
        if choice == *answer {
            break;
        }
    }
    clues
}

fn _histogram<const N: usize>(bot: &mut Bot<N>, answers: Vec<Word<N>>) -> HashMap<usize, usize> {
    answers
        .iter()
        .map(|answer| {
            let clues = _play(bot, answer);

            clues.len()
        })
        .fast_counts()
        .into_iter()
        .collect()
}

#[derive(StructOpt)]
pub struct Cli {
    wordlist: String,
    #[structopt(default_value = "")]
    clues: String,
    #[structopt(long)]
    adversarial: bool,
}

fn _word<const N: usize>(line: &str) -> Word<N> {
    line.trim()
        .chars()
        .collect::<Vec<char>>()
        .as_slice()
        .try_into()
        .unwrap()
}

fn _answers<const N: usize>(text: &str) -> Vec<Word<N>> {
    text.lines()
        .take_while(|line| !line.is_empty())
        .map(_word)
        .collect()
}

fn _guesses<const N: usize>(text: &str) -> Vec<Word<N>> {
    text.lines()
        .filter(|line| !line.is_empty())
        .map(_word)
        .collect()
}

fn _parse_clues<const N: usize>(clues: &str) -> Vec<(Word<N>, Score<N>)> {
    let mut result = Vec::new();
    if clues.is_empty() {
        return result;
    }
    for clue in clues.split(',') {
        let clue = clue.split(':').collect::<Vec<&str>>();
        let guess = _word(clue[0]);
        let score = clue[1]
            .chars()
            .map(|c| match c {
                '1' => 1,
                '2' => 2,
                '3' => 3,
                _ => panic!("Expected [1-3] but found {}", c),
            })
            .collect::<Vec<u8>>()
            .as_slice()
            .try_into()
            .unwrap();
        result.push((guess, score));
    }
    result
}

fn _read_bot<const N: usize>(args: &Cli) -> Bot<N> {
    let wordlist = fs::read_to_string(&args.wordlist).unwrap();
    Bot::new(_guesses(&wordlist), _answers(&wordlist), args.adversarial)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Cli::from_args();
    let mut bot: Bot<5> = _read_bot(&args);
    match bot.choice(&_parse_clues(&args.clues)) {
        Some(guess) => {
            println!("{}", guess.iter().join(""));
            Ok(())
        }
        None => Err("Could not find a best guess".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_words_can_be_solved() {
        let mut bot: Bot<5> = _read_bot(&Cli {
            wordlist: "wordlist.txt".into(),
            clues: "".into(),
            adversarial: false,
        });
        let answers = bot.allowed_answers.clone();

        let histogram = _histogram(&mut bot, answers);
        let num_answer = histogram.values().sum::<usize>();
        let num_guess = histogram.iter().map(|(k, v)| k * v).sum::<usize>();
        let max_guess = *histogram.keys().max().unwrap();
        println!(
            "Histogram: {:?}",
            histogram
                .into_iter()
                .sorted()
                .collect::<Vec<(usize, usize)>>()
        );
        println!("Num answers: {}", num_answer);
        println!("Num guesses: {}", num_guess);
        println!("Avg guesses: {}", num_guess as f64 / num_answer as f64);
        println!("Max guesses: {}", max_guess);
        println!(
            "Cache hits: {}, misses: {}",
            bot.num_cache_hit,
            bot.cache.len(),
        );
    }
}
