use std::collections::{HashMap, HashSet};
use std::{cmp, fs, iter};

use itertools::Itertools;
use rayon::prelude::*;
use structopt::StructOpt;

fn _score(guess: &str, answer: &str) -> Vec<char> {
    let len = answer.len();
    assert_eq!(len, guess.len());
    assert_eq!(len, answer.len());

    let mut result: Vec<char> = iter::repeat('-').take(len).collect();
    let mut remaining: Vec<char> = answer.chars().collect();

    for (i, (g, a)) in guess.chars().zip(answer.chars()).enumerate() {
        if g == a {
            result[i] = '3';
            remaining[i] = ' ';
        }
    }

    for (i, g) in guess.chars().enumerate() {
        if result[i] == '3' {
            continue;
        }
        match remaining.iter().position(|v| *v == g) {
            Some(j) => {
                result[i] = '2';
                remaining[j] = ' ';
            }
            None => {
                result[i] = '1';
            }
        }
    }

    result
}

fn _entropy<T>(distribution: &HashMap<T, usize>) -> f64 {
    let denominator = distribution.values().sum::<usize>() as f64;
    -distribution
        .values()
        .map(|v| {
            let p = *v as f64 / denominator;
            p * f64::log2(p)
        })
        .sum::<f64>()
}

fn _min_surprise<T>(distribution: &HashMap<T, usize>) -> Option<f64> {
    let numerator = *distribution.values().max()? as f64;
    let denominator = distribution.values().sum::<usize>() as f64;
    Some(-f64::log2(numerator / denominator))
}

struct Constraint {
    permitted: Vec<HashSet<char>>,
    lo: HashMap<char, usize>,
    hi: HashMap<char, usize>,
}

impl Constraint {
    fn new() -> Constraint {
        let mut permitted = Vec::with_capacity(5);
        for _ in 0..5 {
            permitted.push("abcdefghijklmnopqrstuvwxyz".chars().collect());
        }
        Constraint {
            permitted,
            lo: HashMap::new(),
            hi: HashMap::new(),
        }
    }

    fn from_updates(updates: &str) -> Constraint {
        let mut result = Constraint::new();
        if updates.is_empty() {
            return result;
        }
        for update in updates.split(',') {
            let update = update.split(':').collect::<Vec<&str>>();
            let guess = update[0];
            let score = update[1];
            result.update(guess, score);
        }
        result
    }

    fn update(&mut self, guess: &str, score: &str) {
        let mut required = HashSet::new();
        for (i, (g, s)) in guess.chars().zip(score.chars()).enumerate() {
            match s {
                '1' => {
                    self.permitted[i].remove(&g);
                    if !required.contains(&g) {
                        for p in self.permitted.iter_mut() {
                            p.remove(&g);
                        }
                    }
                }
                '2' => {
                    self.permitted[i].remove(&g);
                    required.insert(g);
                }
                '3' => {
                    self.permitted[i].clear();
                    self.permitted[i].insert(g);
                    required.insert(g);
                }
                _ => {
                    panic!("Invalid score {}", score);
                }
            }
        }

        let positive = guess
            .chars()
            .zip(score.chars())
            .filter_map(|(g, s)| match s {
                '2' => Some(g),
                '3' => Some(g),
                _ => None,
            })
            .counts();
        let negative = guess
            .chars()
            .zip(score.chars())
            .filter_map(|(g, s)| match s {
                '1' => Some(g),
                _ => None,
            })
            .counts();
        for (k, v) in positive {
            let lo = self.lo.entry(k).or_insert(0);
            *lo = cmp::max(*lo, v);
            if negative.contains_key(&k) {
                let hi = self.hi.entry(k).or_insert(5);
                *hi = cmp::min(*hi, v);
            }
        }
    }

    fn permits(&self, answer: &str) -> bool {
        for (a, p) in answer.chars().zip(&self.permitted) {
            if !p.contains(&a) {
                return false;
            }
        }

        let counts = answer.chars().counts();
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

struct Bot {
    allowed_guesses: Vec<String>,
    allowed_answers: Vec<String>,
    adversarial: bool,
    cache: HashMap<String, String>,
    num_cache_hit: usize,
}

impl Bot {
    fn new(guesses: Vec<String>, answers: Vec<String>, adversarial: bool) -> Bot {
        Bot {
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

    fn choice(&mut self, updates: &str) -> Option<String> {
        if let Some(result) = self.cache.get(updates) {
            self.num_cache_hit += 1;
            return Some(result.clone());
        }

        let constraint = Constraint::from_updates(updates);

        let plausible_answers: Vec<&String> = self
            .allowed_answers
            .iter()
            .filter(|a| constraint.permits(a))
            .collect();

        let good_guesses: Vec<&String> = if plausible_answers.len() <= 3 {
            plausible_answers.iter().sorted().cloned().collect()
        } else {
            self.allowed_guesses.iter().collect()
        };

        let guesses: Vec<(f64, &&String)> = match self.adversarial {
            false => good_guesses
                .par_iter()
                .map(|guess| {
                    (
                        _entropy(
                            &plausible_answers
                                .iter()
                                .map(|answer| _score(guess, answer))
                                .counts(),
                        ),
                        guess,
                    )
                })
                .collect(),
            true => good_guesses
                .par_iter()
                .map(|guess| {
                    (
                        _min_surprise(
                            &plausible_answers
                                .iter()
                                .map(|answer| _score(guess, answer))
                                .counts(),
                        )
                        .expect("at least one answer"),
                        guess,
                    )
                })
                .collect(),
        };

        let plausible_answers: HashSet<&String> = plausible_answers.iter().cloned().collect();
        let best = guesses
            .iter()
            .max_by(|a, b| match a.0.partial_cmp(&b.0).unwrap() {
                cmp::Ordering::Equal => {
                    match (
                        plausible_answers.contains(a.1),
                        plausible_answers.contains(b.1),
                    ) {
                        (false, true) => cmp::Ordering::Less,
                        (true, false) => cmp::Ordering::Greater,
                        _ => b.1.cmp(a.1),
                    }
                }
                ord => ord,
            })?;
        self.cache.insert(updates.to_string(), (*best.1).clone());
        Some((*best.1).clone())
    }
}

fn _play(bot: &mut Bot, answer: &str) -> String {
    let mut updates = String::new();
    let mut i = 10;
    loop {
        let choice = bot.choice(&updates).unwrap();

        if !updates.is_empty() {
            updates.push(',');
        }
        updates.push_str(&choice);
        updates.push(':');
        updates.extend(_score(&choice, answer));
        if choice == *answer {
            break;
        }
        i -= 1;
        if i == 0 {
            panic!("Ouups {} {}", answer, updates);
        }
    }
    updates
}

fn _histogram(bot: &mut Bot, answers: Vec<String>) -> HashMap<usize, usize> {
    answers
        .iter()
        .map(|answer| {
            let updates = _play(bot, answer);
            let n = updates.matches(',').count() + 1;
            n
        })
        .counts()
}

#[derive(StructOpt)]
pub struct Cli {
    guesses: String,
    answers: String,
    #[structopt(default_value = "")]
    updates: String,
    #[structopt(long)]
    adversarial: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Cli::from_args();

    let mut bot = Bot::new(
        fs::read_to_string(args.guesses)?
            .lines()
            .map(|line: &str| line.trim().into())
            .collect(),
        fs::read_to_string(args.answers)?
            .lines()
            .map(|line: &str| line.trim().into())
            .collect(),
        args.adversarial,
    );
    match bot.choice(&args.updates) {
        Some(guess) => {
            println!("{}", guess);
            Ok(())
        }
        None => Err("Could not find a best guess".into()),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn read_guesses() -> Vec<String> {
        fs::read_to_string("wordlist.txt")
            .unwrap()
            .lines()
            .map(|line: &str| line.trim().into())
            .collect()
    }

    fn read_answers() -> Vec<String> {
        fs::read_to_string("wordlist.txt")
            .unwrap()
            .lines()
            .map(|line: &str| line.trim().into())
            .collect()
    }

    fn read_bot() -> Bot {
        Bot::new(read_guesses(), read_answers(), false)
    }

    #[test]
    fn all_words_can_be_solved() {
        let mut bot = read_bot();
        let answers = read_answers();

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
