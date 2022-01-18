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

    let constraint = Constraint::from_updates(&args.updates);

    let answers_string = fs::read_to_string(args.answers).unwrap();
    let answers = answers_string
        .lines()
        .map(|a| a.trim())
        .filter(|a| constraint.permits(a))
        .collect::<HashSet<&str>>();

    let guesses_string = fs::read_to_string(args.guesses).unwrap();
    let guesses = guesses_string
        .lines()
        .map(|line| line.trim())
        .chain(answers.clone())
        .collect::<HashSet<&str>>();

    let expected_information: Vec<f64> = match args.adversarial {
        false => guesses
            .par_iter()
            .map(|guess| _entropy(&answers.iter().map(|answer| _score(guess, answer)).counts()))
            .collect(),
        true => guesses
            .par_iter()
            .map(|guess| {
                _min_surprise(&answers.iter().map(|answer| _score(guess, answer)).counts())
                    .expect("at least one answer")
            })
            .collect(),
    };

    let result: Vec<(&str, u64)> = guesses
        .into_iter()
        .zip(expected_information.iter().map(|e| (1000000.0 * e) as u64))
        .sorted_by_key(|(g, _)| answers.contains(g))
        .sorted_by_key(|(_, e)| *e)
        .collect();
    for (guess, microbits) in result.iter().take(10) {
        println!(
            "{} {}: {}",
            if answers.contains(guess) { '!' } else { ' ' },
            guess,
            *microbits as f64 / 1000000.0
        );
    }
    println!("...");
    for (guess, microbits) in result.iter().rev().take(10).rev() {
        println!(
            "{} {}: {}",
            if answers.contains(guess) { '!' } else { ' ' },
            guess,
            *microbits as f64 / 1000000.0
        );
    }

    Ok(())
}
