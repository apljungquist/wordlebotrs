use rayon::prelude::*;
use std::collections::HashMap;
use std::{fs, iter};

use itertools::Itertools;
use structopt::StructOpt;

fn _score(guess: &[char], answer: &[char]) -> Vec<char> {
    let len = answer.len();
    assert_eq!(len, guess.len());
    assert_eq!(len, answer.len());

    let mut result: Vec<char> = iter::repeat('-').take(len).collect();
    let mut remaining: Vec<char> = answer.to_vec();

    for i in 0..len {
        if guess[i] == answer[i] {
            result[i] = '3';
            remaining[i] = ' ';
        }
    }

    for i in 0..len {
        if result[i] == '3' {
            continue;
        }
        match remaining.iter().position(|v| v == &guess[i]) {
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

#[derive(StructOpt)]
pub struct Cli {
    guesses: String,
    answers: String,
    #[structopt(long)]
    adversarial: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let args = Cli::from_args();

    let guesses = fs::read_to_string(args.guesses)
        .unwrap()
        .lines()
        .map(|line| line.trim().chars().collect())
        .collect::<Vec<Vec<char>>>();
    let answers = fs::read_to_string(args.answers)
        .unwrap()
        .lines()
        .map(|line| line.trim().chars().collect())
        .collect::<Vec<Vec<char>>>();

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

    let result: Vec<(Vec<char>, u64)> = guesses
        .into_iter()
        .zip(expected_information.iter().map(|e| (1000000.0 * e) as u64))
        .sorted_by_key(|(_, e)| *e)
        .collect();
    for (guess, microbits) in result.iter().take(10) {
        println!(
            "{}: {}",
            String::from_iter(guess),
            *microbits as f64 / 1000000.0
        );
    }
    println!("...");
    for (guess, microbits) in result.iter().rev().take(10).rev() {
        println!(
            "{}: {}",
            String::from_iter(guess),
            *microbits as f64 / 1000000.0
        );
    }

    Ok(())
}
