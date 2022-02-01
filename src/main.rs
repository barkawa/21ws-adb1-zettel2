use std::collections::HashMap;
use std::{fmt, vec};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum State {
    Fair,
    Loaded,
}

// For printing the States as letters like this:
// Fair   = 'F'
// Loaded = 'L'
impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            State::Fair => write!(f, "F"),
            State::Loaded => write!(f, "L"),
        }
    }
}

struct ViterbiGraphNode {
    back_ref: Option<State>, // back-reference to the most probable previous state
    prob: f64,               // probability of the current state
}

struct HiddenMarkovModel {
    states: Vec<State>,
    transition_p: HashMap<(State, State), f64>,
    emission_p: HashMap<State, HashMap<u8, f64>>,
}

impl HiddenMarkovModel {
    fn viterbi(&self, observations: &[u8], initial_p: HashMap<State, f64>) -> (Vec<State>, f64) {
        let mut graph: HashMap<State, Vec<ViterbiGraphNode>> = HashMap::new();

        for state in &self.states {
            graph.insert(*state, Vec::with_capacity(observations.len()));
        }

        // initialize nodes for the first observation
        // using the provided initial state probabilities
        for (state, prob) in initial_p {
            graph.get_mut(&state).unwrap().push(ViterbiGraphNode {
                back_ref: None, 
                prob: prob * self.emission_p[&state][&observations[0]],
            });
        }

        // Build the Viterbi Graph iteratively
        for (i, obs) in observations.iter().enumerate().skip(1) {
            for current_state in &self.states {
                let mut best = ViterbiGraphNode { back_ref: None, prob: 0.0 };

                for prev_state in &self.states {
                    let p = graph[prev_state][i - 1].prob
                        * self.transition_p[&(*prev_state, *current_state)];
                    if p > best.prob {
                        best.back_ref = Some(*prev_state);
                        best.prob = p;
                    }
                }

                graph.get_mut(&current_state).unwrap().push(ViterbiGraphNode {
                    back_ref: best.back_ref,
                    prob: best.prob * self.emission_p[&current_state][obs],
                });
            }
        }

        // Find the most probable of the last nodes,
        // to get a starting point for the traceback
        let mut start = &ViterbiGraphNode { back_ref: None, prob: 0.0 };
        for state in &self.states {
            let new_best = graph[state].last().unwrap();
            if new_best.prob > start.prob {
                start = new_best;
            }
        }

        // Traceback
        let mut final_path = Vec::new();
        let mut prev = start;
        for i in (1..graph[&self.states[0]].len()).rev() {
            final_path.push(prev.back_ref.unwrap());
            prev = &graph[&prev.back_ref.unwrap()][i-1]
        }

        final_path.reverse();

        (final_path, start.prob)
    }
}

fn main() {
    let rolls = std::fs::read_to_string("data/Casino.txt").unwrap();

    let rolls: Vec<u8> = rolls
        .chars()
        .map(|c| c.to_digit(10).unwrap() as u8)
        .collect();

    assert_eq!(rolls.len(), 300);


    use State::*;

    let transition_p = HashMap::from([
        ((Fair,   Fair),   0.95),
        ((Fair,   Loaded), 0.05),
        ((Loaded, Loaded), 0.9 ),
        ((Loaded, Fair),   0.1 ),
    ]);

    let emission_p = HashMap::from([
        ( 
            Fair, 
            HashMap::from(
                [(1, 1./6.), (2, 1./6.), (3, 1./6.), (4, 1./6.), (5, 1./6.), (6, 1./6.)]
            ) 
        ),
        ( 
            Loaded, 
            HashMap::from(
                [(1, 0.1  ), (2, 0.1  ), (3, 0.1  ), (4, 0.1  ), (5, 0.1  ), (6, 0.5  )]
            ) 
        )
    ]);

    let initial_p = HashMap::from([
        (Fair,   1.0),
        (Loaded, 0.0)
    ]);

    let states = vec![Fair, Loaded];

    let hmm = HiddenMarkovModel { transition_p, emission_p, states };
    let (best_path, best_prob) = hmm.viterbi(&rolls, initial_p);

    println!("Most probable path with probability {:e}:", best_prob);
    for s in best_path {
        print!("{s}");
    }
}
