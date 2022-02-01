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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            State::Fair => write!(f, "F"),
            State::Loaded => write!(f, "L"),
        }
    }
}

// This would usually be called a DP-Matrix,
// but I find it easier to think of it as a graph.
// Sort of like a HMM extended by an additional time dimension
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

    // Finds the most probable sequence of states for a given sequence of observations, 
    // using the Viterbi algorithm.
    // Returns a tuple containing the state sequence, and it's probability.
    fn viterbi(&self, observations: &[u8], initial_p: HashMap<State, f64>) -> (Vec<State>, f64) {
        // Initialize an empty Viterbi Graph
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

    // Calculate the probability of a path going through a certain node, for all nodes,
    // using the Forward-Algorithm.
    // This is essentially the Viterbi algorithm without back-references and
    // using sum instead of max.
    // Returns the graph containing all the probabilities, 
    // and a final probability for the entire observed sequence
    fn forward(&self, observations: &[u8], initial_p: HashMap<State, f64>) -> (HashMap<State, Vec<f64>>, f64) {
        let mut graph: HashMap<State, Vec<f64>> = HashMap::new();
        
        for state in &self.states {
            graph.insert(*state, Vec::with_capacity(observations.len()));
        }

        // initialize nodes for the first observation
        // using the provided initial state probabilities
        for (state, prob) in initial_p {
            graph.get_mut(&state)
                .unwrap()
                .push(prob * self.emission_p[&state][&observations[0]]);
        }

        for (i, obs) in observations.iter().enumerate().skip(1) {
            for current_state in &self.states {
                let mut sum = 0.0;

                for prev_state in &self.states {
                    sum += graph[current_state][i-1] * self.transition_p[&(*prev_state, *current_state)];
                }

                graph.get_mut(current_state)
                    .unwrap()
                    .push(sum * self.emission_p[current_state][obs]);
            }
        }

        let total_p: f64 = self.states.iter()
            .map(|s| *graph[s].last().unwrap())
            .sum();

        (graph, total_p)
    }
}

fn main() {
    use State::*;

    let rolls: Vec<u8> = std::fs::read_to_string("data/Casino.txt")
        .unwrap()
        .chars()
        .map(|c| c.to_digit(10).unwrap() as u8)
        .collect();

    assert_eq!(rolls.len(), 300);


    // === Define the Hidden Markov Model ===

    let transition_p = HashMap::from([
        ((Fair,   Fair),   0.95),
        ((Fair,   Loaded), 0.05),
        ((Loaded, Loaded), 0.9 ),
        ((Loaded, Fair),   0.1 ),
    ]);

    let emission_p = HashMap::from([
        ( Fair,   HashMap::from([(1, 1./6.), (2, 1./6.), (3, 1./6.), (4, 1./6.), (5, 1./6.), (6, 1./6.)]) ),
        ( Loaded, HashMap::from([(1, 0.1  ), (2, 0.1  ), (3, 0.1  ), (4, 0.1  ), (5, 0.1  ), (6, 0.5  )]) )
    ]);

    // Let's start fair
    let initial_p = HashMap::from([
        (Fair,   1.0),
        (Loaded, 0.0)
    ]);

    let states = vec![Fair, Loaded];

    let hmm = HiddenMarkovModel { transition_p, emission_p, states };


    // === Run the Viterbi Algorithm, and calculate the TPR and TNR ===
    let (viterbi_path, path_prob) = &hmm.viterbi(&rolls, initial_p);

    println!(" === Viterbi-Algorithm ===");
    println!("Most probable path with probability {:.4e}:", path_prob);
    for s in viterbi_path {
        print!("{s}");
    }
    println!();

    // TPR (sensitivity), TNR (specificity)
    // calculated using the true states that generated the given observations (Durbin)
    let ground_truth: Vec<State> = std::fs::read_to_string("data/ground_truth.txt")
        .unwrap()
        .chars()
        .map(|c| match c {
            'F' => Fair,
            'L' => Loaded,
            _   => panic!(),
        })
        .collect();
    
    let tp = ground_truth
        .iter()
        .zip(viterbi_path.iter())
        .filter(|x| x == &(&Loaded, &Loaded))
        .count();
    
    let tn = ground_truth
        .iter()
        .zip(viterbi_path.iter())
        .filter(|x| x == &(&Fair, &Fair))
        .count();

    let p = ground_truth.iter().filter(|x| **x == Loaded).count();
    let n = ground_truth.iter().filter(|x| **x == Fair  ).count();

    let tpr = tp as f64 / p as f64;
    let tnr = tn as f64 / n as f64;

    println!("Sensitivitvity: {:.1}%, Specificity: {:.2}%", tpr * 100., tnr * 100.);

}
