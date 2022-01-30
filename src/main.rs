use std::fmt;
use std::ops::{Index, IndexMut};



#[derive(Clone, Copy)]
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
            State::Loaded => write!(f, "L")
        }
    }
}



struct TransitionProbability {
    matrix: [[f64; 2]; 2],
}

// For indexing like this: transition_prob[(Fair, Loaded)]
impl Index<(State, State)> for TransitionProbability {
    type Output = f64;

    fn index(&self, index: (State, State)) -> &Self::Output {
        let (from, to) = index;

        let i = match from { State::Fair => 0, State::Loaded => 1 };
        let j = match to   { State::Fair => 0, State::Loaded => 1 };

        &self.matrix[i][j]
    }

}



struct EmissionProbability {
    fair: [f64; 7],
    loaded: [f64; 7],
}

// For indexing like this: emission_prob[Loaded][6]
impl Index<State> for EmissionProbability {
    type Output = [f64; 7];

    fn index(&self, index: State) -> &Self::Output {
        match index {
            State::Fair => &self.fair,
            State::Loaded => &self.loaded,
        }
    }
}



struct ViterbiGraphNode {
    back_ref: Option<State>, // back-reference to the most probable previous state
    prob: f64,               // probability of the current state
}

struct ViterbiGraph {
    graph: [Vec<ViterbiGraphNode>; 2]
}

// Indexing again, same as before
impl Index<State> for ViterbiGraph {
    type Output = Vec<ViterbiGraphNode>;

    fn index(&self, index: State) -> &Self::Output {
        match index {
            State::Fair => &self.graph[0],
            State::Loaded => &self.graph[1],
        }
    }
}

// This time we also need mutable references
impl IndexMut<State> for ViterbiGraph {
    fn index_mut(&mut self, index: State) -> &mut Self::Output {
        match index {
            State::Fair => &mut self.graph[0],
            State::Loaded => &mut self.graph[1],
        }
    }
}

impl ViterbiGraph {
    fn new() -> Self {
        Self { graph: [Vec::new(), Vec::new()] } 
   }
}



struct InitialProbabilities {
    fair: f64,
    loaded: f64,
}



struct HiddenMarkovModel {
    transition_p: TransitionProbability,
    emission_p: EmissionProbability,
}

impl HiddenMarkovModel {

    // This is the important code:
    fn find_most_probable_path(&self, observations: Vec<u8>, initial_p: InitialProbabilities) -> Vec<State> {
        use State::*;

        let mut graph = ViterbiGraph::new();

        // initialize nodes for the first observation
        // using the provided initial state probabilities
        
        graph[Fair].push(
            ViterbiGraphNode {
                back_ref: None,
                prob: initial_p.fair * self.emission_p[Fair][observations[0] as usize] 
            }
        );

        graph[Loaded].push( 
            ViterbiGraphNode {
                back_ref: None,
                prob: initial_p.loaded * self.emission_p[Fair][observations[0] as usize] 
            }
        );

        // Build the Viterbi Tree iteratively
        for (i, obs) in observations.iter().enumerate().skip(1) {
            for current_state in [Fair, Loaded] {
                let mut best = ViterbiGraphNode { back_ref: None, prob: 0.0 };

                for prev_state in [Fair, Loaded] {
                    let p = graph[current_state][i-1].prob * self.transition_p[(prev_state, current_state)];
                    if p > best.prob {
                        best.back_ref = Some(current_state);
                        best.prob = p;
                    }
                }

                graph[current_state].push( 
                    ViterbiGraphNode {
                        back_ref: best.back_ref,
                        prob: best.prob * self.emission_p[current_state][*obs as usize],
                    }
                );
            }
        }

        // traceback
        todo!()
    }

}



fn main() {
    let rolls = std::fs::read_to_string("data/Casino.txt").unwrap();

    let rolls: Vec<u8> = rolls.chars()
        .map(|c| c.to_digit(10).unwrap() as u8)
        .collect();
    
    assert_eq!(rolls.len(), 300);


    // The matrix is structured like this:
    //       F  L
    //   F [[_, _],
    //   L  [_, _]]
    let transition_p = TransitionProbability { matrix: [[0.95, 0.05], [0.1, 0.9]] };

    // Index = emitted value
    // Arrays start at zero, dice don't, that's why the virst value is 0.0
    let emission_p = EmissionProbability {
        fair: [0., 1./6., 1./6., 1./6., 1./6., 1./6., 1./6.],
        loaded: [0., 0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    };

    let initial_p = InitialProbabilities { fair: 1., loaded: 0. };

    let hmm = HiddenMarkovModel { transition_p, emission_p };
    let result = hmm.find_most_probable_path(rolls, initial_p);


}