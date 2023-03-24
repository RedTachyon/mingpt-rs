use std::any::Any;
use std::fs;
use std::iter::FromIterator;
use std::collections::HashMap;
use std::ops::Index;
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;

use tch::{Device, IndexOp, Kind, nn, Tensor};
use tch::nn::{Module, ModuleT, OptimizerConfig};
use tch;


use crate::models::{LanguageModel, AttnLanguageModel};

mod models;

static BLOCK_SIZE: i64 = 32;
static BATCH_SIZE: i64 = 16;

enum Split {
    Train,
    Test,
}

fn read_file(path: &str) -> String {
    let contents = fs::read_to_string(path).expect("Something went wrong reading the file");
    contents
}

fn get_charset(text: &str) -> Vec<char> {
    let mut charset: Vec<char> = Vec::new();
    for c in text.chars() {
        if !charset.contains(&c) {
            charset.push(c);
        }
    }
    charset.sort();
    charset
}

fn get_token_maps(chars: Vec<char>) -> (HashMap<char, i32>, HashMap<i32, char>) {
    let mut char_to_token: HashMap<char, i32> = HashMap::new();
    let mut token_to_char: HashMap<i32, char> = HashMap::new();
    for (i, c) in chars.iter().enumerate() {
        char_to_token.insert(*c, i as i32);
        token_to_char.insert(i as i32, *c);
    }
    (char_to_token, token_to_char)
}

fn encode(text: &str, stoi: &HashMap<char, i32>) -> Vec<i32> {
    let mut encoded: Vec<i32> = Vec::new();
    for c in text.chars() {
        encoded.push(*stoi.get(&c).unwrap());
    }
    encoded
}

fn decode(encoded: &Vec<i32>, itos: &HashMap<i32, char>) -> String {
    let mut decoded: Vec<char> = Vec::new();
    for i in encoded {
        decoded.push(*itos.get(i).unwrap());
    }
    String::from_iter(decoded)
}

fn get_batch(data: &Tensor) -> (Tensor, Tensor) {
    let mut rng = thread_rng();
    // let range = Uniform(0, data.size()[0] - BLOCK_SIZE);
    let max_range = data.size()[0] - BLOCK_SIZE;

    let ix = (0..BATCH_SIZE).map(|_| rng.gen_range(0..max_range)).collect::<Vec<i64>>();

    let xs = ix
        .iter()
        .map(|i| data.i((*i..*i+BLOCK_SIZE)))
        .collect::<Vec<Tensor>>();
    let ys = ix
        .iter()
        .map(|i| data.i((*i+1..*i+BLOCK_SIZE+1)))
        .collect::<Vec<Tensor>>();


    let x = Tensor::stack(&xs, 0);
    let y = Tensor::stack(&ys, 0);

    (x, y)

}


fn estimate_loss(model: &dyn LanguageModel, data: &Tensor, eval_iters: i32) -> f32 {
    // let mut losses = Vec::with_capacity(eval_iters as usize);
    let mut result = 0.0;
    tch::no_grad(|| {
        for _ in 0..eval_iters {
            let (xb, yb) = get_batch(&data);
            let loss = model.loss(&xb, &yb);
            let loss_value = f32::from(&loss);
            // losses.push(loss_value);
            result += loss_value;
        }
    });

    // losses.iter().sum::<f32>() / eval_iters as f32
    result / eval_iters as f32
}

fn main() {
    tch::manual_seed(1337);
    let text = read_file("data/input.txt");
    let chars = get_charset(&text);

    let (stoi, itos) = get_token_maps(chars.clone());


    let data = Tensor::of_slice(&encode(&text, &stoi));

    let n = (0.9 * data.size()[0] as f32) as i64;

    let train_data = data.i((0..n));
    let test_data = data.i((n..data.size()[0]));

    let (x, y) = get_batch(&train_data);

    // for b in 0..BATCH_SIZE {
    //     for t in 0..BLOCK_SIZE {
    //         let context: Vec<i32> = Vec::from(&x.i((b, 0..t+1)));
    //         let target: Vec<i32> = Vec::from(&y.i((b, t)));
    //         println!("when input is {:?} the target is {:?}", context, target);
    //         // println!("when input is {} the target is {}", itos.get(&(x[b][t].item::<i32>() as i32)).unwrap(), itos.get(&(y[b][t].item::<i32>() as i32)).unwrap()
    //     }
    // }
    let vs = nn::VarStore::new(tch::Device::Cpu);

    let vocab_size = chars.len() as i64;
    println!("vocab_size: {}", vocab_size);
    let n_embd = 64;
    let n_head = 4;
    let n_layer = 4;
    let dropout = 0.0;
    let batch_size = BATCH_SIZE;
    let block_size = BLOCK_SIZE;
    let max_iters = 5000;
    let eval_interval = 100;
    let lr = 1e-3;
    let eval_iters = 200;


    let mut model = AttnLanguageModel::new(&vs.root(),
                                           vocab_size,
                                           n_embd,
                                           n_head,
                                           n_embd / n_head,
                                            block_size,
                                    4*n_embd,
                                            n_layer,
                                            dropout);


    let mut optimizer = nn::AdamW::default().build(&vs, lr).unwrap();


    for step in 0..5000 {
        let (xb, yb) = get_batch(&train_data);
        let loss = model.loss(&xb, &yb);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if step % 100 == 0 {
            let loss_value = f32::from(&loss);
            println!("step: {}, loss: {}", step, loss_value);
        }

    }

    let context = Tensor::zeros(&[1, 1], (Kind::Int64, Device::Cpu));
    let out = model.generate(&context, 1000);
    let foo = Vec::<i32>::from(&out);
    for line in decode(&foo, &itos).lines() {
        println!("{}", line);
    }
    // print!("{:?}", decode(&foo, &itos));




}
